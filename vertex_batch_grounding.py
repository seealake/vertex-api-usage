# filename: vertex_batch_grounding.py
import argparse
import asyncio
import json
import random
import time
from typing import Any, Dict, List, Optional, Union

from google import genai
from google.genai import types

RETRYABLE_STATUS = {429, 500, 503, 504}


# =========================
# 系统指令（非 structured output）
# =========================
SYSTEM_INSTRUCTION = r"""
你可以使用 Google 搜索接地（grounding）工具来回答问题（如果已启用）。
要求：
- 尽量给出可靠来源；如果是通过搜索得到的结论，请在回答中体现出处/引用依据（如可用）。
- 不要编造不存在的来源；不确定就直说不确定。
- 回答用中文，条理清晰。
""".strip()


def safe_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """
    Try to get nested attrs/keys with multiple candidate key names.
    Example: safe_get(resp, "candidates", default=[])
    Example: safe_get(cand0, "grounding_metadata", "groundingMetadata", default=None)
    """
    cur = obj
    for k in keys:
        if cur is None:
            return default
        # object attribute
        if hasattr(cur, k):
            cur = getattr(cur, k)
            continue
        # dict key
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
            continue
        return default
    return cur


def get_status_code(err: Exception) -> Optional[int]:
    """
    Best-effort extraction of HTTP-ish status code from different exception types.
    """
    for attr in ("status_code", "code", "status"):
        v = getattr(err, attr, None)
        if isinstance(v, int):
            return v
    # sometimes nested: err.response.status_code
    resp = getattr(err, "response", None)
    if resp is not None:
        v = getattr(resp, "status_code", None)
        if isinstance(v, int):
            return v
    return None


def extract_text(resp: Any) -> str:
    """
    Extract text from response candidates content parts (best effort).
    """
    candidates = safe_get(resp, "candidates", default=[]) or []
    if not candidates:
        return ""

    cand0 = candidates[0]
    content = safe_get(cand0, "content", default=None)
    if content is None:
        return ""

    parts = safe_get(content, "parts", default=[]) or []
    texts: List[str] = []
    for p in parts:
        t = safe_get(p, "text", default=None)
        if isinstance(t, str) and t.strip():
            texts.append(t)
    return "\n".join(texts).strip()


def extract_grounding(resp: Any) -> Dict[str, Any]:
    """
    Extract grounding metadata:
      - web_search_queries
      - sources (from grounding_chunks[*].web)
    """
    candidates = safe_get(resp, "candidates", default=[]) or []
    if not candidates:
        return {"has_grounding": False, "web_search_queries": [], "sources": []}

    cand0 = candidates[0]
    gm = safe_get(cand0, "grounding_metadata", "groundingMetadata", default=None)
    if not gm:
        return {"has_grounding": False, "web_search_queries": [], "sources": []}

    queries = safe_get(gm, "web_search_queries", "webSearchQueries", default=[]) or []
    chunks = safe_get(gm, "grounding_chunks", "groundingChunks", default=[]) or []

    sources: List[Dict[str, Any]] = []
    for ch in chunks:
        web = safe_get(ch, "web", default=None)
        if not web:
            continue
        uri = safe_get(web, "uri", "url", default=None)
        title = safe_get(web, "title", default=None)
        # Some responses include "publisher", "description" etc. Keep minimal & stable.
        src = {}
        if isinstance(title, str) and title.strip():
            src["title"] = title.strip()
        if isinstance(uri, str) and uri.strip():
            src["url"] = uri.strip()
        if src:
            sources.append(src)

    # de-dup by url
    seen = set()
    uniq_sources = []
    for s in sources:
        u = s.get("url")
        if u and u in seen:
            continue
        if u:
            seen.add(u)
        uniq_sources.append(s)

    # normalize queries to strings
    norm_q: List[str] = []
    for q in queries:
        if isinstance(q, str):
            if q.strip():
                norm_q.append(q.strip())
        else:
            # sometimes query is object with "text"
            qt = safe_get(q, "text", default=None)
            if isinstance(qt, str) and qt.strip():
                norm_q.append(qt.strip())

    return {
        "has_grounding": True,
        "web_search_queries": norm_q,
        "sources": uniq_sources,
    }


class RateLimiter:
    """
    Simple global QPS limiter across concurrent tasks.
    Enforces minimum interval of 1/qps between requests.
    """

    def __init__(self, qps: float):
        self.qps = float(qps)
        self._lock = asyncio.Lock()
        self._next_time = 0.0

    async def wait(self) -> None:
        if self.qps <= 0:
            return
        interval = 1.0 / self.qps
        async with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                await asyncio.sleep(self._next_time - now)
                now = time.monotonic()
            self._next_time = now + interval


def load_prompts_from_file(path: str) -> List[str]:
    """
    Supports:
      - .txt: one prompt per line (empty lines ignored)
      - .json: {"prompts": ["...", "..."]} or ["...", "..."]
      - .jsonl: each line {"prompt": "..."} or {"text": "..."} or raw string
    """
    path_l = path.lower()
    if path_l.endswith(".txt"):
        out: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    out.append(s)
        return out

    if path_l.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x) for x in data if str(x).strip()]
        if isinstance(data, dict) and "prompts" in data and isinstance(data["prompts"], list):
            return [str(x) for x in data["prompts"] if str(x).strip()]
        raise ValueError('Unsupported JSON format. Use a list or {"prompts": [...]}')

    if path_l.endswith(".jsonl"):
        out: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                # try JSON first
                try:
                    obj = json.loads(s)
                    if isinstance(obj, str):
                        if obj.strip():
                            out.append(obj.strip())
                        continue
                    if isinstance(obj, dict):
                        p = obj.get("prompt") or obj.get("text") or obj.get("q") or obj.get("question")
                        if isinstance(p, str) and p.strip():
                            out.append(p.strip())
                        continue
                except Exception:
                    pass
                # fallback: raw line
                out.append(s)
        return out

    raise ValueError("Unsupported file type. Use .txt/.json/.jsonl")


async def generate_one(
    client: Any,
    model: str,
    prompt: str,
    *,
    enable_grounding: bool,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> Dict[str, Any]:
    """
    One request with retry.
    """
    cfg_kwargs: Dict[str, Any] = dict(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        system_instruction=SYSTEM_INSTRUCTION,
    )

    if enable_grounding:
        cfg_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

    config = types.GenerateContentConfig(**cfg_kwargs)

    last_err: Optional[str] = None
    t0 = time.monotonic()

    for attempt in range(max_retries + 1):
        try:
            await rate_limiter.wait()
            resp = await client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )

            text = extract_text(resp)
            grounding = extract_grounding(resp) if enable_grounding else {
                "has_grounding": False,
                "web_search_queries": [],
                "sources": [],
            }

            return {
                "ok": True,
                "prompt": prompt,
                "model": model,
                "enable_grounding": enable_grounding,
                "latency_s": round(time.monotonic() - t0, 4),
                "text": text,
                "grounding": grounding,
            }

        except Exception as err:
            sc = get_status_code(err)
            last_err = f"{type(err).__name__}: {err}"
            retryable = (sc in RETRYABLE_STATUS) if sc is not None else False

            if attempt >= max_retries or not retryable:
                return {
                    "ok": False,
                    "prompt": prompt,
                    "model": model,
                    "enable_grounding": enable_grounding,
                    "latency_s": round(time.monotonic() - t0, 4),
                    "error": last_err,
                    "status_code": sc,
                }

            # exponential backoff with jitter
            base = 0.8 * (2 ** attempt)
            sleep_s = base + random.uniform(0.0, 0.3)
            await asyncio.sleep(sleep_s)

    # should not reach
    return {
        "ok": False,
        "prompt": prompt,
        "model": model,
        "enable_grounding": enable_grounding,
        "latency_s": round(time.monotonic() - t0, 4),
        "error": last_err or "Unknown error",
    }


async def run_batch(
    *,
    api_key: str,
    model: str,
    prompts: List[str],
    outfile: str,
    concurrency: int,
    qps: float,
    max_retries: int,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    enable_grounding: bool,
) -> None:
    client = genai.Client(vertexai=True, api_key=api_key).aio

    sem = asyncio.Semaphore(max(1, int(concurrency)))
    rate_limiter = RateLimiter(qps)
    write_lock = asyncio.Lock()

    # ensure output dir exists if nested path
    # (keep simple; if outfile is just filename, this does nothing)
    import os
    outdir = os.path.dirname(outfile)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    async def worker(i: int, p: str) -> None:
        async with sem:
            result = await generate_one(
                client=client,
                model=model,
                prompt=p,
                enable_grounding=enable_grounding,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                max_retries=max_retries,
                rate_limiter=rate_limiter,
            )
            result["index"] = i
            # write jsonl line
            line = json.dumps(result, ensure_ascii=False)
            async with write_lock:
                with open(outfile, "a", encoding="utf-8") as f:
                    f.write(line + "\n")

    # fresh write
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("")

    tasks = [asyncio.create_task(worker(i, p)) for i, p in enumerate(prompts)]
    await asyncio.gather(*tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", required=True, help="Vertex Express mode API key")
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--outfile", default="results_grounding.jsonl")

    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--qps", type=float, default=5.0, help="0 = unlimited")
    ap.add_argument("--max_retries", type=int, default=5)

    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_output_tokens", type=int, default=8192)
    ap.add_argument("--disable_grounding", action="store_true", help="Disable Google Search grounding tool")

    ap.add_argument("--prompts_file", default="", help="Load prompts from .txt/.json/.jsonl (optional)")
    ap.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Single prompt (can be repeated). If provided, overrides default prompts unless --prompts_file is also set.",
    )

    args = ap.parse_args()

    enable_grounding = not args.disable_grounding

    # Default prompts (when no --prompts_file and no --prompt)
    prompts: List[str] = [
        "用一句话解释什么是 RAG，并给出一个可靠来源。",
        "最近一个月 NVIDIA 有哪些重要发布？请给出来源。",
    ]
    if args.prompt:
        prompts = [p.strip() for p in args.prompt if p and p.strip()]

    if args.prompts_file.strip():
        prompts = load_prompts_from_file(args.prompts_file.strip())

    if not prompts:
        raise ValueError("No prompts provided. Use --prompt or --prompts_file.")

    asyncio.run(
        run_batch(
            api_key=args.api_key,
            model=args.model,
            prompts=prompts,
            outfile=args.outfile,
            concurrency=args.concurrency,
            qps=args.qps,
            max_retries=args.max_retries,
            temperature=args.temperature,
            top_p=args.top_p,
            max_output_tokens=args.max_output_tokens,
            enable_grounding=enable_grounding,
        )
    )
    print(f"Done. Wrote: {args.outfile}")


if __name__ == "__main__":
    main()


