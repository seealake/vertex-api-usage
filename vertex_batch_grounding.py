# filename: vertex_batch_grounded.py
import argparse
import asyncio
import json
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

RETRYABLE_STATUS = {429, 500, 503, 504}


# =========================
# 可选：系统指令（不要求 structured output）
# =========================
SYSTEM_INSTRUCTION = r"""
你将启用 Google 搜索接地工具来回答问题。
要求：
1) 尽量基于搜索结果回答，不要编造。
2) 回答要清晰、简洁；如果信息不足请明确说明不确定点。
3) 不要输出任何 JSON 或代码块（除非用户明确要求）。
"""
# =========================


def get_status_code(err: Exception) -> Optional[int]:
    return getattr(err, "status_code", None)


def backoff(attempt: int, base: float = 0.8, cap: float = 20.0) -> float:
    delay = min(cap, base * (2 ** attempt))
    return delay + random.uniform(0, 0.5)


class RateLimiter:
    """Global QPS limiter: at most qps requests/sec."""
    def __init__(self, qps: float):
        self.qps = qps
        self._lock = asyncio.Lock()
        self._next_time = 0.0

    async def acquire(self):
        if self.qps <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._next_time - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_time = max(self._next_time, now) + 1.0 / self.qps


def safe_get(obj: Any, *names: str, default=None):
    """Try multiple attribute names (snake/camel) on SDK objects."""
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return default


def build_config(
    temperature: float,
    max_output_tokens: int,
    top_p: float,
    enable_grounding: bool,
    use_legacy_grounding_tool: bool,
) -> types.GenerateContentConfig:
    """
    Build ONE shared config reused across all prompts.

    Grounding tool:
      - Recommended (Gemini 2.0+ / 2.5 / 3): google_search
      - Legacy (Gemini 1.5): google_search_retrieval with dynamic config
    """
    if not (0.0 <= top_p <= 1.0):
        raise ValueError("top_p must be within [0, 1].")

    kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "top_p": top_p,
    }

    if SYSTEM_INSTRUCTION.strip():
        kwargs["system_instruction"] = [SYSTEM_INSTRUCTION]

    if enable_grounding:
        if use_legacy_grounding_tool:
            # Legacy for Gemini 1.5: google_search_retrieval
            # (kept here for compatibility if you ever switch to 1.5 models)
            retrieval_tool = types.Tool(
                google_search_retrieval=types.GoogleSearchRetrieval(
                    dynamic_retrieval_config=types.DynamicRetrievalConfig(
                        mode=types.DynamicRetrievalConfigMode.MODE_DYNAMIC,
                        dynamic_threshold=0.7,
                    )
                )
            )
            kwargs["tools"] = [retrieval_tool]
        else:
            # Recommended for current models: google_search
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            kwargs["tools"] = [grounding_tool]

    return types.GenerateContentConfig(**kwargs)


def extract_grounding_metadata(resp: Any) -> Dict[str, Any]:
    """
    Extract queries + sources from response.candidates[0].grounding_metadata (if present).
    """
    candidates = safe_get(resp, "candidates", default=[])
    if not candidates:
        return {"has_grounding": False, "web_search_queries": [], "sources": []}

    cand0 = candidates[0]
    gm = safe_get(cand0, "grounding_metadata", "groundingMetadata", default=None)
    if not gm:
        return {"has_grounding": False, "web_search_queries": [], "sources": []}

    queries = safe_get(gm, "web_search_queries", "webSearchQueries", default=[]) or []
    chunks = safe_get(gm, "grounding_chunks", "groundingChunks", default=[]) or []

    sources: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        web = safe_get(ch, "web", default=None)
        if not web:
            continue
        uri = safe_get(web, "uri", default=None)
        title = safe_get(web, "title", default=None)
        if uri or title:
            sources.append({"index": i + 1, "title": title, "uri": uri})

    return {
        "has_grounding": True,
        "web_search_queries": list(queries),
        "sources": sources,
    }


def add_inline_citations(resp: Any) -> Optional[str]:
    """
    Insert markdown citations into resp.text using groundingSupports + groundingChunks.
    Mirrors the pattern shown in official docs.
    """
    text = safe_get(resp, "text", default=None)
    if not isinstance(text, str) or not text:
        return None

    candidates = safe_get(resp, "candidates", default=[])
    if not candidates:
        return None

    gm = safe_get(candidates[0], "grounding_metadata", "groundingMetadata", default=None)
    if not gm:
        return None

    supports = safe_get(gm, "grounding_supports", "groundingSupports", default=[]) or []
    chunks = safe_get(gm, "grounding_chunks", "groundingChunks", default=[]) or []

    if not supports or not chunks:
        return None

    # Sort supports by end index descending (avoid shifting indices when inserting)
    def end_index_of_support(s) -> int:
        seg = safe_get(s, "segment", default=None)
        if not seg:
            return -1
        end_idx = safe_get(seg, "end_index", "endIndex", default=-1)
        try:
            return int(end_idx)
        except Exception:
            return -1

    sorted_supports = sorted(supports, key=end_index_of_support, reverse=True)

    out = text
    for s in sorted_supports:
        seg = safe_get(s, "segment", default=None)
        if not seg:
            continue
        end_idx = safe_get(seg, "end_index", "endIndex", default=None)
        if end_idx is None:
            continue
        try:
            end_idx = int(end_idx)
        except Exception:
            continue

        idxs = safe_get(s, "grounding_chunk_indices", "groundingChunkIndices", default=[]) or []
        if not idxs:
            continue

        links: List[str] = []
        for i in idxs:
            try:
                i = int(i)
            except Exception:
                continue
            if 0 <= i < len(chunks):
                web = safe_get(chunks[i], "web", default=None)
                uri = safe_get(web, "uri", default=None) if web else None
                if uri:
                    # Use 1-based display index
                    links.append(f"[{i + 1}]({uri})")
        if not links:
            continue

        citation_str = ", ".join(links)
        if 0 <= end_idx <= len(out):
            out = out[:end_idx] + citation_str + out[end_idx:]

    return out


async def call_with_retry(
    aclient: Any,
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
    limiter: RateLimiter,
    max_retries: int,
) -> Dict[str, Any]:
    last_err: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            await limiter.acquire()
            resp = await aclient.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            grounded_text = add_inline_citations(resp)
            grounding = extract_grounding_metadata(resp)

            return {
                "ok": True,
                "text": resp.text,
                "text_with_citations": grounded_text,
                **grounding,
            }
        except Exception as e:
            code = get_status_code(e)
            last_err = f"{type(e).__name__}: {e}"
            if code in RETRYABLE_STATUS and attempt < max_retries:
                await asyncio.sleep(backoff(attempt))
                continue
            return {"ok": False, "error": last_err, "status_code": code}
    return {"ok": False, "error": last_err or "Unknown error"}


async def run_batch(
    api_key: str,
    model: str,
    prompts: List[str],
    outfile: str,
    concurrency: int,
    qps: float,
    max_retries: int,
    temperature: float,
    max_output_tokens: int,
    top_p: float,
    enable_grounding: bool,
    use_legacy_grounding_tool: bool,
):
    if not (1 <= len(prompts) <= 100):
        raise ValueError(f"USER_PROMPTS must contain 1~100 items, got {len(prompts)}")

    config = build_config(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        enable_grounding=enable_grounding,
        use_legacy_grounding_tool=use_legacy_grounding_tool,
    )
    limiter = RateLimiter(qps=qps)
    sem = asyncio.Semaphore(concurrency)

    async with genai.Client(vertexai=True, api_key=api_key).aio as aclient:
        async def one(i: int, p: str) -> Dict[str, Any]:
            async with sem:
                r = await call_with_retry(
                    aclient=aclient,
                    model=model,
                    contents=p,
                    config=config,
                    limiter=limiter,
                    max_retries=max_retries,
                )
                return {"index": i, "input": p, **r}

        tasks = [asyncio.create_task(one(i, p)) for i, p in enumerate(prompts)]
        with open(outfile, "w", encoding="utf-8") as f:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()


def load_prompts_from_file(path: str) -> List[str]:
    """
    Load prompts from a text file.
    Supports:
      - .txt: one prompt per line (empty lines ignored)
      - .json: {"prompts": ["...", "..."]} or ["...", "..."]
      - .jsonl: each line {"prompt": "..."} or {"text": "..."} or raw string
    """
    if path.lower().endswith(".txt"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    out.append(s)
        return out

    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [str(x) for x in obj if str(x).strip()]
        if isinstance(obj, dict) and "prompts" in obj and isinstance(obj["prompts"], list):
            return [str(x) for x in obj["prompts"] if str(x).strip()]
        raise ValueError("Unsupported JSON format. Use a list or {\"prompts\": [...]}")

    if path.lower().endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        p = obj.get("prompt") or obj.get("text")
                        if p and str(p).strip():
                            out.append(str(p))
                    elif isinstance(obj, str) and obj.strip():
                        out.append(obj)
                except Exception:
                    # fallback: treat as raw prompt line
                    out.append(line)
        return out

    raise ValueError("Unsupported file type. Use .txt/.json/.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", required=True, help="Express mode API key")
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--outfile", default="results_grounded.jsonl")

    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--qps", type=float, default=5.0)  # 0=unlimited
    ap.add_argument("--max_retries", type=int, default=5)

    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_output_tokens", type=int, default=8192)

    ap.add_argument("--enable_grounding", action="store_true", help="Enable Google Search grounding tool")
    ap.add_argument(
        "--legacy_grounding_tool",
        action="store_true",
        help="Use legacy google_search_retrieval tool (for Gemini 1.5 models)",
    )

    ap.add_argument("--prompts_file", default="", help="Load prompts from .txt/.json/.jsonl (optional)")
    args = ap.parse_args()

    # Default prompts (if you don't pass --prompts_file)
    prompts: List[str] = [
        "最近一周OpenAI有什么重要发布？请给出来源。",
        "Euro 2024 的冠军是谁？给出来源。",
    ]
    if args.prompts_file.strip():
        prompts = load_prompts_from_file(args.prompts_file.strip())

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
            max_output_tokens=args.max_output_tokens,
            top_p=args.top_p,
            enable_grounding=args.enable_grounding,
            use_legacy_grounding_tool=args.legacy_grounding_tool,
        )
    )
    print(f"Done. Wrote: {args.outfile}")


if __name__ == "__main__":
    main()


