import argparse
import asyncio
import json
import random
import time
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

RETRYABLE_STATUS = {429, 500, 503, 504}

# =========================
# 填充后的模板区
# =========================
SYSTEM_INSTRUCTION = """
"""

RESPONSE_SCHEMA = {}

USER_PROMPTS: List[str] = [
    r"""
任务：请对下面 TEXT 做结构化抽取，并严格按 JSON Schema 输出 JSON。
TEXT:
王小明于2025年12月17日与北京某科技公司签订劳动合同，月薪税前2.5万元，工作地点在北京朝阳区。合同期两年，试用期三个月。若提前离职需提前30天书面通知。
""",
    r"""
任务：请对下面 TEXT 做结构化抽取，并严格按 JSON Schema 输出 JSON。
TEXT:
The invoice INV-2025-001 was issued on Dec 30, 2025 by ACME Ltd. Total amount: USD 1,240.50. Due date: Jan 15, 2026. Billing address: 123 Market St, San Francisco.
""",
]
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


def build_config(
    temperature: float,
    max_output_tokens: int,
    top_p: float,
) -> types.GenerateContentConfig:
    """
    Build ONE shared config reused across all prompts.
    If SYSTEM_INSTRUCTION/RESPONSE_SCHEMA are empty, they are simply not applied.
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

    # structured output only when schema provided
    if isinstance(RESPONSE_SCHEMA, dict) and RESPONSE_SCHEMA:
        kwargs["response_mime_type"] = "application/json"
        # ✅ 关键改动：用 response_json_schema（dict JSON Schema）
        kwargs["response_json_schema"] = RESPONSE_SCHEMA

    return types.GenerateContentConfig(**kwargs)


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
            return {"ok": True, "text": resp.text}
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
):
    if not (1 <= len(prompts) <= 100):
        raise ValueError(f"USER_PROMPTS must contain 1~100 items, got {len(prompts)}")

    config = build_config(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
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

                out: Dict[str, Any] = {
                    "index": i,
                    "input": p,
                    **r,
                }

                if r.get("ok") and isinstance(r.get("text"), str) and RESPONSE_SCHEMA:
                    out["raw_text"] = r["text"]
                    try:
                        out["parsed"] = json.loads(r["text"])
                    except Exception as e:
                        out["parsed_error"] = f"{type(e).__name__}: {e}"

                return out

        tasks = [asyncio.create_task(one(i, p)) for i, p in enumerate(prompts)]
        with open(outfile, "w", encoding="utf-8") as f:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", required=True, help="Express mode API key")
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--outfile", default="results.jsonl")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--qps", type=float, default=5.0)  # 0=unlimited
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_output_tokens", type=int, default=65535)
    args = ap.parse_args()

    asyncio.run(
        run_batch(
            api_key=args.api_key,
            model=args.model,
            prompts=USER_PROMPTS,
            outfile=args.outfile,
            concurrency=args.concurrency,
            qps=args.qps,
            max_retries=args.max_retries,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            top_p=args.top_p,
        )
    )
    print(f"Done. Wrote: {args.outfile}")


if __name__ == "__main__":
    main()


