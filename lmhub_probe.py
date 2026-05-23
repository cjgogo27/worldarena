#!/usr/bin/env python3
"""LMHub/OpenAI-compatible endpoint probe.

Features:
1) Discover available models via /models
2) Smoke-test chat usability for a subset of models
3) Ramp-load test to estimate stable request rate
4) Output recommended settings for 5000 calls
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class CallResult:
    status: int
    latency_s: float
    headers: Dict[str, str]
    body: str
    error: str = ""


def normalize_base_url(base_url: str) -> str:
    value = (base_url or "").strip()
    if value.startswith("url:"):
        value = value[4:]
    return value.rstrip("/")


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    rank = (len(sorted_vals) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_vals[int(rank)]
    weight = rank - low
    return sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight


def http_json(
    method: str,
    url: str,
    api_key: str,
    timeout: float,
    payload: dict | None = None,
) -> CallResult:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = Request(url=url, data=data, headers=headers, method=method)
    start = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            latency = time.perf_counter() - start
            return CallResult(
                status=getattr(resp, "status", 200),
                latency_s=latency,
                headers={k.lower(): v for k, v in resp.headers.items()},
                body=body,
            )
    except HTTPError as err:
        latency = time.perf_counter() - start
        body = ""
        try:
            body = err.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        hdrs = {}
        if err.headers:
            hdrs = {k.lower(): v for k, v in err.headers.items()}
        return CallResult(
            status=err.code,
            latency_s=latency,
            headers=hdrs,
            body=body,
            error=f"HTTPError {err.code}",
        )
    except URLError as err:
        latency = time.perf_counter() - start
        return CallResult(
            status=-1,
            latency_s=latency,
            headers={},
            body="",
            error=f"URLError: {err}",
        )
    except Exception as err:  # pylint: disable=broad-except
        latency = time.perf_counter() - start
        return CallResult(
            status=-1,
            latency_s=latency,
            headers={},
            body="",
            error=f"Exception: {err}",
        )


def fetch_models(base_url: str, api_key: str, timeout: float) -> Tuple[List[str], CallResult]:
    result = http_json("GET", f"{base_url}/models", api_key=api_key, timeout=timeout)
    models: List[str] = []
    if 200 <= result.status < 300:
        try:
            payload = json.loads(result.body)
            for item in payload.get("data", []):
                model_id = item.get("id")
                if model_id:
                    models.append(str(model_id))
        except json.JSONDecodeError:
            pass
    return models, result


def chat_once(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> CallResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    return http_json(
        "POST",
        f"{base_url}/chat/completions",
        api_key=api_key,
        timeout=timeout,
        payload=payload,
    )


def check_model_subset(
    base_url: str,
    api_key: str,
    models: List[str],
    mode: str,
    interval_s: float,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> Tuple[List[str], List[Tuple[str, int]]]:
    if mode == "none":
        return [], []
    if mode == "first10":
        subset = models[:10]
    elif mode == "first20":
        subset = models[:20]
    else:
        subset = models

    ok_models: List[str] = []
    bad_models: List[Tuple[str, int]] = []

    for idx, model in enumerate(subset):
        if idx > 0 and interval_s > 0:
            time.sleep(interval_s)
        result = chat_once(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if 200 <= result.status < 300:
            ok_models.append(model)
        else:
            bad_models.append((model, result.status))

    return ok_models, bad_models


def parse_rpm_limit_from_429(body: str) -> int | None:
    if not body:
        return None

    patterns = [
        r"1分钟内最多请求\s*(\d+)\s*次",
        r"每分钟最多请求\s*(\d+)\s*次",
        r"(\d+)\s*requests?\s*per\s*minute",
        r"max\s*(\d+)\s*rpm",
    ]
    for pattern in patterns:
        match = re.search(pattern, body, flags=re.IGNORECASE)
        if match:
            try:
                value = int(match.group(1))
                if value > 0:
                    return value
            except Exception:
                continue
    return None


def recommendation_from_rpm_limit(rpm_limit: int, safety_factor: float, total_calls: int = 5000) -> dict:
    safe_rpm = max(1.0, rpm_limit * safety_factor)
    safe_rps = safe_rpm / 60.0
    eta_seconds = total_calls / safe_rps
    return {
        "detected_limit_rpm": rpm_limit,
        "suggested_operational_rpm": round(safe_rpm, 2),
        "suggested_operational_rps": round(safe_rps, 4),
        "suggested_concurrency": 1,
        "timeout_s": 45,
        "retry": {
            "max_retries": 6,
            "backoff": "exponential_with_jitter",
            "initial_delay_s": 2,
            "max_delay_s": 45,
            "retry_on": [429, 500, 502, 503, 504],
        },
        "batching": {
            "calls_per_batch": int(max(20, safe_rpm * 5)),
            "pause_between_batches_s": 30,
        },
        "estimated_total_calls": total_calls,
        "estimated_total_time_s": round(eta_seconds, 1),
        "estimated_total_time_human": format_seconds(eta_seconds),
    }


def run_load_test(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    target_rps: float,
    duration_s: float,
    worker_factor: float,
) -> dict:
    total_requests = max(1, int(target_rps * duration_s))
    workers = max(1, math.ceil(target_rps * worker_factor))
    latencies: List[float] = []
    statuses: List[int] = []
    errors: List[str] = []
    header_samples: Dict[str, str] = {}

    def do_one() -> CallResult:
        return chat_once(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for i in range(total_requests):
            fire_at = start + (i / target_rps)
            now = time.perf_counter()
            wait_s = fire_at - now
            if wait_s > 0:
                time.sleep(wait_s)
            futures.append(executor.submit(do_one))

        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            statuses.append(res.status)
            latencies.append(res.latency_s)
            if res.error:
                errors.append(res.error)
            if not header_samples and res.headers:
                header_samples = res.headers

    elapsed = max(time.perf_counter() - start, 1e-9)
    success = sum(1 for s in statuses if 200 <= s < 300)
    ratelimited = sum(1 for s in statuses if s == 429)
    server_errors = sum(1 for s in statuses if s >= 500)
    client_errors = sum(1 for s in statuses if 400 <= s < 500 and s != 429)
    transport_errors = sum(1 for s in statuses if s < 0)

    return {
        "target_rps": target_rps,
        "actual_rps": len(statuses) / elapsed,
        "workers": workers,
        "total": len(statuses),
        "success": success,
        "success_rate": success / max(len(statuses), 1),
        "ratelimited": ratelimited,
        "server_errors": server_errors,
        "client_errors": client_errors,
        "transport_errors": transport_errors,
        "lat_avg": statistics.mean(latencies) if latencies else 0.0,
        "lat_p95": percentile(latencies, 95),
        "header_samples": header_samples,
        "errors_sample": errors[:5],
    }


def format_seconds(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    mins, rem = divmod(int(sec), 60)
    if mins < 60:
        return f"{mins}m{rem}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h{mins}m{rem}s"


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe OpenAI-compatible endpoint")
    parser.add_argument("--base-url", default="https://lmhub.fatui.xyz/v1")
    parser.add_argument("--api-key", default=os.getenv("LMHUB_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--model", default="")
    parser.add_argument("--prompt", default="Reply with one short word: pong")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--check-models", choices=["none", "first10", "first20", "all"], default="first20")
    parser.add_argument("--model-check-interval", type=float, default=0.0)
    parser.add_argument("--list-models-limit", type=int, default=50)
    parser.add_argument("--rps-start", type=float, default=0.5)
    parser.add_argument("--rps-step", type=float, default=0.5)
    parser.add_argument("--rps-max", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--worker-factor", type=float, default=2.0)
    parser.add_argument("--stability-success-rate", type=float, default=0.99)
    parser.add_argument("--safety-factor", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if not args.api_key:
        print("[ERROR] Missing API key. Provide --api-key or set LMHUB_API_KEY/OPENAI_API_KEY.")
        return 2

    base_url = normalize_base_url(args.base_url)
    print(f"[INFO] Endpoint: {base_url}")

    models, models_resp = fetch_models(base_url=base_url, api_key=args.api_key, timeout=args.timeout)
    print(f"[INFO] /models status: {models_resp.status}, discovered models: {len(models)}")
    if models:
        to_show = models[: max(0, args.list_models_limit)]
        for m in to_show:
            print(f"  - {m}")
        hidden = len(models) - len(to_show)
        if hidden > 0:
            print(f"[INFO] ... {hidden} more models omitted (use --list-models-limit to adjust)")
    else:
        print("[WARN] No models discovered. Check endpoint/key/permissions.")
        if models_resp.body:
            print("[DEBUG] /models response snippet:")
            print(models_resp.body[:500])

    chat_ok, chat_bad = check_model_subset(
        base_url=base_url,
        api_key=args.api_key,
        models=models,
        mode=args.check_models,
        interval_s=args.model_check_interval,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    if args.check_models != "none":
        checked = len(chat_ok) + len(chat_bad)
        print(f"[INFO] Chat smoke test checked={checked}, usable={len(chat_ok)}, failed={len(chat_bad)}")
        if chat_ok:
            print("[INFO] Chat-usable models:")
            for m in chat_ok:
                print(f"  + {m}")
        if chat_bad:
            print("[INFO] Chat-failed models (status):")
            for m, s in chat_bad:
                print(f"  - {m}: {s}")
            if all(status == 429 for _, status in chat_bad):
                print(
                    "[WARN] All checked models got 429. You may be hitting per-minute request caps; "
                    "try --model-check-interval 8 to slow model checks."
                )

    model = args.model.strip()
    if not model:
        model = chat_ok[0] if chat_ok else (models[0] if models else "")
    if not model:
        print("[ERROR] No model available for load test.")
        return 3

    warmup = chat_once(
        base_url=base_url,
        api_key=args.api_key,
        model=model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    print(f"[INFO] Warmup model={model}, status={warmup.status}, latency={warmup.latency_s:.3f}s")
    if warmup.status < 200 or warmup.status >= 300:
        if warmup.status == 429:
            rpm_limit = parse_rpm_limit_from_429(warmup.body)
            print("[WARN] Warmup returned 429 (rate limited).")
            if warmup.body:
                print(warmup.body[:500])
            if rpm_limit:
                advice = recommendation_from_rpm_limit(
                    rpm_limit=rpm_limit,
                    safety_factor=min(max(args.safety_factor, 0.1), 0.95),
                    total_calls=5000,
                )
                print("\n[RESULT] Inferred stable config from 429 limit message")
                print(json.dumps(advice, ensure_ascii=False, indent=2))
                return 0
        print("[ERROR] Warmup failed, cannot continue load test.")
        if warmup.body:
            print(warmup.body[:500])
        return 4

    print("[INFO] Ramp load test starts...")
    rps = args.rps_start
    last_stable = None
    all_stats: List[dict] = []
    while rps <= args.rps_max + 1e-9:
        stats = run_load_test(
            base_url=base_url,
            api_key=args.api_key,
            model=model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            target_rps=rps,
            duration_s=args.duration,
            worker_factor=args.worker_factor,
        )
        all_stats.append(stats)

        stable = (
            stats["ratelimited"] == 0
            and stats["transport_errors"] == 0
            and stats["server_errors"] == 0
            and stats["success_rate"] >= args.stability_success_rate
        )

        print(
            "[LOAD] "
            f"target_rps={stats['target_rps']:.2f} actual_rps={stats['actual_rps']:.2f} "
            f"ok={stats['success']}/{stats['total']} ({stats['success_rate']*100:.1f}%) "
            f"429={stats['ratelimited']} 5xx={stats['server_errors']} net={stats['transport_errors']} "
            f"p95={stats['lat_p95']:.3f}s stable={stable}"
        )

        if stable:
            last_stable = stats
            rps += args.rps_step
        else:
            break

    if not all_stats:
        print("[ERROR] No load-test stats collected.")
        return 5

    if last_stable is None:
        print("[WARN] No stable point found. Use very conservative settings (e.g. 0.2 rps, single worker).")
        return 0

    recommended_rps = max(0.1, last_stable["target_rps"] * args.safety_factor)
    p95 = max(last_stable["lat_p95"], 0.2)
    recommended_concurrency = max(1, math.ceil(recommended_rps * p95 * 1.5))
    total_calls = 5000
    eta_seconds = total_calls / recommended_rps

    print("\n[RESULT] Stable limit estimate")
    print(f"  model: {model}")
    print(f"  last_stable_target_rps: {last_stable['target_rps']:.2f}")
    print(f"  suggested_operational_rps (~70%): {recommended_rps:.2f}")
    print(f"  suggested_concurrency: {recommended_concurrency}")
    print(f"  p95_latency_at_stable: {last_stable['lat_p95']:.3f}s")
    print(f"  ETA for 5000 calls: {format_seconds(eta_seconds)}")

    advice = {
        "base_url": base_url,
        "model": model,
        "target_rps": round(recommended_rps, 2),
        "concurrency": recommended_concurrency,
        "timeout_s": max(30, int(math.ceil(4 * p95))),
        "retry": {
            "max_retries": 6,
            "backoff": "exponential_with_jitter",
            "initial_delay_s": 1,
            "max_delay_s": 30,
            "retry_on": [429, 500, 502, 503, 504],
        },
        "batching": {
            "calls_per_batch": 100,
            "pause_between_batches_s": 2,
        },
        "estimated_total_calls": total_calls,
        "estimated_total_time_s": round(eta_seconds, 1),
    }
    print("\n[RESULT] Suggested config for stable 5000 calls")
    print(json.dumps(advice, ensure_ascii=False, indent=2))

    limit_headers = {
        k: v
        for k, v in warmup.headers.items()
        if "ratelimit" in k or "x-ratelimit" in k
    }
    if limit_headers:
        print("\n[INFO] Rate-limit headers from warmup response")
        print(json.dumps(limit_headers, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
