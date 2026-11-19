#!/usr/bin/env python3
"""
03_evaluate.py
--------------
Evaluates a local GGUF model (base or fine-tuned) on `data/test.jsonl` using
`llama-cpp-python` with dynamic per-request JSON Schema constrained decoding,
and prints a Markdown benchmark table ready for hackathon pitch slides.

Usage:
  python scripts/03_evaluate.py \
      --gguf models/gguf/unsloth.Q8_0.gguf \
      --label "Fine-tuned FunctionGemma 270M (CPU)" \
      --test-file data/test.jsonl \
      --limit 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any


def build_dynamic_schema(
    base_schema: dict[str, Any],
    targets: list[str],
    floors: list[str],
    zones: list[str],
) -> dict[str, Any]:
    """
    Restrict JSON Schema enums dynamically to ONLY the items present in this
    request's context menu. Prevents hallucinated labels at decoding time!
    """
    schema = json.loads(json.dumps(base_schema))
    schema["properties"]["target"]["enum"] = sorted(set(targets + ["none"]))
    schema["properties"]["filters"]["properties"]["floor"] = {
        "anyOf": [{"type": "string", "enum": sorted(set(floors))}, {"type": "null"}]
    }
    schema["properties"]["filters"]["properties"]["zone"] = {
        "anyOf": [{"type": "string", "enum": sorted(set(zones))}, {"type": "null"}]
    }
    return schema


def normalize_call(obj: dict[str, Any]) -> dict[str, Any]:
    filters = obj.get("filters") or {}
    return {
        "action": obj.get("action"),
        "target": obj.get("target"),
        "filters": {
            "floor": filters.get("floor"),
            "zone": filters.get("zone"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a local GGUF model on test.jsonl.")
    parser.add_argument("--gguf", type=Path, required=True, help="Path to .gguf model file.")
    parser.add_argument("--label", type=str, default="Local GGUF Model", help="Name for benchmark table.")
    parser.add_argument("--test-file", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--schema", type=Path, default=Path("schema/task_schema.json"))
    parser.add_argument("--limit", type=int, default=200, help="Max test examples to run (0 = all).")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for llama.cpp.")
    parser.add_argument("--no-grammar", action="store_true", help="Disable JSON schema constrained decoding.")
    parser.add_argument("--show-errors", type=int, default=5, help="Print up to N failure cases.")
    args = parser.parse_args()

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise SystemExit(
            "llama-cpp-python is not installed. Run: pip install llama-cpp-python"
        ) from exc

    base_schema = json.loads(args.schema.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.test_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"Loading {args.gguf} on CPU ({args.threads} threads)...")
    llm = Llama(
        model_path=str(args.gguf),
        n_ctx=1024,
        n_threads=args.threads,
        verbose=False,
    )

    valid_json = 0
    exact_match = 0
    action_match = 0
    target_match = 0
    latencies_ms: list[float] = []
    errors: list[dict[str, Any]] = []

    for row in rows:
        messages = row["messages"][:2]  # system + user (hide assistant ground truth)
        meta = row["meta"]
        expected = normalize_call(meta["expected"])

        response_format = None
        if not args.no_grammar:
            dyn_schema = build_dynamic_schema(
                base_schema,
                targets=meta["targets"],
                floors=meta["floors"],
                zones=meta["zones"],
            )
            response_format = {"type": "json_object", "schema": dyn_schema}

        t0 = time.perf_counter()
        out = llm.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=128,
            response_format=response_format,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        raw = out["choices"][0]["message"]["content"].strip()
        try:
            pred = normalize_call(json.loads(raw))
            valid_json += 1
        except Exception:
            pred = None

        if pred is not None:
            if pred["action"] == expected["action"]:
                action_match += 1
            if pred["target"] == expected["target"]:
                target_match += 1
            if pred == expected:
                exact_match += 1
            elif len(errors) < args.show_errors:
                errors.append({"query": messages[1]["content"], "expected": expected, "got": pred})
        elif len(errors) < args.show_errors:
            errors.append({"query": messages[1]["content"], "expected": expected, "got": raw})

    n = len(rows)
    mean_ms = statistics.mean(latencies_ms) if latencies_ms else 0.0
    p95_ms = sorted(latencies_ms)[int(0.95 * (n - 1))] if latencies_ms else 0.0

    print("\n### Benchmark Results (copy straight into your pitch slides!)\n")
    print("| Model | Samples | Valid JSON | Exact Match | Action Acc | Target Acc | Avg Latency | P95 Latency | API Cost |")
    print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    print(
        f"| {args.label} | {n} | "
        f"{valid_json/n*100:.1f}% | "
        f"**{exact_match/n*100:.1f}%** ({exact_match}/{n}) | "
        f"{action_match/n*100:.1f}% | "
        f"{target_match/n*100:.1f}% | "
        f"{mean_ms:.0f} ms | "
        f"{p95_ms:.0f} ms | "
        f"$0.00 |"
    )

    if errors:
        print(f"\n### Sample Errors (first {len(errors)}):")
        for e in errors:
            print(f"- Query:    {e['query']}")
            print(f"  Expected: {e['expected']}")
            print(f"  Got:      {e['got']}")


if __name__ == "__main__":
    main()
