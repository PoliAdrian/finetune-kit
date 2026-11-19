#!/usr/bin/env python3
"""
04_serve_local.py
-----------------
Local FastAPI server running your fine-tuned GGUF model on CPU via
`llama-cpp-python` with per-request dynamic JSON Schema constrained decoding.

Zero cloud API calls, zero API cost, <1s latency on a laptop CPU.

Usage:
  MODEL_PATH=models/gguf/unsloth.Q8_0.gguf uvicorn scripts.04_serve_local:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

SCHEMA_PATH = Path(os.getenv("SCHEMA_PATH", "schema/task_schema.json"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/gguf/unsloth.Q8_0.gguf"))
CPU_THREADS = int(os.getenv("CPU_THREADS", "8"))

SYSTEM_PROMPT_TEMPLATE = (
    "You are a fast, strict JSON router. Convert the user's request into a single JSON call "
    "matching the schema: {{action, target, filters: {{floor, zone}}}}.\n"
    "Valid actions: count_elements, list_elements, measure_area, check_compliance, unsupported.\n"
    "Use ONLY values from the available context menu below (or null if not specified).\n"
    "Context menu:\n"
    "- targets: {targets}\n"
    "- floors: {floors}\n"
    "- zones: {zones}\n"
    "Respond with JSON only."
)

state: dict[str, Any] = {"llm": None, "base_schema": None}


def build_dynamic_schema(
    base_schema: dict[str, Any],
    targets: list[str],
    floors: list[str],
    zones: list[str],
) -> dict[str, Any]:
    schema = json.loads(json.dumps(base_schema))
    schema["properties"]["target"]["enum"] = sorted(set(targets + ["none"]))
    schema["properties"]["filters"]["properties"]["floor"] = {
        "anyOf": [{"type": "string", "enum": sorted(set(floors))}, {"type": "null"}]
    }
    schema["properties"]["filters"]["properties"]["zone"] = {
        "anyOf": [{"type": "string", "enum": sorted(set(zones))}, {"type": "null"}]
    }
    return schema


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["base_schema"] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if MODEL_PATH.exists():
        from llama_cpp import Llama

        print(f"Loading GGUF model from {MODEL_PATH} ({CPU_THREADS} CPU threads)...")
        state["llm"] = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=1024,
            n_threads=CPU_THREADS,
            verbose=False,
        )
    else:
        print(f"[warn] MODEL_PATH not found: {MODEL_PATH}. Server started without model loaded.")
    yield
    state["llm"] = None


app = FastAPI(title="Finetune Kit Local Router", lifespan=lifespan)


class RouteRequest(BaseModel):
    query: str = Field(..., examples=["¿Cuántas ventanas hay en la planta 1?"])
    targets: list[str] = Field(default_factory=lambda: ["window", "door", "toilet", "balcony"])
    floors: list[str] = Field(default_factory=lambda: ["ground", "floor_1", "floor_2"])
    zones: list[str] = Field(default_factory=lambda: ["apt_A", "apt_B", "lobby"])


def execute_deterministic_action(call: dict[str, Any]) -> dict[str, Any]:
    """
    Plug your real hackathon backend here (SQL query, CAD/DXF parser, YOLO detector,
    smart-home API, robot command, etc.).
    """
    if call.get("action") == "unsupported":
        return {"status": "unsupported", "message": "Request is outside supported actions."}
    return {
        "status": "ok",
        "executed_action": call["action"],
        "target": call["target"],
        "filters": call["filters"],
        "note": "Replace execute_deterministic_action() with your real deterministic tool/API.",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_loaded": state["llm"] is not None,
        "model_path": str(MODEL_PATH),
        "threads": CPU_THREADS,
    }


@app.post("/route")
def route_query(req: RouteRequest) -> dict[str, Any]:
    llm = state["llm"]
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail=f"GGUF model not loaded. Place your model at '{MODEL_PATH}' or set MODEL_PATH.",
        )

    system_msg = SYSTEM_PROMPT_TEMPLATE.format(
        targets=", ".join(req.targets),
        floors=", ".join(req.floors),
        zones=", ".join(req.zones),
    )
    dyn_schema = build_dynamic_schema(
        state["base_schema"], req.targets, req.floors, req.zones
    )

    t0 = time.perf_counter()
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": req.query},
        ],
        temperature=0.0,
        max_tokens=128,
        response_format={"type": "json_object", "schema": dyn_schema},
    )
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    parsed_call = json.loads(out["choices"][0]["message"]["content"])
    execution = execute_deterministic_action(parsed_call)

    return {
        "call": parsed_call,
        "execution": execution,
        "latency_ms": latency_ms,
        "provider": "local-gguf-cpu",
    }
