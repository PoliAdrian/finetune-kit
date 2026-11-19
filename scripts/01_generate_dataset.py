#!/usr/bin/env python3
"""
01_generate_dataset.py
----------------------
Generates a synthetic fine-tuning dataset (train/val/test JSONL) for small
function-calling / intent-routing models (e.g., FunctionGemma 270M, Gemma 1B).

Design pattern (inspired by Junction 2026 winning project Blueline):
  1. Inject a compact "context menu" in the system/developer message listing
     valid targets, floors, and zones for the current document/session.
  2. Pair diverse natural-language user queries (EN + ES, casual, typos) with
     strict JSON calls validated against `schema/task_schema.json`.
  3. Optionally enrich a subset of queries using any OpenAI-compatible LLM API
     (Gemini, OpenRouter, Groq, Ollama) for extra linguistic diversity.

Usage:
  python scripts/01_generate_dataset.py --num-samples 2000 --out-dir data
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

# Optional: only required if you pass --paraphrase-count > 0
try:
    import httpx
except ImportError:
    httpx = None

try:
    import jsonschema
except ImportError:
    jsonschema = None


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

# Domain vocabulary (swap this block in 5 minutes for any new hackathon!)
ALL_TARGETS = [
    ("window", ["window", "windows", "ventana", "ventanas", "ventanales"]),
    ("door", ["door", "doors", "puerta", "puertas"]),
    ("fire_exit", ["fire exit", "emergency exit", "salida de emergencia", "salidas de incendios"]),
    ("toilet", ["toilet", "wc", "inodoro", "inodoros", "baño"]),
    ("sink", ["sink", "sinks", "lavabo", "lavabos"]),
    ("balcony", ["balcony", "balconies", "balcón", "balcones", "terraza"]),
    ("kitchen", ["kitchen", "kitchens", "cocina", "cocinas"]),
    ("bedroom", ["bedroom", "bedrooms", "dormitorio", "dormitorios", "habitación", "habitaciones"]),
    ("sauna", ["sauna", "saunas"]),
    ("elevator", ["elevator", "lift", "ascensor", "ascensores"]),
]

ALL_FLOORS = [
    ("ground", ["ground floor", "planta baja", "PB", "nivel 0"]),
    ("floor_1", ["first floor", "1st floor", "floor 1", "primera planta", "planta 1", "piso 1"]),
    ("floor_2", ["second floor", "2nd floor", "floor 2", "segunda planta", "planta 2", "piso 2"]),
    ("floor_3", ["third floor", "3rd floor", "floor 3", "tercera planta", "planta 3"]),
    ("roof", ["roof", "rooftop", "azotea", "cubierta"]),
]

ALL_ZONES = [
    ("apt_A", ["apartment A", "apt A", "piso A", "vivienda A"]),
    ("apt_B", ["apartment B", "apt B", "piso B", "vivienda B"]),
    ("north_wing", ["north wing", "ala norte", "zona norte"]),
    ("south_wing", ["south wing", "ala sur", "zona sur"]),
    ("lobby", ["lobby", "recepción", "entrada", "hall"]),
]

TEMPLATES: dict[str, list[str]] = {
    "count_elements": [
        "How many {target} are there{loc}?",
        "Count all {target}{loc}",
        "Total number of {target}{loc}?",
        "give me the count of {target}{loc}",
        "¿Cuántos {target} hay{loc}?",
        "cuantas {target} tiene{loc}",
        "dime el total de {target}{loc}",
        "numero de {target}{loc}",
    ],
    "list_elements": [
        "Show me all {target}{loc}",
        "Highlight every {target}{loc}",
        "Where are the {target}{loc}?",
        "Muéstrame todos los {target}{loc}",
        "enseñame donde estan las {target}{loc}",
        "marca los {target}{loc} en el plano",
    ],
    "measure_area": [
        "What is the total area of {target}{loc}?",
        "Measure the square meters of {target}{loc}",
        "How big is the {target}{loc}?",
        "¿Cuál es el área de {target}{loc}?",
        "cuantos m2 tiene {target}{loc}",
        "superficie total de {target}{loc}",
    ],
    "check_compliance": [
        "Do the {target}{loc} meet building code?",
        "Check compliance for {target}{loc}",
        "Verify regulations on {target}{loc}",
        "¿Cumplen normativa los {target}{loc}?",
        "revisa si {target}{loc} pasa la inspección",
    ],
}

UNSUPPORTED_QUERIES = [
    "Who is the architect of this building?",
    "What will the weather be tomorrow in Helsinki?",
    "Write a poem about concrete",
    "Estimate the mortgage interest rate for 30 years",
    "¿Cuánto cuesta el metro cuadrado en Madrid?",
    "Tradúceme esto al japonés",
    "Explícame cómo funciona un reactor nuclear",
    "Can you order pizza for the team?",
]


def inject_light_noise(text: str, rng: random.Random) -> str:
    """Occasionally simulate casual hackathon/user typing (lowercase, drop punctuation)."""
    r = rng.random()
    if r < 0.25:
        text = text.lower()
    if r < 0.20:
        text = text.rstrip("?").lstrip("¿")
    return text.strip()


def build_location_phrase(
    floor_surface: str | None, zone_surface: str | None, is_spanish: bool, rng: random.Random
) -> str:
    parts = []
    if floor_surface:
        prep = "en" if is_spanish else rng.choice(["on", "in"])
        parts.append(f"{prep} {floor_surface}")
    if zone_surface:
        prep = "en" if is_spanish else "in"
        parts.append(f"{prep} {zone_surface}")
    return (" " + " ".join(parts)) if parts else ""


def generate_one_sample(rng: random.Random) -> dict[str, Any]:
    # Pick a random subset of targets/floors/zones as this drawing's "context menu"
    menu_targets = rng.sample(ALL_TARGETS, k=rng.randint(4, 7))
    menu_floors = rng.sample(ALL_FLOORS, k=rng.randint(2, 4))
    menu_zones = rng.sample(ALL_ZONES, k=rng.randint(2, 4))

    system_msg = SYSTEM_PROMPT_TEMPLATE.format(
        targets=", ".join(t[0] for t in menu_targets),
        floors=", ".join(f[0] for f in menu_floors),
        zones=", ".join(z[0] for z in menu_zones),
    )

    # 10% out-of-scope / unsupported queries
    if rng.random() < 0.10:
        user_msg = inject_light_noise(rng.choice(UNSUPPORTED_QUERIES), rng)
        expected = {
            "action": "unsupported",
            "target": "none",
            "filters": {"floor": None, "zone": None},
        }
    else:
        action = rng.choice(list(TEMPLATES.keys()))
        target_id, target_syns = rng.choice(menu_targets)
        template = rng.choice(TEMPLATES[action])
        is_spanish = any(w in template.lower() for w in ["cuánt", "cuant", "dime", "muéstr", "enseñ", "marca", "área", "superficie", "cumplen", "revisa", "numero"])

        # Pick matching-language synonym when possible
        target_surface = rng.choice(target_syns)

        floor_id, floor_surface = None, None
        if rng.random() < 0.45:
            floor_id, floor_syns = rng.choice(menu_floors)
            floor_surface = rng.choice(floor_syns)

        zone_id, zone_surface = None, None
        if rng.random() < 0.35:
            zone_id, zone_syns = rng.choice(menu_zones)
            zone_surface = rng.choice(zone_syns)

        loc = build_location_phrase(floor_surface, zone_surface, is_spanish, rng)
        user_msg = inject_light_noise(template.format(target=target_surface, loc=loc), rng)
        expected = {
            "action": action,
            "target": target_id,
            "filters": {"floor": floor_id, "zone": zone_id},
        }

    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": json.dumps(expected, ensure_ascii=False)},
        ],
        "meta": {
            "targets": [t[0] for t in menu_targets],
            "floors": [f[0] for f in menu_floors],
            "zones": [z[0] for z in menu_zones],
            "expected": expected,
        },
    }


def paraphrase_with_llm(
    samples: list[dict[str, Any]],
    count: int,
    api_url: str,
    api_key: str,
    model: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Optional: rewrite a subset of user prompts with a stronger LLM for natural diversity."""
    if count <= 0:
        return []
    if httpx is None:
        raise RuntimeError("Install httpx (`pip install httpx`) to use --paraphrase-count.")

    chosen = rng.sample(samples, k=min(count, len(samples)))
    augmented: list[dict[str, Any]] = []
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    with httpx.Client(timeout=30.0) as client:
        for item in chosen:
            orig_q = item["messages"][1]["content"]
            prompt = (
                "Rewrite the following user request in a natural, slightly different way "
                "(casual tone, shorthand, or minor typo) while preserving the EXACT meaning, "
                "entities, floor, and zone. Return ONLY the rewritten sentence.\n\n"
                f"Original: {orig_q}"
            )
            try:
                resp = client.post(
                    f"{api_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.9,
                        "max_tokens": 80,
                    },
                )
                resp.raise_for_status()
                rewritten = resp.json()["choices"][0]["message"]["content"].strip().strip('"')
                if rewritten:
                    new_item = json.loads(json.dumps(item))
                    new_item["messages"][1]["content"] = rewritten
                    augmented.append(new_item)
            except Exception as exc:
                print(f"[warn] Paraphrase failed for '{orig_q}': {exc}")

    return augmented


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic fine-tuning dataset.")
    parser.add_argument("--num-samples", type=int, default=2000, help="Base deterministic samples.")
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="Output directory.")
    parser.add_argument("--schema", type=Path, default=Path("schema/task_schema.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    # Optional LLM paraphrasing (OpenAI-compatible endpoint)
    parser.add_argument("--paraphrase-count", type=int, default=0)
    parser.add_argument("--paraphrase-api-url", type=str, default=os.getenv("LLM_API_URL", ""))
    parser.add_argument("--paraphrase-api-key", type=str, default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--paraphrase-model", type=str, default=os.getenv("LLM_MODEL", "gemini-2.5-flash"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    schema = json.loads(args.schema.read_text(encoding="utf-8")) if args.schema.exists() else None

    samples = [generate_one_sample(rng) for _ in range(args.num_samples)]

    if args.paraphrase_count > 0:
        if not args.paraphrase_api_url or not args.paraphrase_api_key:
            raise SystemExit("Set --paraphrase-api-url and --paraphrase-api-key (or LLM_API_URL / LLM_API_KEY).")
        print(f"Paraphrasing {args.paraphrase_count} samples via {args.paraphrase_model}...")
        samples.extend(
            paraphrase_with_llm(
                samples,
                count=args.paraphrase_count,
                api_url=args.paraphrase_api_url,
                api_key=args.paraphrase_api_key,
                model=args.paraphrase_model,
                rng=rng,
            )
        )

    # Validate every generated JSON label against the schema
    if schema is not None and jsonschema is not None:
        for idx, s in enumerate(samples):
            jsonschema.validate(instance=s["meta"]["expected"], schema=schema)
        print(f"Schema validation passed for all {len(samples)} samples.")

    rng.shuffle(samples)
    n_test = int(len(samples) * args.test_ratio)
    n_val = int(len(samples) * args.val_ratio)
    test_rows = samples[:n_test]
    val_rows = samples[n_test : n_test + n_val]
    train_rows = samples[n_test + n_val :]

    write_jsonl(args.out_dir / "train.jsonl", train_rows)
    write_jsonl(args.out_dir / "val.jsonl", val_rows)
    write_jsonl(args.out_dir / "test.jsonl", test_rows)

    print(
        f"Done! Wrote:\n"
        f"  - {args.out_dir / 'train.jsonl'} ({len(train_rows)} rows)\n"
        f"  - {args.out_dir / 'val.jsonl'}   ({len(val_rows)} rows)\n"
        f"  - {args.out_dir / 'test.jsonl'}  ({len(test_rows)} rows)"
    )


if __name__ == "__main__":
    main()
