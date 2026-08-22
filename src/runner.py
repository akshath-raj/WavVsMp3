"""Step 5 — the executor.

Grid (Phi-4, 3-condition codec study):
  50 items x 3 transforms x 2 tasks x 3 repeats  (main)
  + prompt-sensitivity: emotion_v2/v3 on {ref, roundtrip_wav} x 3 repeats

Resumable by construction: run_id is a hash of the call's identity; any run_id
already in results.parquet is skipped. Every raw response is written to
data/responses/{run_id}.json and never deleted. Aborts if cumulative spend
exceeds COST_CEILING_USD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from .config import (
    EMOTION_CODE_TO_WORD,
    MANIFEST_PATH,
    RESPONSES_DIR,
    RESULTS_PATH,
    SENTENCES,
    cfg,
)
from .model_client import get_client
from .parse import parse_emotion, wer_against_sentence

RESULT_COLUMNS = [
    "run_id", "timestamp", "model_id", "model_version", "item_id", "transform_id",
    "task", "prompt_id", "repeat_idx", "raw_response_path", "parsed_label",
    "parse_status", "gold_label", "latency_ms", "input_tokens", "output_tokens",
    "cost_usd", "fuzzy_score", "needs_review", "wer",
]
N_REPEATS = 3
TASKS = ["emotion", "transcribe"]


def run_id_for(model: str, item: str, tid: str, task: str, pid: str, rep: int) -> str:
    key = f"{model}|{item}|{tid}|{task}|{pid}|{rep}"
    return hashlib.sha256(key.encode()).hexdigest()


def build_plan(model: str, item_ids: list[str]) -> list[dict]:
    plan: list[dict] = []
    # main grid
    for item in item_ids:
        for tid in cfg.model_transforms:
            for task in TASKS:
                pid = cfg.canonical_prompt(task)
                for rep in range(N_REPEATS):
                    plan.append(_cell(model, item, tid, task, pid, rep))
    # prompt-sensitivity arm (emotion only, non-canonical prompts)
    emo = cfg.prompts["emotion"]
    canon = emo["canonical_prompt"]
    sens_prompts = [p for p in emo["prompts"] if p != canon]
    for item in item_ids:
        for tid in emo["sensitivity_transforms"]:
            for pid in sens_prompts:
                for rep in range(N_REPEATS):
                    plan.append(_cell(model, item, tid, "emotion", pid, rep))
    return plan


def _cell(model, item, tid, task, pid, rep) -> dict:
    return {
        "item_id": item, "transform_id": tid, "task": task,
        "prompt_id": pid, "repeat_idx": rep,
        "run_id": run_id_for(model, item, tid, task, pid, rep),
    }


def _load_done() -> tuple[pd.DataFrame, set[str], float]:
    if RESULTS_PATH.exists():
        df = pd.read_parquet(RESULTS_PATH)
        return df, set(df["run_id"]), float(df["cost_usd"].fillna(0).sum())
    return pd.DataFrame(columns=RESULT_COLUMNS), set(), 0.0


def _cost(input_tokens, output_tokens) -> float:
    it = input_tokens or 0
    ot = output_tokens or 0
    return (it / 1e6) * cfg.price_input_per_1m + (ot / 1e6) * cfg.price_output_per_1m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=None,
                    help="limit to first N item_ids (sorted) — for smoke tests")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan size and exit without calling the API")
    ap.add_argument("--min-interval", type=float, default=0.0,
                    help="min seconds between calls (rate-limit pacing)")
    ap.add_argument("--batch", type=int, default=50,
                    help="write results to parquet every N new rows")
    ap.add_argument("--abort-after", type=int, default=25,
                    help="abort (resumably) after this many consecutive api_errors "
                         "— usually means quota exhaustion")
    args = ap.parse_args()

    cfg.ensure_dirs()
    manifest = pd.read_parquet(MANIFEST_PATH)
    lookup = _build_lookup(manifest)
    all_items = sorted(manifest["item_id"].unique())
    item_ids = all_items[: args.items] if args.items else all_items

    client = get_client()
    plan = build_plan(client.model_id, item_ids)
    done_df, done_ids, spent = _load_done()
    todo = [c for c in plan if c["run_id"] not in done_ids]

    print(f"model={client.model_id}  items={len(item_ids)}  "
          f"planned={len(plan)}  done={len(plan) - len(todo)}  todo={len(todo)}")
    print(f"cost so far ${spent:.4f} / ceiling ${cfg.cost_ceiling_usd:.2f}")
    if args.dry_run or not todo:
        return

    batch: list[dict] = []
    consecutive_errors = 0
    pbar = tqdm(todo, unit="call")
    for cell in pbar:
        row = _execute(cell, lookup, client)
        spent += row["cost_usd"] or 0.0
        batch.append(row)
        pbar.set_postfix(cost=f"${spent:.3f}", errs=consecutive_errors)

        consecutive_errors = consecutive_errors + 1 if row["parse_status"] == "api_error" else 0
        if consecutive_errors >= args.abort_after:
            _flush(done_df, batch)
            raise SystemExit(
                f"ABORT: {consecutive_errors} consecutive api_errors "
                f"(likely quota/endpoint down). Progress saved; rerun to resume."
            )

        if spent > cfg.cost_ceiling_usd:
            done_df = _flush(done_df, batch)
            batch = []
            raise SystemExit(
                f"ABORT: cumulative cost ${spent:.4f} exceeded ceiling "
                f"${cfg.cost_ceiling_usd:.2f}. Progress saved; rerun to resume."
            )
        if len(batch) >= args.batch:
            done_df = _flush(done_df, batch)
            batch = []
        if args.min_interval:
            time.sleep(args.min_interval)

    if batch:
        _flush(done_df, batch)
    print(f"\nDone. Total spend ${spent:.4f}. Results -> {RESULTS_PATH}")


def _build_lookup(manifest: pd.DataFrame) -> dict:
    """(item_id, transform_id) -> stim_path; item_id -> gold fields."""
    stim = {(r.item_id, r.transform_id): r.stim_path for r in manifest.itertuples()}
    item_meta = (manifest.drop_duplicates("item_id")
                 .set_index("item_id")[["emotion_gold", "sentence_code"]]
                 .to_dict("index"))
    return {"stim": stim, "meta": item_meta}


def _execute(cell: dict, lookup: dict, client) -> dict:
    item, tid, task, pid = (cell["item_id"], cell["transform_id"],
                            cell["task"], cell["prompt_id"])
    stim_path = lookup["stim"][(item, tid)]
    meta = lookup["meta"][item]
    prompt = cfg.prompt_text(task, pid)

    result = client.call(stim_path, prompt)
    ts = datetime.now(timezone.utc).isoformat()

    # Persist the raw response + request metadata (never deleted).
    resp_path = RESPONSES_DIR / f"{cell['run_id']}.json"
    resp_path.write_text(json.dumps({
        "run_id": cell["run_id"], "timestamp": ts,
        "request": {
            "model_id": client.model_id, "item_id": item, "transform_id": tid,
            "task": task, "prompt_id": pid, "repeat_idx": cell["repeat_idx"],
            "prompt": prompt, "audio_path": stim_path,
        },
        "response": result,
    }, indent=2))

    raw = result["raw_text"]
    err = result["error"]
    if task == "emotion":
        p = parse_emotion(raw, err)
        gold = EMOTION_CODE_TO_WORD[meta["emotion_gold"]]
        wer = None
    else:  # transcribe
        p = _parse_transcribe(raw, err)
        gold = SENTENCES[meta["sentence_code"]]
        wer = None if err else wer_against_sentence(raw, meta["sentence_code"])

    usage = result["usage"]
    return {
        "run_id": cell["run_id"], "timestamp": ts,
        "model_id": client.model_id, "model_version": result["model_version"],
        "item_id": item, "transform_id": tid, "task": task, "prompt_id": pid,
        "repeat_idx": cell["repeat_idx"], "raw_response_path": str(resp_path),
        "parsed_label": p["parsed_label"], "parse_status": p["parse_status"],
        "gold_label": gold, "latency_ms": result["latency_ms"],
        "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
        "cost_usd": _cost(usage["input_tokens"], usage["output_tokens"]),
        "fuzzy_score": p.get("fuzzy_score"), "needs_review": p.get("needs_review", False),
        "wer": wer,
    }


def _parse_transcribe(raw, err) -> dict:
    if err:
        return {"parsed_label": None, "parse_status": "api_error"}
    if raw is None or not raw.strip():
        return {"parsed_label": None, "parse_status": "empty"}
    return {"parsed_label": None, "parse_status": "ok"}


def _flush(done_df: pd.DataFrame, batch: list[dict]) -> pd.DataFrame:
    if not batch:
        return done_df
    new = pd.DataFrame(batch)[RESULT_COLUMNS]
    combined = pd.concat([done_df, new], ignore_index=True)
    combined.to_parquet(RESULTS_PATH, index=False)
    return combined


if __name__ == "__main__":
    main()
