#!/usr/bin/env python3
"""
Recipe Retrieval Evaluation – Text-Embedding PRO Edition 🏎️📈 (v1.0-te-pro, debug)
---------------------------------------------------------------------------
* Uses sentence-transformers (all-MiniLM-L6-v2) to embed recipes & queries
* No per-criterion caching, no top-K candidate limit
* Detailed logging + tqdm progress bars + step-by-step debug outputs
* Computes Hits@K, Precision@K, MRR for each scenario
"""

from __future__ import annotations
import ast
import csv
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Set, Tuple, Dict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

# ── CONFIG & LOGGING ───────────────────────────────────────────────────────────

# ensure no CSV field size limits
csv.field_size_limit(sys.maxsize)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "RecipeRag-Evulation" / "data"
TRIPLES_CSV = DATA_DIR / "triples_new_without_ct_ss.csv"
GT_PATTERN = "ground_truth_*.csv"
OUTPUT_DIR = BASE_DIR / "output" / "evaluation_results" / "text_embed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
TOP_K = (1, 5, 10)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("recipe_eval")


# ── HELPERS ───────────────────────────────────────────────────────────────────


def tuple_to_canonical(s: str) -> str:
    """Convert ('recipe', '123') → 'recipe_id_123', else 'rel_val'."""
    try:
        head, tail = ast.literal_eval(s)
        tail = str(tail).strip()
        return f"{head}_id_{tail}" if head == "recipe" else f"{head}_{tail}"
    except Exception:
        return s.strip()


def load_ground_truth(
    path: Path,
) -> Tuple[
    List[List[Tuple[str, str]]],  # scenarios
    List[Set[str]],  # truth id sets
    List[int],  # counts
    List[Tuple[str, int]],  # (global_id, index)
]:
    """
    Parses a ground-truth CSV with columns:
      scenario_global_id, scenario_index_in_size,
      scenario_criteria (repr list of tuples),
      matching_recipe_ids (repr list),
      matching_recipe_count
    Returns aligned lists of scenarios, truths, counts, meta.
    """
    scenarios: List[List[Tuple[str, str]]] = []
    truths: List[Set[str]] = []
    counts: List[int] = []
    meta: List[Tuple[str, int]] = []

    total_rows = dropped = 0
    logger.info("▶ Loading GT file: %s", path.name)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",", quotechar='"', escapechar="\\")
        for row_ix, row in enumerate(reader, start=1):
            total_rows += 1

            # 1) parse scenario_criteria
            try:
                crits = ast.literal_eval(row["scenario_criteria"])
                scenario = [(r.strip(), v.strip()) for r, v in crits]
            except Exception as e:
                logger.error("  ✗ criteria parse error @row %d: %s", row_ix, e)
                dropped += 1
                continue

            # 2) parse matching_recipe_ids
            raw = row["matching_recipe_ids"]
            ids: Set[str] = set()
            # CSV quoting handles nested commas; strip brackets then split
            for tok in raw.strip("[] ").split(","):
                token = tok.strip().strip("'\"")
                if not token:
                    continue
                if token.isdigit():
                    ids.add(f"recipe_id_{token}")
                else:
                    try:
                        ent, val = ast.literal_eval(token)
                        ids.add(f"{ent}_{val}")
                    except Exception:
                        logger.error("  ✗ ID format error @row %d: %r", row_ix, tok)
            if not ids:
                logger.warning("  ! no IDs parsed @row %d", row_ix)

            # 3) parse matching_recipe_count
            try:
                cnt = int(row["matching_recipe_count"])
            except Exception:
                logger.error(
                    "  ✗ count parse error @row %d: %r",
                    row_ix,
                    row["matching_recipe_count"],
                )
                cnt = len(ids)

            # 4) parse meta fields
            gid = row.get("scenario_global_id", "")
            try:
                idx = int(row.get("scenario_index_in_size", "0"))
            except Exception:
                logger.error(
                    "  ✗ index parse error @row %d: %r",
                    row_ix,
                    row.get("scenario_index_in_size"),
                )
                idx = -1

            # collect
            scenarios.append(scenario)
            truths.append(ids)
            counts.append(cnt)
            meta.append((gid, idx))

    logger.info(
        "✔ GT loaded: total=%d, parsed=%d, dropped=%d",
        total_rows,
        len(scenarios),
        dropped,
    )
    return scenarios, truths, counts, meta


# ── RECIPE EMBEDDING ──────────────────────────────────────────────────────────


def build_recipe_texts(triples_csv: Path) -> pd.DataFrame:
    """Aggregate recipe triples into full_text per recipe_id."""
    df = pd.read_csv(triples_csv, dtype=str)
    df["Head"] = df["Head"].apply(tuple_to_canonical)
    df["Tail"] = df["Tail"].apply(tuple_to_canonical)
    df["Relation"] = df["Relation"].str.strip()

    recipe_triples = df[df["Head"].str.startswith("recipe_")]
    grouped = (
        recipe_triples.groupby("Head")
        .apply(
            lambda g: ". ".join(f"{r} {t}" for r, t in zip(g["Relation"], g["Tail"]))
        )
        .reset_index(name="full_text")
        .rename(columns={"Head": "recipe_id"})
        .set_index("recipe_id")
    )
    logger.info("✔ Built texts for %d recipes", len(grouped))
    return grouped


def embed_recipes(texts: pd.Series, model_name: str, batch_size: int) -> np.ndarray:
    """Encode recipe texts to embeddings."""
    model = SentenceTransformer(model_name)
    embs = model.encode(texts.tolist(), batch_size=batch_size, show_progress_bar=True)
    logger.info("✔ Embedded recipes: embeddings shape %s", embs.shape)
    return embs, model


# ── RANKING & METRICS ─────────────────────────────────────────────────────────


def rank_for_scenario_debug(
    scenario: List[Tuple[str, str]],
    recipe_embs: np.ndarray,
    recipe_ids: List[str],
    model: SentenceTransformer,
) -> Tuple[List[str], int]:
    """Embed the scenario text and rank all recipes by cosine similarity."""
    logger.info("→ New scenario: %r", scenario)
    query = "; ".join(f"{r} = {v}" for r, v in scenario)
    logger.debug("   • Query string: %s", query)
    q_emb = model.encode([query])[0]
    sims = cosine_similarity([q_emb], recipe_embs)[0]
    idxs = np.argsort(-sims)
    ranked = [recipe_ids[i] for i in idxs]
    logger.info("   • Ranked %d candidates", len(ranked))
    return ranked, len(ranked)


def compute_metrics(
    preds: List[str], truth: Set[str], ks: Tuple[int, ...] = TOP_K
) -> Dict[str, float]:
    """Compute Hits@K, Precision@K, and MRR."""
    res: Dict[str, float] = {}
    for k in ks:
        topk = preds[:k]
        hits = sum(1 for p in topk if p in truth)
        res[f"hits@{k}"] = 1.0 if hits > 0 else 0.0
        res[f"precision@{k}"] = hits / k
    ranks = [preds.index(p) + 1 for p in truth if p in preds]
    res["mrr"] = 1.0 / min(ranks) if ranks else 0.0
    return res


# ── MAIN WORKFLOW ─────────────────────────────────────────────────────────────


def main() -> None:
    start_all = time.time()

    # 1) build & embed recipes
    rec_texts = build_recipe_texts(TRIPLES_CSV)
    recipe_ids = rec_texts.index.tolist()
    recipe_embs, model = embed_recipes(
        rec_texts["full_text"], EMBEDDING_MODEL, BATCH_SIZE
    )

    # 2) load & evaluate ground-truth files
    gt_files = sorted(DATA_DIR.glob(GT_PATTERN))
    logger.info("Found %d GT files: %s", len(gt_files), [f.name for f in gt_files])

    for gt_file in tqdm(gt_files, desc="GT files", unit="file"):
        t0 = time.time()
        name = gt_file.stem
        logger.info("=== Evaluating GT: %s ===", name)

        scenarios, truths, counts, meta = load_ground_truth(gt_file)
        records: List[Dict[str, float]] = []

        for scen, truth_set, cnt, (gid, idx) in tqdm(
            zip(scenarios, truths, counts, meta),
            total=len(scenarios),
            desc=name,
            unit="scn",
        ):
            preds, cand_cnt = rank_for_scenario_debug(
                scen, recipe_embs, recipe_ids, model
            )
            metrics = compute_metrics(preds, truth_set)
            metrics.update(
                {
                    "scenario_global_id": gid,
                    "scenario_index_in_size": idx,
                    "matching_recipe_count": cnt,
                    "candidate_recipe_count": cand_cnt,
                    "criteria": repr(scen),
                }
            )
            records.append(metrics)

        df = pd.DataFrame.from_records(records)
        cols = (
            [
                "scenario_global_id",
                "scenario_index_in_size",
                "criteria",
                "matching_recipe_count",
                "candidate_recipe_count",
            ]
            + [f"hits@{k}" for k in TOP_K]
            + [f"precision@{k}" for k in TOP_K]
            + ["mrr"]
        )
        df = df[cols]

        # atomic write
        fd, tmp = tempfile.mkstemp(suffix=".csv", dir=OUTPUT_DIR)
        os.close(fd)
        out_path = OUTPUT_DIR / f"textembed_{name}_results.csv"
        df.to_csv(tmp, index=False)
        shutil.move(tmp, out_path)
        logger.info(
            "✔ Wrote %s (%d rows) – elapsed %.1fs",
            out_path.name,
            len(df),
            time.time() - t0,
        )

    logger.info("✅ ALL DONE in %.1fs", time.time() - start_all)


if __name__ == "__main__":
    main()
