#!/usr/bin/env python3
"""
Recipe Retrieval Evaluation – Text-Embedding Ingredient‑Only PRO Edition 🥕📈 (v1.2-te-ingredient-only, debug)
---------------------------------------------------------------------------
* Uses sentence-transformers (all-MiniLM-L6-v2) to embed recipes & ingredient-only queries
* Reads `combo` column (ingredients separated by ';') for scenarios
* Robustly parses `recipe_ids` that may be a list literal **or** a single integer
* Detailed logging + tqdm progress bars + debug outputs
* Computes Hits@K, Precision@K, and MRR for each scenario
"""
from __future__ import annotations

import ast
import csv
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

# ── CONFIG & LOGGING ───────────────────────────────────────────────────────────
csv.field_size_limit(sys.maxsize)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "RecipeRag-Evulation" / "data"
TRIPLES_CSV = DATA_DIR / "triples_new_without_ct_ss.csv"
GT_PATTERN = "ground_truth_size_*.csv"
OUTPUT_DIR = BASE_DIR / "output" / "evaluation_results" / "text_embed_ingredient_only"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
TOP_K = (1, 5, 10)
ING_REL = "containsIngredient"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("te_ing_eval")

# ── HELPERS ───────────────────────────────────────────────────────────────────


def iterify(x):
    """Return *x* itself if iterable; otherwise wrap it in a one‑element list."""
    return x if isinstance(x, (list, tuple, set)) else [x]


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
    List[List[Tuple[str, str]]], List[Set[str]], List[int], List[Tuple[str, int]]
]:
    scenarios: List[List[Tuple[str, str]]] = []
    truths: List[Set[str]] = []
    counts: List[int] = []
    meta: List[Tuple[str, int]] = []

    logger.info("Loading GT file: %s", path.name)
    df = pd.read_csv(path, dtype=str)

    for idx, row in df.iterrows():
        combo = row.get("combo", "") or ""
        ings = [m.strip() for m in combo.split(";") if m.strip()]
        if not ings:
            continue
        scenario = [(ING_REL, ing) for ing in ings]

        # ── robust parse of recipe_ids ────────────────────────────────────────
        raw_cell = row.get("recipe_ids", "")
        try:
            parsed = ast.literal_eval(raw_cell)
        except Exception:
            parsed = raw_cell  # keep as-is (may already be int / str)

        ids: Set[str] = set()
        for tok in iterify(parsed):
            if pd.isna(tok) or tok == "":
                continue
            if isinstance(tok, (list, tuple)) and len(tok) == 2:
                head, val = tok
                ids.add(f"{head}_{val}")
            else:
                s = str(tok).strip()
                if s.isdigit():
                    ids.add(f"recipe_id_{s}")
                else:
                    # last‑ditch attempt: maybe it's a repr of a tuple
                    try:
                        h, v = ast.literal_eval(s)
                        ids.add(f"{h}_{v}")
                    except Exception:
                        logger.debug(
                            "Unrecognised recipe_id token @row %d: %r", idx, tok
                        )

        if not ids:
            logger.warning("No recipe IDs parsed @row %d", idx)

        cnt = (
            int(row.get("match_count", len(ids)))
            if row.get("match_count")
            else len(ids)
        )
        gid = row.get("scenario_global_id", path.stem)
        try:
            midx = int(row.get("scenario_index_in_size", idx))
        except Exception:
            midx = idx
        scenarios.append(scenario)
        truths.append(ids)
        counts.append(cnt)
        meta.append((gid, midx))

    logger.info("GT loaded: scenarios=%d", len(scenarios))
    return scenarios, truths, counts, meta


def build_recipe_texts(triples_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(triples_csv, dtype=str)
    df["Head"] = df["Head"].apply(tuple_to_canonical)
    df["Tail"] = df["Tail"].apply(tuple_to_canonical)
    df["Relation"] = df["Relation"].str.strip()

    recs = df[df["Head"].str.startswith("recipe_")]
    grouped = (
        recs.groupby("Head")
        .apply(
            lambda g: ". ".join(f"{r} {t}" for r, t in zip(g["Relation"], g["Tail"]))
        )
        .reset_index(name="full_text")
        .rename(columns={"Head": "recipe_id"})
        .set_index("recipe_id")
    )
    logger.info("Built texts for %d recipes", len(grouped))
    return grouped


def embed_recipes(
    texts: pd.Series, model_name: str, batch_size: int
) -> Tuple[np.ndarray, SentenceTransformer]:
    model = SentenceTransformer(model_name)
    embs = model.encode(texts.tolist(), batch_size=batch_size, show_progress_bar=True)
    logger.info("Embedded recipes: shape %s", embs.shape)
    return embs, model


def rank_for_scenario(
    scenario: List[Tuple[str, str]],
    recipe_embs: np.ndarray,
    recipe_ids: List[str],
    model: SentenceTransformer,
) -> List[str]:
    query = "; ".join(f"{r} = {v}" for r, v in scenario)
    logger.debug("Query: %s", query)
    q_emb = model.encode([query])[0]
    sims = cosine_similarity([q_emb], recipe_embs)[0]
    idxs = np.argsort(-sims)
    return [recipe_ids[i] for i in idxs]


def compute_metrics(
    preds: List[str], truth: Set[str], ks: Tuple[int, ...] = TOP_K
) -> Dict[str, float]:
    res: Dict[str, float] = {}
    for k in ks:
        topk = preds[:k]
        hits = sum(1 for p in topk if p in truth)
        res[f"hits@{k}"] = float(hits > 0)
        res[f"precision@{k}"] = hits / k
    ranks = [preds.index(p) + 1 for p in truth if p in preds]
    res["mrr"] = 1.0 / min(ranks) if ranks else 0.0
    return res


def main() -> None:
    start_all = time.time()

    # 1) build & embed recipe texts
    rec_texts = build_recipe_texts(TRIPLES_CSV)
    recipe_ids = rec_texts.index.tolist()
    recipe_embs, model = embed_recipes(
        rec_texts["full_text"], EMBEDDING_MODEL, BATCH_SIZE
    )

    # 2) iterate over ground‑truth files
    gt_files = sorted(DATA_DIR.glob(GT_PATTERN))
    if not gt_files:
        logger.error("No ground-truth files found (%s)", GT_PATTERN)
        sys.exit(1)
    logger.info("Found %d GT files", len(gt_files))

    for gt in tqdm(gt_files, desc="GT files", unit="file"):
        scenarios, truths, counts, meta = load_ground_truth(gt)
        records: List[Dict[str, float]] = []

        for scen, truth_set, cnt, (gid, idx) in tqdm(
            zip(scenarios, truths, counts, meta),
            total=len(scenarios),
            desc=gt.stem,
            unit="scn",
        ):
            preds = rank_for_scenario(scen, recipe_embs, recipe_ids, model)
            metrics = compute_metrics(preds, truth_set)
            records.append(
                {
                    **metrics,
                    "scenario_global_id": gid,
                    "scenario_index_in_size": idx,
                    "matching_recipe_count": cnt,
                    "criteria": repr(scen),
                }
            )

        df = pd.DataFrame.from_records(records)
        cols = (
            [
                "scenario_global_id",
                "scenario_index_in_size",
                "criteria",
                "matching_recipe_count",
            ]
            + [f"hits@{k}" for k in TOP_K]
            + [f"precision@{k}" for k in TOP_K]
            + ["mrr"]
        )
        df = df[cols]

        # atomic write with error handling
        try:
            fd, tmp = tempfile.mkstemp(suffix=".csv", dir=OUTPUT_DIR)
            os.close(fd)
            out_path = OUTPUT_DIR / f"textembed_ingredient_only_{gt.stem}_results.csv"
            df.to_csv(tmp, index=False)
            shutil.move(tmp, out_path)
            logger.info("Wrote %s (%d rows)", out_path.name, len(df))
        except Exception as e:
            logger.error("Failed to write results for %s: %s", gt.name, e)

    logger.info("ALL DONE in %.1fs", time.time() - start_all)


if __name__ == "__main__":
    main()
