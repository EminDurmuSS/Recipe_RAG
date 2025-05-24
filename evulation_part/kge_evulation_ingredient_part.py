#!/usr/bin/env python3
"""
Recipe Retrieval Ingredient-Only PRO Evaluation 🥕 (v1.13-pro-ingredient-only, debug)
-----------------------------------------------------------
* Sadece “containsIngredient” kriteri üzerinden çalışır
* Ground-truth’daki combo sütununu ‘;’ ile ayırır, her satır bir senaryo
* Her senaryo için adım adım predict_target + normalize (cache yok)
* hits@1, hits@5, hits@10, MRR metrikleri
* Detaylı logging + tqdm ilerleme çubukları
"""
from __future__ import annotations
import ast
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
from pandas.errors import MergeError
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
import torch
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

# ── CONFIG ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "RecipeRag-Evulation" / "data"
TRIPLES_CSV = DATA_DIR / "triples_new_without_ct_ss.csv"
GT_PATTERN = "ground_truth_size_*.csv"
MODEL_ROOT = BASE_DIR / "train_new_kge_model"
OUTPUT_ROOT = BASE_DIR / "output" / "ingredient_pro_evaluation"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

INGREDIENT_REL = "containsIngredient"
REL2NODE = {INGREDIENT_REL: "ingredient"}

# ── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ing_eval_pro")


def log_mem():
    if psutil:
        logger.info(
            "Memory usage: %.1f MB",
            psutil.Process(os.getpid()).memory_info().rss / 2**20,
        )


# ── HELPERS ─────────────────────────────────────────────────────────────────
def tuple_to_canonical(s: str) -> str:
    try:
        head, tail = ast.literal_eval(s)
        tail = str(tail).strip()
        return f"{head}_id_{tail}" if head == "recipe" else f"{head}_{tail}"
    except Exception:
        return s.strip()


def load_triples_factory(path: Path) -> TriplesFactory:
    logger.info("Loading KG triples from %s …", path.name)
    df = pd.read_csv(path, dtype=str)
    df["Head"] = df["Head"].apply(tuple_to_canonical)
    df["Tail"] = df["Tail"].apply(tuple_to_canonical)
    df["Relation"] = df["Relation"].str.strip()
    tf = TriplesFactory.from_labeled_triples(
        np.stack([df.Head, df.Relation, df.Tail], axis=1),
        create_inverse_triples=True,
    )
    logger.info(
        "→ KG loaded: entities=%d, relations=%d, triples=%d",
        tf.num_entities,
        tf.num_relations,
        tf.num_triples,
    )
    return tf


def load_ground_truth(
    path: Path,
) -> Tuple[List[List[str]], List[Set[str]], List[int], List[Tuple[str, int]]]:
    """
    Reads GT files with columns: combo, match_count, recipe_ids
    Returns:
      - all_ings: list of ingredient lists per row (split on ';')
      - truths: set of recipe_id_x per row
      - counts: ground-truth match_count per row
      - meta: (filename stem, row index)
    """
    logger.info("Reading ground truth from %s …", path.name)
    df = pd.read_csv(path, dtype=str)
    all_ings: List[List[str]] = []
    truths: List[Set[str]] = []
    counts: List[int] = []
    meta: List[Tuple[str, int]] = []

    required = {"combo", "match_count", "recipe_ids"}
    if not required.issubset(df.columns):
        logger.error("Expected columns %s in %s", required, path.name)
        return all_ings, truths, counts, meta

    for idx, row in df.iterrows():
        combo = row["combo"] or ""
        # Burada ';' ile bölüyoruz, çünkü her ingredient grubu kendi içinde virgül içeriyor
        ings = [m.strip() for m in combo.split(";") if m.strip()]
        if not ings:
            continue
        all_ings.append(ings)

        # parse recipe_ids
        ids: Set[str] = set()
        for part in re.split(r"[,\[\]]+", row["recipe_ids"] or ""):
            p = part.strip().strip("'\"")
            if not p:
                continue
            if p.isdigit():
                ids.add(f"recipe_id_{p}")
            else:
                try:
                    ent, val = ast.literal_eval(p)
                    ids.add(f"{ent}_{val}")
                except Exception:
                    logger.warning("ID parse failed: %r", p)
        truths.append(ids)

        counts.append(int(row["match_count"]))
        meta.append((path.stem, idx))

    logger.info("→ Loaded %d ground-truth rows.", len(all_ings))
    return all_ings, truths, counts, meta


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df["norm_score"] = [] if df.empty else MinMaxScaler().fit_transform(df[["score"]])
    return df


def get_predictions(model, tf: TriplesFactory, rel: str, tail: str) -> pd.DataFrame:
    dfp = predict_target(model=model, relation=rel, tail=tail, triples_factory=tf).df
    logger.info("    • predict_target(%s, %s) → %d rows", rel, tail, len(dfp))
    return dfp[["head_label", "score"]].copy()


def rank_for_scenario_debug(
    model, tf: TriplesFactory, scenario: List[Tuple[str, str]]
) -> Tuple[List[str], int]:
    """
    PRO flow for a single scenario:
      1) for each (rel,val) get_predictions + normalise
      2) inner-merge on head_label (skip merge-errors)
      3) sum norm_scores → score_sum → sort desc
    """
    logger.info("→ New scenario: %r", scenario)
    dfs: List[pd.DataFrame] = []
    for idx, (rel, val) in enumerate(scenario, start=1):
        node = REL2NODE.get(rel)
        if node is None:
            logger.warning("  %2d) No mapping for %s, skipping", idx, rel)
            continue
        tail = f"{node}_{val}"
        logger.info("  %2d) Processing %s → %s", idx, rel, tail)
        if rel not in tf.relation_to_id or tail not in tf.entity_to_id:
            logger.warning("     • %s or %s not in KG, skipping", rel, tail)
            continue
        dfp = get_predictions(model, tf, rel, tail)
        norm = normalise(dfp.copy())
        logger.info(
            "     • Normalized: min=%.4f, max=%.4f",
            norm["norm_score"].min(),
            norm["norm_score"].max(),
        )
        dfs.append(
            norm[["head_label", "norm_score"]].rename(
                columns={"norm_score": f"{rel}_{idx}"}
            )
        )

    if not dfs:
        logger.error("   !!! No criteria produced any DataFrame; returning empty")
        return [], 0

    merged = dfs[0]
    logger.info("   • Candidates after first criterion: %d", len(merged))
    for j, other in enumerate(dfs[1:], start=2):
        before = len(merged)
        try:
            merged = merged.merge(other, on="head_label", how="inner")
        except MergeError as e:
            logger.warning(
                "   • MergeError on criterion %d (%s): %s – skipping this one",
                j,
                other.columns.tolist(),
                e,
            )
            continue
        after = len(merged)
        logger.info("   • After merging %d criteria: %d → %d", j, before, after)
        if merged.empty:
            logger.warning("   → No candidates left after merge")
            return [], 0

    merged["score_sum"] = merged.drop(columns=["head_label"]).sum(axis=1)
    ranked = merged.sort_values("score_sum", ascending=False)
    logger.info("   → Top 5 heads: %s", ranked["head_label"].tolist()[:5])
    return ranked["head_label"].tolist(), len(ranked)


def compute_metrics(preds: List[str], truth: Set[str], ks=(1, 5, 10)) -> dict:
    res = {f"hits@{k}": float(any(p in truth for p in preds[:k])) for k in ks}
    ranks = [preds.index(p) for p in truth if p in preds]
    res["mrr"] = 1.0 / (min(ranks) + 1) if ranks else 0.0
    return res


# ── MAIN ─────────────────────────────────────────────────────────────────
def main():
    start_all = time.time()
    try:
        from importlib.metadata import version

        logger.info("PyKEEN %s", version("pykeen"))
    except Exception:
        logger.info("PyKEEN version unknown")

    # 1) load KG
    tf = load_triples_factory(TRIPLES_CSV)
    log_mem()

    # 2) load models & GT files
    model_dirs = sorted(MODEL_ROOT.glob("trained_*_model_new_without_ct_ss"))
    gt_files = sorted(DATA_DIR.glob(GT_PATTERN))
    if not model_dirs or not gt_files:
        logger.error("Models or GT files not found.")
        return

    # 3) evaluate each model
    for md in model_dirs:
        model_name = md.name.split("_")[1]
        logger.info("=== Evaluating model: %s ===", model_name)
        mp = md / "trained_model.pkl"
        if not mp.exists():
            logger.warning("Model file missing: %s", mp)
            continue
        model = torch.load(mp, map_location="cpu", weights_only=False).eval()
        log_mem()

        outd = OUTPUT_ROOT / model_name
        outd.mkdir(parents=True, exist_ok=True)

        for gt in tqdm(gt_files, desc=f"{model_name} GT files", unit="file"):
            gt_name = gt.stem
            logger.info("-> Processing GT: %s", gt_name)

            # Ground-truth’tan doğrudan combo listesini alıyoruz
            all_ings, truths, counts, meta = load_ground_truth(gt)
            combos = [
                ([(INGREDIENT_REL, ing) for ing in combo], truth_set, cnt, (gid, idx))
                for combo, truth_set, cnt, (gid, idx) in zip(
                    all_ings, truths, counts, meta
                )
            ]

            records = []
            for i, (scenario, truth_set, cnt, (gid, idx)) in enumerate(
                tqdm(combos, desc=f"{model_name}:{gt_name}", unit="scn"), start=1
            ):
                preds, cand_cnt = rank_for_scenario_debug(model, tf, scenario)
                metrics = compute_metrics(preds, truth_set)
                rec = {
                    "scenario_global_id": gid,
                    "scenario_index_in_size": idx,
                    "combo_size": len(scenario),
                    "criteria": repr(scenario),
                    "matching_recipe_count": cnt,
                    "candidate_recipe_count": cand_cnt,
                    **metrics,
                }
                records.append(rec)
                if i % 50 == 0:
                    log_mem()

            df = pd.DataFrame.from_records(records)
            fd, tmp = tempfile.mkstemp(
                suffix=".csv", prefix=f"{model_name}_{gt_name}_", dir=outd
            )
            os.close(fd)
            df.to_csv(tmp, index=False)
            final = outd / f"{model_name}_{gt_name}_hits_mrr_results.csv"
            shutil.move(tmp, final)
            logger.info(
                "[%s] ✔ Wrote %s (%d rows) in %.1fs",
                model_name,
                final.name,
                len(df),
                time.time() - start_all,
            )

    logger.info("✅ ALL MODELS DONE in %.1fs.", time.time() - start_all)


if __name__ == "__main__":
    main()
