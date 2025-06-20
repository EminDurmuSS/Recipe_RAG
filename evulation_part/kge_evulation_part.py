#!/usr/bin/env python3
"""
Recipe Retrieval Evaluation – PRO Edition 🏎️📈 (v1.3-pro, debug)
-----------------------------------------------------------
* Removed per-criterion caching
* No top-K candidate limit
* Detailed logging + tqdm progress bars + step-by-step debug outputs
* Now evaluates ALL trained KGE models (TransE, RotatE, QuatE) separately
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
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
import torch
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "RecipeRag-Evulation" / "data"
TRIPLES_CSV = DATA_DIR / "triples_new_without_ct_ss.csv"
GT_DIR = DATA_DIR
OUTPUT_DIR = BASE_DIR / "output" / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REL2NODE = {
    "usesCookingMethod": "cooking_method",
    "hasCuisineRegion": "cuisine_region",
    "hasDietType": "diet_type",
    "isForMealType": "meal_type",
    "containsIngredient": "ingredient",
    "HasProteinLevel": "health_attribute",
    "HasCarbLevel": "health_attribute",
    "HasFatLevel": "health_attribute",
    "HasSaturatedFatLevel": "health_attribute",
    "HasCalorieLevel": "health_attribute",
    "HasSodiumLevel": "health_attribute",
    "HasSugarLevel": "health_attribute",
    "HasFiberLevel": "health_attribute",
    "HasCholesterolLevel": "health_attribute",
}

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("recipe_eval")


# ── HELPERS ───────────────────────────────────────────────────────────────────
def tuple_to_canonical(s: str) -> str:
    try:
        head, tail = ast.literal_eval(s)
        tail = str(tail).strip()
        return f"{head}_id_{tail}" if head == "recipe" else f"{head}_{tail}"
    except Exception:
        return s.strip()


def load_triples_factory(path: Path) -> TriplesFactory:
    df = pd.read_csv(path, dtype=str)
    df["Head"] = df["Head"].apply(tuple_to_canonical)
    df["Tail"] = df["Tail"].apply(tuple_to_canonical)
    df["Relation"] = df["Relation"].str.strip()
    tf = TriplesFactory.from_labeled_triples(
        np.stack([df.Head, df.Relation, df.Tail], axis=1),
        create_inverse_triples=True,
    )
    logger.info(
        "Loaded KG – entities=%d, relations=%d, triples=%d",
        tf.num_entities,
        tf.num_relations,
        tf.num_triples,
    )
    return tf


def load_ground_truth(path: Path):
    df = pd.read_csv(path, dtype=str)
    scenarios, truths, counts, meta = [], [], [], []
    for _, row in df.iterrows():
        scenarios.append(
            [
                (r.strip(), v.strip())
                for r, v in ast.literal_eval(row["scenario_criteria"])
            ]
        )
        ids: Set[str] = set()
        for part in re.split(r"[\,\[\]]+", row["matching_recipe_ids"]):
            part = part.strip().strip("'\"")
            if not part:
                continue
            if part.isdigit():
                ids.add(f"recipe_id_{part}")
                continue
            try:
                ent, val = ast.literal_eval(part)
                ids.add(f"{ent}_{val}")
            except Exception:
                logger.warning("Unrecognized ID format: %r", part)
        truths.append(ids)
        counts.append(int(row["matching_recipe_count"]))
        meta.append((row["scenario_global_id"], int(row["scenario_index_in_size"])))
    return scenarios, truths, counts, meta


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df["norm_score"] = 0.0 if df.empty else MinMaxScaler().fit_transform(df[["score"]])
    return df


# ── CORE – debug version─────────────────────────────────────────────────────
def get_predictions(model, tf: TriplesFactory, rel: str, tail: str) -> pd.DataFrame:
    return predict_target(
        model=model,
        relation=rel,
        tail=tail,
        triples_factory=tf,
    ).df[["head_label", "score"]]


def rank_for_scenario_debug(model, tf: TriplesFactory, scenario: List[Tuple[str, str]]):
    logger.info("→ New scenario: %r", scenario)
    dfs = []
    for idx, (rel, val) in enumerate(scenario, 1):
        node_type = REL2NODE.get(rel)
        if node_type is None:
            logger.warning("  %2d) No mapping for relation `%s`, skipping.", idx, rel)
            continue

        tail = f"{node_type}_{val}"
        logger.info("  %2d) Processing: rel=`%s`, tail=`%s`", idx, rel, tail)

        if rel not in tf.relation_to_id:
            logger.warning("     • Relation `%s` not found in KG!", rel)
            continue
        if tail not in tf.entity_to_id:
            logger.warning("     • Entity `%s` not found in KG!", tail)
            continue

        dfp = get_predictions(model, tf, rel, tail)
        logger.info(
            "     • predict_target returned: %d rows (first 3 head_labels): %s",
            len(dfp),
            dfp["head_label"].tolist()[:3],
        )

        norm = normalise(dfp.copy())
        logger.info(
            "     • After normalization, norm_score distribution: min=%.3f, max=%.3f",
            norm["norm_score"].min(),
            norm["norm_score"].max(),
        )
        df_col = norm[["head_label", "norm_score"]].rename(columns={"norm_score": rel})
        dfs.append(df_col)

    if not dfs:
        logger.error(
            "   !!! No dataframe could be created for any criteria; returning empty."
        )
        return [], 0

    merged = dfs[0]
    logger.info("   • Initially %d candidates (first criterion).", len(merged))
    for j, other in enumerate(dfs[1:], 2):
        before = len(merged)
        merged = merged.merge(other, on="head_label", how="inner")
        after = len(merged)
        logger.info(
            "   • After merging %d criteria: before %d → after %d candidates",
            j,
            before,
            after,
        )

    if merged.empty:
        logger.warning(
            "   → After merge, no candidates left; will return empty result."
        )
        return [], 0

    merged["score_sum"] = merged.drop(columns=["head_label"]).sum(axis=1)
    ranked = merged.sort_values("score_sum", ascending=False)
    logger.info(
        "   → After merge, total %d candidates remaining; top 5 heads: %s",
        len(ranked),
        ranked["head_label"].tolist()[:5],
    )
    return ranked["head_label"].tolist(), len(ranked)


def compute_metrics(preds: List[str], truth: Set[str], ks=(1, 5, 10)):
    res = {f"hits@{k}": float(any(p in truth for p in preds[:k])) for k in ks}
    ranks = [preds.index(p) for p in truth if p in preds]
    res["mrr"] = 1.0 / (min(ranks) + 1) if ranks else 0.0
    return res


def log_mem():
    if psutil:
        logger.info(
            "Memory usage: %.1f MB",
            psutil.Process(os.getpid()).memory_info().rss / 2**20,
        )


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    start_all = time.time()

    try:
        from importlib.metadata import version

        pk_version = version("pykeen")
    except Exception:
        pk_version = "unknown"
    logger.info("PyKEEN version: %s", pk_version)

    # 1) Load KG
    tf = load_triples_factory(TRIPLES_CSV)

    # 2) Find all trained_*_model_new_without_ct_ss folders under train_new_kge_model
    MODEL_ROOT = BASE_DIR / "train_new_kge_model"
    model_dirs = sorted(MODEL_ROOT.glob("trained_*_model_new_without_ct_ss"))

    for model_dir in model_dirs:
        # folder name: trained_QuatE_model_new_without_ct_ss → model_name = QuatE
        model_name = model_dir.name.split("_")[1]
        logger.info("=== Evaluating model: %s ===", model_name)

        # load .pkl file
        pretrained_model_path = model_dir / "trained_model.pkl"
        if not pretrained_model_path.exists():
            logger.error(
                "[%s] Pretrained model not found: %s",
                model_name,
                pretrained_model_path,
            )
            continue

        model = torch.load(
            pretrained_model_path, map_location=torch.device("cpu"), weights_only=False
        ).eval()
        logger.info("[%s] Model loaded and set to eval() mode.", model_name)
        log_mem()

        # process ground-truth CSV files
        gt_files = sorted(GT_DIR.glob("ground_truth_*.csv"))
        logger.info("[%s] Found %d ground-truth file(s).", model_name, len(gt_files))

        # create a separate output directory for each model
        out_dir = OUTPUT_DIR / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for gt_file in tqdm(gt_files, desc=f"{model_name} GT files", unit="file"):
            t0, gt_name = time.time(), gt_file.stem
            logger.info("[%s] ➤ Evaluating %s …", model_name, gt_name)

            scenarios, truths, counts, meta = load_ground_truth(gt_file)
            records = []
            for i, (scenario, truth_set, _, (gid, idx)) in enumerate(
                tqdm(
                    zip(scenarios, truths, counts, meta),
                    total=len(scenarios),
                    desc=f"{model_name}:{gt_name}",
                    unit="scn",
                ),
                1,
            ):
                preds, cand_cnt = rank_for_scenario_debug(model, tf, scenario)
                rec = compute_metrics(preds, truth_set) | {
                    "scenario_global_id": gid,
                    "scenario_index_in_size": idx,
                    "matching_recipe_count": len(truth_set),
                    "candidate_recipe_count": cand_cnt,
                    "criteria": repr(scenario),
                }
                records.append(rec)
                if i % 50 == 0:
                    logger.info(
                        "[%s:%s] processed %d/%d",
                        model_name,
                        gt_name,
                        i,
                        len(scenarios),
                    )
                    log_mem()

            df = pd.DataFrame.from_records(records)[
                [
                    "scenario_global_id",
                    "scenario_index_in_size",
                    "criteria",
                    "matching_recipe_count",
                    "candidate_recipe_count",
                    "hits@1",
                    "hits@5",
                    "hits@10",
                    "mrr",
                ]
            ]

            # Atomic move via temporary file
            fd, tmp = tempfile.mkstemp(
                suffix=".csv",
                prefix=f"{model_name}_{gt_name}_",
                dir=out_dir,
            )
            os.close(fd)
            df.to_csv(tmp, index=False)
            final_path = out_dir / f"{model_name}_{gt_name}_hits_mrr_results.csv"
            shutil.move(tmp, final_path)
            logger.info(
                "[%s] ✔ Wrote %s (%d rows) – elapsed %.1fs",
                model_name,
                final_path.name,
                len(df),
                time.time() - t0,
            )

    logger.info("✅ ALL MODELS DONE in %.1fs.", time.time() - start_all)


if __name__ == "__main__":
    main()
