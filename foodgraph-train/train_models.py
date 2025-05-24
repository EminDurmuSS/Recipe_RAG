#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train_kge_models_en.py

"""Train multiple knowledge‑graph embedding (KGE) models on a CSV of triples.

This script loads a (Head, Relation, Tail) triples CSV, converts tuple‑shaped
cells (e.g. "('recipe', 38)") into a canonical form (``recipe_38``), and then
trains the requested KGE models (TransE, RotatE, QuatE) using *PyKEEN*.
Each model is persisted to its own output directory together with training
metrics.
"""

import os
import ast
import pandas as pd
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

###############################################################################
# Helper Functions
###############################################################################


def tuple_to_canonical(s: str) -> str:
    """Convert textual tuples read from the CSV into a canonical string.

    Example
    -------
    >>> tuple_to_canonical("('recipe', 38)")
    'recipe_38'

    Parameters
    ----------
    s : str
        The raw string representation of the tuple.

    Returns
    -------
    str
        Canonicalised string in the form ``<label>_<id>``.
    """
    try:
        t = ast.literal_eval(s)
        return f"{t[0]}_{t[1]}"
    except Exception as e:
        print(f"Tuple parse error: {s}, error: {e}")
        return s


###############################################################################
# Main Training Function
###############################################################################


def train_kge_model(
    triples_csv_path: str,
    output_dir: str,
    model_name: str,
    num_epochs: int = 400,
    num_negs_per_pos: int = 40,
):
    """Train a single KGE model and save it to *output_dir*.

    Parameters
    ----------
    triples_csv_path : str
        Path to the CSV containing triples.
    output_dir : str
        Directory where the trained model and metrics will be stored.
    model_name : str
        Name of the KGE model (``TransE``, ``RotatE``, or ``QuatE``).
    num_epochs : int, optional
        Number of training epochs, by default 400.
    num_negs_per_pos : int, optional
        Negative samples per positive example, by default 40.

    Returns
    -------
    pykeen.pipeline.PipelineResult
        Result object with the trained model, metrics, and metadata.
    """
    print(f"\n=== [{model_name}] Loading CSV: {triples_csv_path}")
    df = pd.read_csv(triples_csv_path)
    print(f"[{model_name}] Rows read: {len(df)}")

    # Build triples array
    triples = []
    for _, row in df.iterrows():
        h = tuple_to_canonical(row["Head"])
        t = tuple_to_canonical(row["Tail"])
        r = row["Relation"].strip()
        triples.append((h, r, t))
    triples = np.array(triples, dtype=str)
    print(f"[{model_name}] Total triples: {len(triples)}")

    # Create a TriplesFactory that also contains inverse relations
    tf = TriplesFactory.from_labeled_triples(triples, create_inverse_triples=True)
    print(f"[{model_name}] TriplesFactory created (including inverse relations).")

    # Run the training pipeline
    print(f"[{model_name}] Training (epochs={num_epochs}, negs/pos={num_negs_per_pos})")
    result = pipeline(
        training=tf,
        validation=tf,
        testing=tf,
        model=model_name,
        negative_sampler="basic",
        negative_sampler_kwargs={
            "num_negs_per_pos": num_negs_per_pos,
            "filtered": True,
        },
        epochs=num_epochs,
        stopper="early",
    )
    print(f"[{model_name}] Training complete. Metrics:\n{result.metric_results}")

    # Persist the trained model and its metrics
    os.makedirs(output_dir, exist_ok=True)
    result.save_to_directory(output_dir)
    print(f"[{model_name}] Model and metrics saved to: {output_dir}")

    return result


###############################################################################
# Script Entry Point
###############################################################################

if __name__ == "__main__":
    # 1) Path to the CSV file containing the triples
    triples_csv = "/app/train_new_kge_model/triples_new_without_ct_ss.csv"

    # 2) Models to train and their respective output folders
    models = {
        "TransE": "trained_TransE_model_new_without_ct_ss",
        "RotatE": "trained_RotatE_model_new_without_ct_ss",
        "QuatE": "trained_QuatE_model_new_without_ct_ss",
    }

    # 3) Shared hyper‑parameters
    num_epochs = 250
    num_negs_per_pos = 40
    base_output_dir = "/app/train_new_kge_model"

    print(">>> Starting multi‑model training run <<<")
    for model_name, folder in models.items():
        out_dir = os.path.join(base_output_dir, folder)
        train_kge_model(
            triples_csv_path=triples_csv,
            output_dir=out_dir,
            model_name=model_name,
            num_epochs=num_epochs,
            num_negs_per_pos=num_negs_per_pos,
        )
    print(">>> All trainings finished successfully! <<<")
