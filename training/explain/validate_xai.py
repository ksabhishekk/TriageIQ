"""Offline XAI validation on the 2024 test set; never imported by the API.

Run after XAI graph parity has passed:
    python training/explain/validate_xai.py --rows 100 --lime-rows 20

Install research-only packages with:
    python -m pip install -r training/explain/requirements.txt
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from api.scorer import Scorer
from api.xai import Explainer

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "training" / "outputs" / "serving_v3"
DATA_PATH = ROOT / "Data" / "data" / "interim" / "triageiq_training_v3.parquet"
OUT_PATH = ROOT / "training" / "results" / "xai_v3.json"


def rank_correlation(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    left_rank = np.argsort(np.argsort(left))
    right_rank = np.argsort(np.argsort(right))
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def model_features(explainer, row):
    return {
        column: int(row[column]) if column in explainer.cat_cols else float(row[column])
        for column in explainer.players
    }


def score_text(explainer, text, cat, num):
    vector = explainer._encode_texts([text])
    return float(explainer._head(vector, cat, num)[0])


def defined_mean(values):
    valid = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(valid)) if valid else None


def faithfulness_case(explainer, row, explanation, rng):
    units = [item["text"] for item in explanation["sentences"]]
    positive_order = sorted(
        range(len(units)),
        key=lambda i: explanation["sentences"][i]["logit_contribution"],
        reverse=True,
    )
    cat, num = explainer._prepare_features(model_features(explainer, row))
    original = explanation["modality"]["current_logit"]
    random = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
    output = []

    for k in range(1, min(3, len(units)) + 1):
        selected = set(positive_order[:k])
        deleted = " ".join(sentence for i, sentence in enumerate(units) if i not in selected)
        top_drop = original - score_text(explainer, deleted, cat, num)
        random_drops, random_insertions = [], []
        for _ in range(5):
            sample = set(random.choice(len(units), size=k, replace=False).tolist())
            random_drops.append(original - score_text(
                explainer, " ".join(sentence for i, sentence in enumerate(units) if i not in sample), cat, num
            ))
            random_insertions.append(score_text(
                explainer, " ".join(sentence for i, sentence in enumerate(units) if i in sample), cat, num
            ) - score_text(explainer, "", cat, num))
        top_insert = score_text(
            explainer, " ".join(units[i] for i in positive_order[:k]), cat, num
        ) - score_text(explainer, "", cat, num)
        output.append({
            "k": k,
            "top_sentence_logit_drop": top_drop,
            "random_sentence_logit_drop_mean": float(np.mean(random_drops)),
            "top_sentence_logit_insertion": top_insert,
            "random_sentence_logit_insertion_mean": float(np.mean(random_insertions)),
        })
    return output


def lime_agreement(explainer, row, explanation, num_samples, seed):
    from lime.lime_tabular import LimeTabularExplainer

    units = [item["text"] for item in explanation["sentences"]]
    if not units:
        return None
    cat, num = explainer._prepare_features(model_features(explainer, row))
    names = [f"sentence_{i}" for i in range(len(units))]
    lime = LimeTabularExplainer(
        training_data=np.asarray([[0] * len(units), [1] * len(units)], dtype=np.float64),
        feature_names=names,
        categorical_features=list(range(len(units))),
        class_names=["raw logit"],
        mode="regression",
        random_state=seed,
    )

    def predict(presence):
        texts = [" ".join(unit for keep, unit in zip(row_mask, units) if keep >= 0.5)
                 for row_mask in presence]
        vectors = explainer._encode_texts(texts)
        return explainer._head(
            vectors,
            np.repeat(cat, len(texts), axis=0),
            np.repeat(num, len(texts), axis=0),
        )

    result = lime.explain_instance(
        np.ones(len(units)), predict, num_features=len(units), num_samples=num_samples
    )
    local_weights = next(iter(result.local_exp.values()))
    lime_top = {
        index for index, _ in sorted(local_weights, key=lambda pair: abs(pair[1]), reverse=True)[:3]
    }
    shap_top = {
        index for index, item in sorted(
            enumerate(explanation["sentences"]),
            key=lambda pair: abs(pair[1]["logit_contribution"]),
            reverse=True,
        )[:3]
    }
    return len(lime_top & shap_top) / max(1, len(lime_top | shap_top))


def global_shap(explainer, rows):
    import shap

    bg = np.column_stack((explainer.background_cat, explainer.background_num)).astype(np.float64)
    masker = shap.maskers.Independent(bg, max_samples=len(bg))
    values = []
    for _, row in rows.iterrows():
        text_vector = explainer._encode_texts([row.text])
        actual_cat, actual_num = explainer._prepare_features(model_features(explainer, row))
        matrix = np.column_stack((actual_cat, actual_num)).astype(np.float64)

        def predict(candidates):
            candidates = np.asarray(candidates, dtype=np.float64)
            cats = candidates[:, :len(explainer.cat_cols)].astype(np.int64)
            nums = candidates[:, len(explainer.cat_cols):].astype(np.float32)
            return explainer._head(np.repeat(text_vector, len(candidates), axis=0), cats, nums)

        model = shap.Explainer(predict, masker, algorithm="permutation")
        explanation = model(matrix, max_evals=2 * len(explainer.players) + 1)
        values.append(np.abs(np.asarray(explanation.values)[0]))
    return np.mean(values, axis=0).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--lime-rows", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rows < 1 or args.lime_rows < 0:
        parser.error("--rows must be positive and --lime-rows cannot be negative")

    scorer = Scorer(MODEL_DIR, threads=2)
    explainer = Explainer(MODEL_DIR, scorer, threads=2)
    data = pd.read_parquet(
        DATA_PATH,
        columns=["split", "complaint_id", "text", *explainer.cat_cols, *explainer.num_cols],
    )
    test = (
        data[data.split == "test"]
        .sample(args.rows, random_state=args.seed)
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(args.seed)
    feature_abs = np.zeros(len(explainer.players), dtype=np.float64)
    faithfulness, lime_scores = [], []
    additivity = {"modality": [], "features": [], "sentences": []}
    seed_stability, feature_seed_stability = [], []
    background_stability = []
    score_parity, explain_ms = [], []

    for index, row in test.iterrows():
        started = time.perf_counter()
        result = explainer.explain(row.text, model_features(explainer, row))
        explain_ms.append((time.perf_counter() - started) * 1000)
        full_score = scorer.score(row.text, model_features(explainer, row))
        calibrated = 1.0 / (1.0 + np.exp(-(
            scorer.cal["a"] * (result["modality"]["current_logit"] + scorer.cal["offset"]) +
            scorer.cal["b"]
        )))
        score_parity.append(abs(float(calibrated) - full_score["probability"]))
        if score_parity[-1] > 1e-4:
            raise RuntimeError(f"Split graph differs from model.onnx on complaint {row.complaint_id}")
        feature = np.asarray([item["logit_contribution"] for item in result["features"]])
        feature_abs += np.abs(feature)
        additivity["modality"].append(result["modality"]["additivity_error"])
        additivity["features"].append(result["feature_additivity_error"])
        additivity["sentences"].append(result["sentence_additivity_error"])
        faithfulness.append(faithfulness_case(explainer, row, result, rng))

        if index < args.lime_rows:
            lime_scores.append(lime_agreement(explainer, row, result, 256, args.seed))
        if index < min(args.rows, 20):
            alternate = copy.copy(explainer)
            alternate.seed = args.seed + 1
            alternate_result = alternate.explain(row.text, model_features(explainer, row))
            seed_stability.append(rank_correlation(
                [x["logit_contribution"] for x in result["sentences"]],
                [x["logit_contribution"] for x in alternate_result["sentences"]],
            ))
            feature_seed_stability.append(rank_correlation(
                [x["logit_contribution"] for x in result["features"]],
                [x["logit_contribution"] for x in alternate_result["features"]],
            ))

            vector = explainer._encode_texts([row.text])
            first, second = copy.copy(explainer), copy.copy(explainer)
            first.background_text, second.background_text = explainer.background_text[:8], explainer.background_text[8:]
            first.background_cat, second.background_cat = explainer.background_cat[:8], explainer.background_cat[8:]
            first.background_num, second.background_num = explainer.background_num[:8], explainer.background_num[8:]
            cat, num = explainer._prepare_features(model_features(explainer, row))
            a = first._modality_shap(vector, cat, num)
            b = second._modality_shap(vector, cat, num)
            background_stability.append({
                "text_effect_abs_difference": abs(a["text_logit_contribution"] - b["text_logit_contribution"]),
                "track_record_effect_abs_difference": abs(
                    a["track_record_logit_contribution"] - b["track_record_logit_contribution"]
                ),
            })

    shap_importance = global_shap(explainer, test)
    importance = [
        {"feature": name, "mean_absolute_sampled_shapley": float(feature_abs[i] / len(test)),
         "mean_absolute_kernel_shap": float(shap_importance[i])}
        for i, name in enumerate(explainer.players)
    ]
    importance.sort(key=lambda item: item["mean_absolute_kernel_shap"], reverse=True)
    summary = {
        "rows": len(test),
        "seed": args.seed,
        "background_rows": len(explainer.background_text),
        "feature_importance": importance,
        "mean_additivity_error": {key: float(np.mean(value)) for key, value in additivity.items()},
        "max_additivity_error": {key: float(np.max(value)) for key, value in additivity.items()},
        "max_probability_difference_vs_model_onnx": float(np.max(score_parity)),
        "explain_latency_ms_p50": float(np.percentile(explain_ms, 50)),
        "explain_latency_ms_p95": float(np.percentile(explain_ms, 95)),
        "deletion_insertion": faithfulness,
        "lime_shap_top3_jaccard_mean": defined_mean(lime_scores),
        "seed_sentence_rank_correlation_mean": defined_mean(seed_stability),
        "seed_feature_rank_correlation_mean": defined_mean(feature_seed_stability),
        "background_modality_effect_differences": background_stability,
        "limitations": [
            "Sentence masking is an off-distribution perturbation, not a causal intervention.",
            "Global feature importance is averaged over the sampled 2024 test complaints.",
            "This script does not compute Integrated Gradients; that remains a separate offline cross-check.",
        ],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")
    print("Largest additivity errors:", summary["max_additivity_error"])


if __name__ == "__main__":
    main()
