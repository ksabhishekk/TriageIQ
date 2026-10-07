"""Explain the existing fusion model without changing its scoring path."""
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort


MAX_SENTENCES = 12
SENTENCE_PERMUTATIONS = 4
FEATURE_PERMUTATIONS = 8
BACKGROUND_ROWS = 16
SEED = 42


def sentence_units(text: str, limit: int = MAX_SENTENCES) -> list[str]:
    units = []
    for chunk in re.split(r"(?<=[.!?])\s+", text):
        cursor = 0
        for marker in re.finditer(r"\[(?:DATE|REDACTED)\]", chunk):
            before = chunk[cursor : marker.start()].strip()
            if before:
                units.append(before)
            units.append(marker.group())
            cursor = marker.end()
        after = chunk[cursor:].strip()
        if after:
            units.append(after)
    if len(units) > limit:
        units = units[: limit - 1] + [" ".join(units[limit - 1 :])]
    return units


class Explainer:
    def __init__(self, model_dir: Path, scorer, threads: int = 2):
        encoder_path = model_dir / "xai_encoder.onnx"
        head_path = model_dir / "xai_head.onnx"
        background_path = model_dir / "xai_background.npz"
        missing = [p.name for p in (encoder_path, head_path, background_path) if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"XAI artifacts are missing: {', '.join(missing)}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.encoder = ort.InferenceSession(
            str(encoder_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.head = ort.InferenceSession(
            str(head_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.scorer = scorer
        self.prep = scorer.prep
        self.cat_cols = list(self.prep["cat"])
        self.num_cols = list(self.prep["num"])
        self.players = self.cat_cols + self.num_cols
        self.seed = SEED

        with np.load(background_path) as data:
            self.background_text = data["text_vector"].astype(np.float32)
            self.background_cat = data["cat"].astype(np.int64)
            self.background_num = data["num"].astype(np.float32)
        expected_cat = (BACKGROUND_ROWS, len(self.cat_cols))
        expected_num = (BACKGROUND_ROWS, len(self.num_cols))
        if self.background_text.shape != (BACKGROUND_ROWS, 128):
            raise ValueError(f"XAI background text vectors must have shape {(BACKGROUND_ROWS, 128)}")
        if self.background_cat.shape != expected_cat or self.background_num.shape != expected_num:
            raise ValueError("XAI background features do not match preprocessing.json")

    def _encode_ids(self, token_sequences: list[list[int]]) -> np.ndarray:
        batch_size = len(token_sequences)
        width = max(map(len, token_sequences))
        ids = np.zeros((batch_size, width), dtype=np.int64)
        mask = np.zeros_like(ids)
        for row, sequence in enumerate(token_sequences):
            ids[row, : len(sequence)] = sequence
            mask[row, : len(sequence)] = 1
        return self.encoder.run(
            ["text_vector"], {"input_ids": ids, "attention_mask": mask}
        )[0].astype(np.float32)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        token_sequences = []
        for text in texts:
            body = self.scorer.tok.encode(text, add_special_tokens=False).ids
            token_sequences.append([self.scorer.cls] + body[: self.scorer.max_body] + [self.scorer.sep])
        vectors = []
        for start in range(0, len(token_sequences), 8):
            vectors.extend(self._encode_ids(token_sequences[start : start + 8]))
        return np.asarray(vectors, dtype=np.float32)

    def _prepare_features(self, inputs: dict) -> tuple[np.ndarray, np.ndarray]:
        for column in self.cat_cols:
            if not 0 <= int(inputs[column]) < self.prep["vocab"][column]:
                raise ValueError(f"{column}={inputs[column]} is outside what the model was trained on")
        cat = np.asarray([[int(inputs[column]) for column in self.cat_cols]], dtype=np.int64)
        values = np.asarray([float(inputs[column]) for column in self.num_cols], dtype=np.float64)
        for column in self.prep["log1p"]:
            values[self.num_cols.index(column)] = np.log1p(values[self.num_cols.index(column)])
        values = ((values - np.asarray([self.prep["mean"][c] for c in self.num_cols])) /
                  np.asarray([self.prep["std"][c] for c in self.num_cols])).astype(np.float32)
        return cat, values[None, :]

    def _head(self, text, cats, nums) -> np.ndarray:
        text = np.asarray(text, dtype=np.float32)
        cats = np.asarray(cats, dtype=np.int64)
        nums = np.asarray(nums, dtype=np.float32)
        return self.head.run(
            ["logit"], {"text_vector": text, "cat": cats, "num": nums}
        )[0].reshape(-1).astype(np.float64)

    def _feature_shap(self, text_vector, actual_cat, actual_num) -> tuple[list[dict], float, float]:
        rng = np.random.default_rng(self.seed)
        backgrounds = min(BACKGROUND_ROWS, self.background_text.shape[0])
        cat_rows, num_rows = [], []
        sampled_orders = []
        for row in range(backgrounds):
            for _ in range(FEATURE_PERMUTATIONS):
                cat = self.background_cat[row].copy()
                num = self.background_num[row].copy()
                permutation = rng.permutation(len(self.players))
                sampled_orders.append(permutation)
                states = [(cat.copy(), num.copy())]
                for player_idx in permutation:
                    column = self.players[player_idx]
                    if column in self.cat_cols:
                        cat[self.cat_cols.index(column)] = actual_cat[0, self.cat_cols.index(column)]
                    else:
                        num[self.num_cols.index(column)] = actual_num[0, self.num_cols.index(column)]
                    states.append((cat.copy(), num.copy()))
                cat_rows.extend(state[0] for state in states)
                num_rows.extend(state[1] for state in states)

        total_states = len(cat_rows)
        logits = np.concatenate([
            self._head(
                np.repeat(text_vector, min(512, total_states - start), axis=0),
                np.asarray(cat_rows[start : start + 512]),
                np.asarray(num_rows[start : start + 512]),
            )
            for start in range(0, total_states, 512)
        ])
        nplayers = len(self.players)
        contributions = np.zeros(nplayers, dtype=np.float64)
        baseline_total = 0.0
        offset = 0
        for permutation in sampled_orders:
            path = logits[offset : offset + nplayers + 1]
            offset += nplayers + 1
            baseline_total += path[0]
            contributions[permutation] += np.diff(path)

        paths = len(sampled_orders)
        contributions /= paths
        baseline = baseline_total / paths
        current = float(self._head(text_vector, actual_cat, actual_num)[0])
        entries = [
            {"feature": name, "value": float(
                actual_cat[0, self.cat_cols.index(name)] if name in self.cat_cols
                else actual_num[0, self.num_cols.index(name)]
            ), "logit_contribution": float(contributions[i])}
            for i, name in enumerate(self.players)
        ]
        return entries, baseline, abs(baseline + contributions.sum() - current)

    def _modality_shap(self, text_vector, actual_cat, actual_num) -> dict:
        rows = self.background_text.shape[0]
        bg_text = self.background_text[:rows]
        bg_cat = self.background_cat[:rows]
        bg_num = self.background_num[:rows]
        actual_text = np.repeat(text_vector, rows, axis=0)
        actual_cat_rows = np.repeat(actual_cat, rows, axis=0)
        actual_num_rows = np.repeat(actual_num, rows, axis=0)

        f00 = self._head(bg_text, bg_cat, bg_num)
        f10 = self._head(actual_text, bg_cat, bg_num)
        f01 = self._head(bg_text, actual_cat_rows, actual_num_rows)
        f11 = self._head(actual_text, actual_cat_rows, actual_num_rows)
        text_effect = float(np.mean(0.5 * ((f10 - f00) + (f11 - f01))))
        table_effect = float(np.mean(0.5 * ((f01 - f00) + (f11 - f10))))
        baseline = float(np.mean(f00))
        current = float(np.mean(f11))
        return {
            "text_logit_contribution": text_effect,
            "track_record_logit_contribution": table_effect,
            "baseline_logit": baseline,
            "current_logit": current,
            "additivity_error": abs(baseline + text_effect + table_effect - current),
        }

    def _sentence_shap(self, clean_text: str, text_vector, actual_cat, actual_num) -> tuple[list[dict], float]:
        body = self.scorer.tok.encode(clean_text, add_special_tokens=False)
        visible = clean_text
        if len(body.ids) > self.scorer.max_body and body.offsets:
            visible = clean_text[: body.offsets[self.scorer.max_body - 1][1]]
        units = sentence_units(visible)
        rng = np.random.default_rng(self.seed)
        coalition_texts: dict[tuple[int, ...], str] = {}
        paths = []
        for _ in range(SENTENCE_PERMUTATIONS):
            order = rng.permutation(len(units)).tolist()
            active: list[int] = []
            path = [tuple()]
            for index in order:
                active.append(index)
                path.append(tuple(sorted(active)))
            paths.append((order, path))
            for coalition in path:
                coalition_texts[coalition] = " ".join(units[i] for i in coalition)

        keys = list(coalition_texts)
        vectors = []
        for start in range(0, len(keys), 8):
            vectors.extend(self._encode_texts([coalition_texts[k] for k in keys[start : start + 8]]))
        text_by_coalition = dict(zip(keys, vectors))
        feature_row = np.repeat(actual_cat, len(keys), axis=0)
        numeric_row = np.repeat(actual_num, len(keys), axis=0)
        text_rows = np.asarray([text_by_coalition[k] for k in keys], dtype=np.float32)
        logits = self._head(text_rows, feature_row, numeric_row)
        scores = dict(zip(keys, logits))
        contributions = np.zeros(len(units), dtype=np.float64)
        for order, path in paths:
            values = np.asarray([scores[coalition] for coalition in path])
            contributions[order] += np.diff(values)
        contributions /= SENTENCE_PERMUTATIONS
        empty_logit = float(scores[tuple()])
        full_logit = float(scores[tuple(range(len(units)))])
        entries = [
            {"text": sentence, "logit_contribution": float(contributions[i])}
            for i, sentence in enumerate(units)
        ]
        return entries, abs(empty_logit + contributions.sum() - full_logit)

    def explain(self, clean_text: str, inputs: dict) -> dict:
        cat, num = self._prepare_features(inputs)
        original_body = self.scorer.tok.encode(clean_text, add_special_tokens=False).ids
        ids = [self.scorer.cls] + original_body[: self.scorer.max_body] + [self.scorer.sep]
        text_vector = self._encode_ids([ids])
        modality = self._modality_shap(text_vector, cat, num)
        feature_values, feature_baseline, feature_error = self._feature_shap(text_vector, cat, num)
        sentences, sentence_error = self._sentence_shap(clean_text, text_vector, cat, num)
        return {
            "modality": modality,
            "features": feature_values,
            "feature_baseline_logit": feature_baseline,
            "feature_additivity_error": feature_error,
            "sentences": sentences,
            "sentence_additivity_error": sentence_error,
            "text_word_pieces": len(original_body),
            "text_cut_at_512": len(original_body) > self.scorer.max_body,
        }
