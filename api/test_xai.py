import unittest
from types import SimpleNamespace

import numpy as np

from api.xai import Explainer, sentence_units


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        ids = [ord(char) for char in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        return SimpleNamespace(ids=ids, offsets=offsets)


class FakeEncoder:
    def run(self, names, feeds):
        mask = feeds["attention_mask"]
        values = (feeds["input_ids"] * mask).sum(axis=1).astype(np.float32) / 1000
        output = np.zeros((len(values), 128), dtype=np.float32)
        output[:, 0] = values
        return [output]


class FakeHead:
    def run(self, names, feeds):
        text = feeds["text_vector"][:, 0]
        category = feeds["cat"][:, 0]
        numeric = feeds["num"][:, 0]
        logit = text * 2 + category * 0.01 + numeric * 0.5 + text * numeric * 0.02
        return [logit.astype(np.float32)[:, None]]


class XAITests(unittest.TestCase):
    def make_explainer(self):
        explainer = Explainer.__new__(Explainer)
        explainer.scorer = SimpleNamespace(
            tok=FakeTokenizer(), cls=101, sep=102, max_body=510,
            prep={
                "cat": ["product_id"],
                "num": ["rate"],
                "vocab": {"product_id": 20},
                "log1p": [],
                "mean": {"rate": 0.0},
                "std": {"rate": 1.0},
            },
        )
        explainer.prep = explainer.scorer.prep
        explainer.cat_cols = ["product_id"]
        explainer.num_cols = ["rate"]
        explainer.players = explainer.cat_cols + explainer.num_cols
        explainer.seed = 42
        explainer.encoder = FakeEncoder()
        explainer.head = FakeHead()
        rng = np.random.default_rng(12)
        explainer.background_text = rng.normal(size=(16, 128)).astype(np.float32)
        explainer.background_cat = rng.integers(0, 20, size=(16, 1), dtype=np.int64)
        explainer.background_num = rng.normal(size=(16, 1)).astype(np.float32)
        return explainer

    def test_modality_and_feature_attributions_are_additive(self):
        explainer = self.make_explainer()
        actual_cat = np.asarray([[7]], dtype=np.int64)
        actual_num = np.asarray([[0.4]], dtype=np.float32)
        text = np.zeros((1, 128), dtype=np.float32)
        text[0, 0] = 0.3

        modality = explainer._modality_shap(text, actual_cat, actual_num)
        _, _, feature_error = explainer._feature_shap(text, actual_cat, actual_num)

        self.assertLess(modality["additivity_error"], 1e-6)
        self.assertLess(feature_error, 1e-6)

    def test_sentence_attributions_are_additive(self):
        explainer = self.make_explainer()
        text = "First sentence. [REDACTED] Last sentence."
        cat = np.asarray([[7]], dtype=np.int64)
        num = np.asarray([[0.4]], dtype=np.float32)
        vector = explainer._encode_texts([text])

        sentences, error = explainer._sentence_shap(text, vector, cat, num)

        self.assertEqual(len(sentences), 3)
        self.assertLess(error, 1e-6)

    def test_redaction_markers_are_kept_as_separate_units(self):
        self.assertEqual(sentence_units("A sentence. [DATE] More words."), ["A sentence.", "[DATE]", "More words."])


if __name__ == "__main__":
    unittest.main()
