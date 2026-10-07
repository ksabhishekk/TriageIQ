"""Export separate text-encoder and fusion-head ONNX graphs for opt-in explanations.

Run on the Mac after restoring the v3 model and training parquet:
    python training/serving/export_xai_onnx.py

The script does not overwrite model.onnx or any existing serving artifact. It writes the three
XAI artifacts only after the split graphs match the shipped graph on 1,000 held-out complaints.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModel, AutoTokenizer

RUN = Path("training/outputs/fusion_distilbert_full")
OUT = Path("training/outputs/serving_v3")
DATA = Path("Data/data/interim/triageiq_training_v3.parquet")
PREP = json.loads((RUN / "preprocessing.json").read_text())
BASE = "distilbert-base-uncased"
EMB = {"product_id": 4, "sub_product_id": 8, "issue_id": 12, "state_id": 8}
BACKGROUND_ROWS = 16
SEED = 42


class Fusion(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        cfg = AutoConfig.from_pretrained(BASE)
        self.enc = AutoModel.from_config(cfg, attn_implementation="eager")
        self.enc.resize_token_embeddings(len(tokenizer))
        self.text = nn.Sequential(nn.Linear(cfg.hidden_size, 128), nn.GELU(), nn.Dropout(0.1))
        self.embs = nn.ModuleList([nn.Embedding(PREP["vocab"][c], EMB[c]) for c in PREP["cat"]])
        self.tab = nn.Sequential(
            nn.Linear(sum(EMB.values()) + len(PREP["num"]), 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(128 + 64, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, ids, attention, cat, num):
        text = self.text(self.enc(input_ids=ids, attention_mask=attention).last_hidden_state[:, 0])
        table = self.tab(torch.cat(
            [embedding(cat[:, i]) for i, embedding in enumerate(self.embs)] + [num], dim=1
        ))
        return self.head(torch.cat([text, table], dim=1)).squeeze(1)


class TextGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.enc, self.text = model.enc, model.text

    def forward(self, ids, attention):
        return self.text(self.enc(input_ids=ids, attention_mask=attention).last_hidden_state[:, 0])


class HeadGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embs, self.tab, self.head = model.embs, model.tab, model.head

    def forward(self, text, cat, num):
        table = self.tab(torch.cat(
            [embedding(cat[:, i]) for i, embedding in enumerate(self.embs)] + [num], dim=1
        ))
        return self.head(torch.cat([text, table], dim=1)).squeeze(1)


def encoded_batch(tokenizer, texts, max_body):
    sequences = [
        [tokenizer.token_to_id("[CLS]")] + encoding.ids[:max_body] + [tokenizer.token_to_id("[SEP]")]
        for encoding in tokenizer.encode_batch(texts, add_special_tokens=False)
    ]
    width = max(map(len, sequences))
    ids = np.zeros((len(sequences), width), dtype=np.int64)
    attention = np.zeros_like(ids)
    for row, sequence in enumerate(sequences):
        ids[row, :len(sequence)] = sequence
        attention[row, :len(sequence)] = 1
    return ids, attention


def features(frame):
    numeric = frame[PREP["num"]].astype("float64").copy()
    numeric.loc[:, PREP["log1p"]] = np.log1p(numeric[PREP["log1p"]])
    numeric = ((numeric - pd.Series(PREP["mean"])) / pd.Series(PREP["std"])).to_numpy(np.float32)
    categorical = frame[PREP["cat"]].to_numpy(np.int64)
    return categorical, numeric


def main():
    for path in (RUN / "model.pt", OUT / "model.onnx", DATA):
        if not path.is_file():
            raise FileNotFoundError(f"Required v3 artifact not found: {path}")

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    tokenizer.add_special_tokens({"additional_special_tokens": PREP["markers"]})
    serving_tokenizer = Tokenizer.from_file(str(OUT / "tokenizer.json"))
    if serving_tokenizer.token_to_id("[CLS]") is None or serving_tokenizer.token_to_id("[SEP]") is None:
        raise RuntimeError("The serving tokenizer is missing [CLS] or [SEP].")
    model = Fusion(tokenizer)
    model.load_state_dict(torch.load(RUN / "model.pt", map_location="cpu"), strict=True)
    model.eval()

    data = pd.read_parquet(
        DATA, columns=["split", "complaint_id", "text", *PREP["cat"], *PREP["num"]]
    )
    test = data[data.split == "test"].sample(1000, random_state=SEED).reset_index(drop=True)
    training_ids = tokenizer(test.text.tolist(), add_special_tokens=False, truncation=False)["input_ids"]
    serving_ids = [encoding.ids for encoding in serving_tokenizer.encode_batch(
        test.text.tolist(), add_special_tokens=False
    )]
    tokenizer_matches = sum(left == right for left, right in zip(training_ids, serving_ids))
    if tokenizer_matches != len(test):
        raise RuntimeError(
            f"XAI tokenizer parity failed: {tokenizer_matches}/{len(test)} identical token sequences"
        )
    background = (
        data[data.split == "test"]
        .sample(BACKGROUND_ROWS, random_state=SEED)
        .sort_values("complaint_id")
        .reset_index(drop=True)
    )
    # Use only intake-time text/features from the natural-prevalence serving demo rows; labels were not loaded.
    background = background[["text", *PREP["cat"], *PREP["num"]]]
    bg_cat, bg_num = features(background)

    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xai-export-", dir=OUT) as temp_dir:
        temp = Path(temp_dir)
        text_graph = TextGraph(model).eval()
        head_graph = HeadGraph(model).eval()
        example_ids, example_attention = encoded_batch(
            serving_tokenizer, [test.text.iloc[0]], PREP["max_len"] - 2
        )
        torch.onnx.export(
            text_graph,
            (torch.from_numpy(example_ids), torch.from_numpy(example_attention)),
            temp / "xai_encoder.onnx",
            input_names=["input_ids", "attention_mask"],
            output_names=["text_vector"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "text_vector": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
        torch.onnx.export(
            head_graph,
            (torch.zeros((1, 128)), torch.from_numpy(bg_cat[:1]), torch.from_numpy(bg_num[:1])),
            temp / "xai_head.onnx",
            input_names=["text_vector", "cat", "num"],
            output_names=["logit"],
            dynamic_axes={
                "text_vector": {0: "batch"},
                "cat": {0: "batch"},
                "num": {0: "batch"},
                "logit": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )

        full = ort.InferenceSession(str(OUT / "model.onnx"), providers=["CPUExecutionProvider"])
        encoder = ort.InferenceSession(str(temp / "xai_encoder.onnx"), providers=["CPUExecutionProvider"])
        head = ort.InferenceSession(str(temp / "xai_head.onnx"), providers=["CPUExecutionProvider"])
        cat, num = features(test)
        full_logits, split_logits = [], []
        for start in range(0, len(test), 8):
            ids, attention = encoded_batch(
                serving_tokenizer, test.text.iloc[start:start + 8].tolist(), PREP["max_len"] - 2
            )
            batch_cat = cat[start:start + len(ids)]
            batch_num = num[start:start + len(ids)]
            reference = full.run(
                ["logit"], {"input_ids": ids, "attention_mask": attention, "cat": batch_cat, "num": batch_num}
            )[0].reshape(-1)
            vector = encoder.run(["text_vector"], {"input_ids": ids, "attention_mask": attention})[0]
            candidate = head.run(
                ["logit"], {"text_vector": vector, "cat": batch_cat, "num": batch_num}
            )[0].reshape(-1)
            full_logits.extend(reference)
            split_logits.extend(candidate)
        max_abs = float(np.max(np.abs(np.asarray(full_logits) - np.asarray(split_logits))))
        if max_abs > 1e-5:
            raise RuntimeError(f"XAI split-graph parity failed: max |logit difference|={max_abs:.8g} (limit 1e-5)")

        vectors = []
        for start in range(0, BACKGROUND_ROWS, 8):
            ids, attention = encoded_batch(
                serving_tokenizer, background.text.iloc[start:start + 8].tolist(), PREP["max_len"] - 2
            )
            vectors.extend(encoder.run(["text_vector"], {"input_ids": ids, "attention_mask": attention})[0])
        np.savez_compressed(
            temp / "xai_background.npz",
            text_vector=np.asarray(vectors, dtype=np.float32),
            cat=bg_cat,
            num=bg_num,
        )

        for name in ("xai_encoder.onnx", "xai_head.onnx", "xai_background.npz"):
            shutil.copy2(temp / name, OUT / name)
    parity = {
        "test_rows": len(test),
        "tokenizer_identical_rows": tokenizer_matches,
        "max_abs_logit_difference": max_abs,
        "parity_limit": 1e-5,
        "passed": True,
    }
    (ROOT / "training" / "results" / "xai_onnx_v3.json").write_text(
        json.dumps(parity, indent=2) + "\n"
    )
    print(f"XAI split-graph parity passed: {len(test)} test complaints; max |Δlogit|={max_abs:.8g}")
    print("Created xai_encoder.onnx, xai_head.onnx and xai_background.npz.")


if __name__ == "__main__":
    main()
