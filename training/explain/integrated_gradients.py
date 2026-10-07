"""Offline Captum Integrated Gradients over the trained model's token embeddings.

Run on the laptop only; this is never imported by the API:
    python training/explain/integrated_gradients.py --rows 20 --steps 32
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from training.serving.export_xai_onnx import BASE, DATA, OUT, PREP, RUN, Fusion

ROOT = Path(__file__).resolve().parents[2]


class EmbeddingForward(nn.Module):
    def __init__(self, model, attention, cat, num):
        super().__init__()
        self.model = model
        self.register_buffer("attention", attention)
        self.register_buffer("cat", cat)
        self.register_buffer("num", num)

    def forward(self, word_embeddings):
        text = self.model.text(
            self.model.enc(
                inputs_embeds=word_embeddings,
                attention_mask=self.attention,
            ).last_hidden_state[:, 0]
        )
        table = self.model.tab(torch.cat(
            [embedding(self.cat[:, i]) for i, embedding in enumerate(self.model.embs)] + [self.num],
            dim=1,
        ))
        return self.model.head(torch.cat([text, table], dim=1)).reshape(-1)


def prepare_features(row):
    numeric = row[PREP["num"]].astype("float64").copy()
    numeric.loc[PREP["log1p"]] = np.log1p(numeric[PREP["log1p"]])
    numeric = ((numeric - pd.Series(PREP["mean"])) / pd.Series(PREP["std"])).to_numpy(np.float32)
    categorical = row[PREP["cat"]].to_numpy(np.int64)
    return torch.from_numpy(categorical[None]), torch.from_numpy(numeric[None])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rows < 1 or args.steps < 2:
        parser.error("--rows must be positive and --steps at least 2")

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    tokenizer.add_special_tokens({"additional_special_tokens": PREP["markers"]})
    rust_tokenizer = Tokenizer.from_file(str(OUT / "tokenizer.json"))
    model = Fusion(tokenizer)
    model.load_state_dict(torch.load(RUN / "model.pt", map_location="cpu"), strict=True)
    model.eval()

    data = pd.read_parquet(
        DATA, columns=["split", "complaint_id", "text", *PREP["cat"], *PREP["num"]]
    )
    sample = data[data.split == "test"].sample(args.rows, random_state=args.seed).reset_index(drop=True)
    output = []
    for _, row in sample.iterrows():
        body = rust_tokenizer.encode(row.text, add_special_tokens=False).ids
        ids = [rust_tokenizer.token_to_id("[CLS]")] + body[: PREP["max_len"] - 2] + [
            rust_tokenizer.token_to_id("[SEP]")
        ]
        input_ids = torch.tensor([ids], dtype=torch.long)
        pad_id = rust_tokenizer.token_to_id("[PAD]")
        baseline_ids = torch.full_like(input_ids, pad_id)
        baseline_ids[0, 0] = input_ids[0, 0]
        baseline_ids[0, -1] = input_ids[0, -1]
        attention = torch.ones_like(input_ids)
        cat, num = prepare_features(row)
        forward = EmbeddingForward(model, attention, cat, num)
        embedding = model.enc.embeddings.word_embeddings
        actual = embedding(input_ids).detach()
        baseline = embedding(baseline_ids).detach()
        attributions, delta = IntegratedGradients(forward).attribute(
            actual,
            baselines=baseline,
            n_steps=args.steps,
            return_convergence_delta=True,
        )
        token_scores = attributions.sum(dim=-1)[0].detach().cpu().numpy()
        output.append({
            "complaint_id": int(row.complaint_id),
            "logit": float(forward(actual).item()),
            "completeness_delta": float(delta.reshape(-1)[0].item()),
            "tokens": [
                {"token": rust_tokenizer.id_to_token(token_id), "logit_contribution": float(score)}
                for token_id, score in zip(ids, token_scores)
            ],
        })

    path = ROOT / "training" / "results" / "integrated_gradients_v3.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "rows": len(output),
        "steps": args.steps,
        "seed": args.seed,
        "baseline": "PAD embeddings at all body positions; actual [CLS]/[SEP] retained",
        "attributions": output,
    }, indent=2) + "\n")
    print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
