from pathlib import Path
from typing import List

import torch
import polars as pl

from transformers import AutoConfig
from recommendation.nrms import NRMS, PLMBasedNewsEncoder, UserEncoder
from utils.text import create_transform_fn_from_pretrained_tokenizer
from mind.dataframe import read_behavior_df, read_news_df
from const.path import MIND_SMALL_VAL_DATASET_DIR


# ----------------- CONFIG -----------------
PRETRAINED = "distilbert-base-uncased"
MAX_LEN = 20          # same as training
HISTORY_SIZE = 20     # same as training
K_LIST = [5, 10]

# 🔴 CHANGE this if your best run is in a different folder:
CHECKPOINT_DIR = Path("output/2025-11-16/03-19-51/checkpoint-300")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------------------------


def load_model() -> NRMS:
    cfg = AutoConfig.from_pretrained(PRETRAINED)
    hidden_size = cfg.hidden_size

    news_encoder = PLMBasedNewsEncoder(PRETRAINED)
    user_encoder = UserEncoder(hidden_size=hidden_size)
    model = NRMS(
        news_encoder=news_encoder,
        user_encoder=user_encoder,
        hidden_size=hidden_size,
        loss_fn=None,  # not needed for inference
    )

    state_dict = torch.load(CHECKPOINT_DIR / "pytorch_model.bin", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def main():
    print(f"Loading model from: {CHECKPOINT_DIR}")
    model = load_model()

    print("Loading validation data...")
    val_news_df = read_news_df(MIND_SMALL_VAL_DATASET_DIR / "news.tsv")
    val_behavior_df = read_behavior_df(MIND_SMALL_VAL_DATASET_DIR / "behaviors.tsv")

    # news_id -> title (we only need text here)
    news_id_to_title = {
        row["news_id"]: row["title"]
        for row in val_news_df.iter_rows(named=True)
    }

    transform_fn = create_transform_fn_from_pretrained_tokenizer(PRETRAINED, MAX_LEN)

    ratio_sums = {k: 0.0 for k in K_LIST}
    num_impressions = 0

    print("Computing Ratio@K on validation set...")

    for row in val_behavior_df.iter_rows(named=True):
        impressions = row["impressions"]  # list of dicts: {"news_id": ..., "clicked": ...}

        if not impressions:
            continue

        titles: List[str] = []
        labels: List[int] = []

        # Build candidate list and labels from impressions
        for imp in impressions:
            nid = imp["news_id"]
            title = news_id_to_title.get(nid)
            if title is None:
                continue
            titles.append(title)
            labels.append(int(imp["clicked"]))  # 1 = real/positive, 0 = fake/negative

        if len(titles) == 0:
            continue

        # Encode candidates: [num_cand, MAX_LEN]
        cand_input_ids = transform_fn(titles)  # [num_cand, MAX_LEN]
        num_cand = cand_input_ids.size(0)

        # Dummy history (all zeros) because GossipCop histories are empty anyway.
        history_ids = torch.zeros((1, HISTORY_SIZE, MAX_LEN), dtype=torch.long)

        # Move to device & add batch dim
        cand_input_ids = cand_input_ids.unsqueeze(0).to(DEVICE)  # [1, num_cand, MAX_LEN]
        history_ids = history_ids.to(DEVICE)

        with torch.no_grad():
            output = model(
                news_histories=history_ids,
                candidate_news=cand_input_ids,
                target=None,
            )

        scores = output.logits.flatten().cpu().numpy()

        # sort by score (desc)
        indexed = list(zip(range(num_cand), scores, labels))
        indexed.sort(key=lambda x: x[1], reverse=True)

        for K in K_LIST:
            top_k = indexed[: min(K, num_cand)]
            if len(top_k) == 0:
                continue
            # label is at index 2 in (idx, score, label)
            real_count = sum(item[2] for item in top_k)
            ratio = real_count / len(top_k)
            ratio_sums[K] += ratio

        num_impressions += 1

    print(f"Used {num_impressions} impressions.")
    if num_impressions == 0:
        print("No impressions found – check behaviors.tsv.")
        return

    print("\n=== Ratio@K (fraction of clicked / real items in top-K) ===")
    for K in K_LIST:
        avg_ratio = ratio_sums[K] / num_impressions
        print(f"Ratio@{K}: {avg_ratio:.4f}  ({avg_ratio*100:.2f}%)")


if __name__ == "__main__":
    main()
