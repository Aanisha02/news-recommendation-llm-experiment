from pathlib import Path

#  project root 
# ROOT_DIR = repo root (…/news-recommendation-llm)
ROOT_DIR = Path(__file__).resolve().parents[2]

#  common dirs 
DATASET_DIR = ROOT_DIR / "dataset"
CACHE_DIR = ROOT_DIR / "cache"
LOG_OUTPUT_DIR = ROOT_DIR / "logs"
MODEL_OUTPUT_DIR = ROOT_DIR / "output"


MIND_DATASET_DIR = DATASET_DIR / "mind"

# For this assignment, we reuse the "small" dataset slot for GossipCop.
# So anywhere the code expects "MIND_SMALL_*" it will actually see GossipCop.
MIND_SMALL_DATASET_DIR = DATASET_DIR / "gossipcop"
MIND_LARGE_DATASET_DIR = MIND_DATASET_DIR / "large"

#  "small" split = GossipCop 
#   dataset/gossipcop/train/{news.tsv,be
haviors.tsv}
#   dataset/gossipcop/val/{news.tsv,behaviors.tsv}
#   (test is optional)
MIND_SMALL_TRAIN_DATASET_DIR = MIND_SMALL_DATASET_DIR / "train"
MIND_SMALL_VAL_DATASET_DIR = MIND_SMALL_DATASET_DIR / "val"
MIND_SMALL_TEST_DATASET_DIR = MIND_SMALL_DATASET_DIR / "test"

MIND_LARGE_TRAIN_DATASET_DIR = MIND_LARGE_DATASET_DIR / "train"
MIND_LARGE_DEV_DATASET_DIR = MIND_LARGE_DATASET_DIR / "dev"
MIND_LARGE_TEST_DATASET_DIR = MIND_LARGE_DATASET_DIR / "test"
