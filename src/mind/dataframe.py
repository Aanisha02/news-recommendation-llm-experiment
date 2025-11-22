from pathlib import Path
from typing import Callable

import pandas as pd
import polars as pl

from utils.logger import logging


# ---------------------------------------------------------------------------
# Simple no-op wrapper kept only for compatibility
# ---------------------------------------------------------------------------
def read_df_function_wrapper(read_fn: Callable, cache_name: str, clear_cache: bool = False):
    """
    Minimal no-op wrapper: just call the provided read_fn.

    We ignore caching because Polars DataFrames are tricky to pickle safely
    in this environment.
    """
    return read_fn()


# ---------------------------------------------------------------------------
# NEWS
# ---------------------------------------------------------------------------
def read_news_df(path_to_tsv: Path, has_entities: bool = False, clear_cache: bool = False) -> pl.DataFrame:
    """
    Read a MIND-style news.tsv into a Polars DataFrame.

    For GossipCop we wrote the file with header:
    news_id    category    subcategory    title    abstract    url    title_entities    abstract_entities
    """
    # You *can* use pure Polars now, but to stay close to the original code we
    # go through pandas then convert.
    news_df = pd.read_csv(path_to_tsv, sep="\t", encoding="utf8")
    # If columns are missing, this will raise and show you what’s wrong.
    expected_cols = [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]
    if list(news_df.columns) != expected_cols:
        logging.warning(f"Unexpected news columns: {list(news_df.columns)} (expected {expected_cols})")

    news_pl = pl.from_pandas(news_df)
    if has_entities:
        return news_pl
    return news_pl.drop("title_entities", "abstract_entities")


# ---------------------------------------------------------------------------
# BEHAVIORS
# ---------------------------------------------------------------------------
def read_behavior_df(path_to_tsv: Path, clear_cache: bool = False) -> pl.DataFrame:
    """
    Read a MIND-style behaviors.tsv into a Polars DataFrame.

    Expected columns in the TSV:
        impression_id    user_id    time    history    impressions

    For GossipCop, `history` may be empty; `impressions` looks like:
        "N123-1 N456-0 ..."
    """
    df = pl.read_csv(
        path_to_tsv,
        separator="\t",
        has_header=True,          # header: impression_id user_id time history impressions
        infer_schema_length=0,
        null_values=["", " "],
    )

    expected_cols = ["impression_id", "user_id", "time", "history", "impressions"]
    if df.columns != expected_cols:
        logging.warning(f"Unexpected behavior columns: {df.columns} (expected {expected_cols})")

    # ---------- history: "N1 N2 N3" -> ["N1", "N2", "N3"], empty / null -> [] ----------
    def _parse_history(s: str):
        if s is None or s == "":
            return []
        return s.split(" ")

    df = df.with_columns(
        pl.col("history")
        .apply(_parse_history, skip_nulls=False)
        .alias("history")
    )

    # ---------- impressions: "N123-1 N456-0" -> list[struct(news_id, clicked)] ----------
    def _parse_impressions(s: str):
        if s is None or s == "":
            return []
        items = s.split(" ")
        out = []
        for it in items:
            if "-" not in it:
                continue
            nid, lab = it.split("-")
            try:
                lab_int = int(lab)
            except ValueError:
                lab_int = 0
            out.append({"news_id": nid, "clicked": lab_int})
        return out

    df = df.with_columns(
        pl.col("impressions")
        .apply(_parse_impressions, skip_nulls=False)
        .alias("impressions")
    )

    logging.info(df.head(1))
    return df
