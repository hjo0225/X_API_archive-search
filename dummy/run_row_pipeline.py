from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from hybrid_collect_json import (
    build_query,
    build_windows,
    fetch_counts_21d,
    fetch_search_week,
    get_bearer_token,
)


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "raw" / "Sample Celebrity.xlsx"
OUTPUT_PATH = BASE_DIR / "data" / "Sample Celebrity_test.xlsx"
JSON_DIR = BASE_DIR / "data" / "json_data"

LAGS = [("lag1w", "1w"), ("lag2w", "2w"), ("lag3w", "3w")]
FEATURE_COLUMNS = [
    "tweet_count_lag_1w",
    "impression_avg_lag_1w",
    "like_avg_lag_1w",
    "retweet_avg_lag_1w",
    "impression_max_lag_1w",
    "like_max_lag_1w",
    "retweet_max_lag_1w",
    "tweet_count_lag_2w",
    "impression_avg_lag_2w",
    "like_avg_lag_2w",
    "retweet_avg_lag_2w",
    "impression_max_lag_2w",
    "like_max_lag_2w",
    "retweet_max_lag_2w",
    "tweet_count_lag_3w",
    "impression_avg_lag_3w",
    "like_avg_lag_3w",
    "retweet_avg_lag_3w",
    "impression_max_lag_3w",
    "like_max_lag_3w",
    "retweet_max_lag_3w",
]


def _safe_file_stem(name: str) -> str:
    stem = re.sub(r'[\\/:*?"<>|]+', "_", str(name)).strip()
    return stem or "unknown"


def _to_ts(value: str) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")


def _weekly_count_from_counts(record: dict[str, Any], lag_key: str) -> int:
    windows = record.get("windows") or {}
    if lag_key == "lag1w":
        start_key, end_key = "lag1_start_time", "lag1_end_time"
    elif lag_key == "lag2w":
        start_key, end_key = "lag2_start_time", "lag2_end_time"
    else:
        start_key, end_key = "lag3_start_time", "lag3_end_time"

    start_ts = _to_ts(windows.get(start_key, ""))
    end_ts = _to_ts(windows.get(end_key, ""))
    if pd.isna(start_ts) or pd.isna(end_ts):
        return 0

    counts_data = (((record.get("counts") or {}).get("response") or {}).get("data") or [])
    total = 0
    for row in counts_data:
        bucket_start = _to_ts(row.get("start", ""))
        if pd.isna(bucket_start):
            continue
        if start_ts <= bucket_start < end_ts:
            total += int(row.get("tweet_count", 0) or 0)
    return total


def _tweets_from_search(record: dict[str, Any], lag_key: str) -> list[dict[str, Any]]:
    node = (record.get("search") or {}).get(lag_key) or {}
    pages = node.get("pages") or []
    tweets: list[dict[str, Any]] = []
    for page in pages:
        tweets.extend(page.get("data") or [])
    return tweets


def _avg_max_metrics(tweets: list[dict[str, Any]]) -> dict[str, float]:
    if not tweets:
        return {
            "impression_avg": 0.0,
            "like_avg": 0.0,
            "retweet_avg": 0.0,
            "impression_max": 0.0,
            "like_max": 0.0,
            "retweet_max": 0.0,
        }

    likes, retweets, impressions = [], [], []
    for t in tweets:
        m = t.get("public_metrics") or {}
        likes.append(float(m.get("like_count", 0) or 0))
        retweets.append(float(m.get("retweet_count", 0) or 0))
        impressions.append(float(m.get("impression_count", m.get("view_count", 0)) or 0))

    return {
        "impression_avg": float(sum(impressions) / len(impressions)),
        "like_avg": float(sum(likes) / len(likes)),
        "retweet_avg": float(sum(retweets) / len(retweets)),
        "impression_max": float(max(impressions)),
        "like_max": float(max(likes)),
        "retweet_max": float(max(retweets)),
    }


def _build_feature_row(record: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for lag_key, suffix in LAGS:
        tweet_count = _weekly_count_from_counts(record, lag_key)
        tweets = _tweets_from_search(record, lag_key)

        if tweet_count == 0:
            metrics = {
                "impression_avg": 0.0,
                "like_avg": 0.0,
                "retweet_avg": 0.0,
                "impression_max": 0.0,
                "like_max": 0.0,
                "retweet_max": 0.0,
            }
        else:
            metrics = _avg_max_metrics(tweets)

        out[f"tweet_count_lag_{suffix}"] = int(tweet_count)
        out[f"impression_avg_lag_{suffix}"] = float(metrics["impression_avg"])
        out[f"like_avg_lag_{suffix}"] = float(metrics["like_avg"])
        out[f"retweet_avg_lag_{suffix}"] = float(metrics["retweet_avg"])
        out[f"impression_max_lag_{suffix}"] = float(metrics["impression_max"])
        out[f"like_max_lag_{suffix}"] = float(metrics["like_max"])
        out[f"retweet_max_lag_{suffix}"] = float(metrics["retweet_max"])
    return out


def _init_output_frame(df: pd.DataFrame) -> pd.DataFrame:
    for col in FEATURE_COLUMNS + ["rate_limit_remaining"]:
        if col not in df.columns:
            df[col] = np.nan
    if "status" not in df.columns:
        df["status"] = pd.Series([None] * len(df), dtype="object")
    if "error" not in df.columns:
        df["error"] = pd.Series([None] * len(df), dtype="object")
    return df


def _save_json_backup(record: dict[str, Any]) -> Path:
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    path = JSON_DIR / f"{_safe_file_stem(record.get('name', 'unknown'))}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        df = pd.read_excel(OUTPUT_PATH)
        print(f"기존 출력 파일 로드: {OUTPUT_PATH}")
    else:
        df = pd.read_excel(INPUT_PATH)
        print(f"원본 입력 파일 로드: {INPUT_PATH}")

    if "group.nd.name" not in df.columns or "start_date" not in df.columns:
        raise ValueError("Input must include 'group.nd.name' and 'start_date'")

    df = _init_output_frame(df)
    token = get_bearer_token()
    headers = {"Authorization": f"Bearer {token}"}
    total = len(df)

    with requests.Session() as session:
        for pos, idx in enumerate(df.index, start=1):
            name = str(df.at[idx, "group.nd.name"]).strip()
            start_date_val = df.at[idx, "start_date"]
            search_query = build_query(name)
            counts_query = build_query(name)
            windows = build_windows(start_date_val)

            print(f"[{pos}/{total}] 처리중: {name}")

            record: dict[str, Any] = {
                "name": name,
                "start_date": str(start_date_val).split(" ")[0],
                "search_query": search_query,
                "counts_query": counts_query,
                "windows": windows,
                "collected_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "counts": None,
                "search": {"lag1w": None, "lag2w": None, "lag3w": None},
                "errors": [],
            }

            try:
                record["counts"] = fetch_counts_21d(
                    session=session,
                    headers=headers,
                    query=counts_query,
                    start_time=windows["counts_start_time"],
                    end_time=windows["counts_end_time"],
                )

                last_remaining = np.nan
                for lag_key, s_key, e_key in [
                    ("lag1w", "lag1_start_time", "lag1_end_time"),
                    ("lag2w", "lag2_start_time", "lag2_end_time"),
                    ("lag3w", "lag3_start_time", "lag3_end_time"),
                ]:
                    record["search"][lag_key] = fetch_search_week(
                        session=session,
                        headers=headers,
                        query=search_query,
                        start_time=windows[s_key],
                        end_time=windows[e_key],
                        max_results=50,
                        max_pages=1,
                    )
                    last_remaining = record["search"][lag_key].get("rate_limit", {}).get(
                        "x-rate-limit-remaining", np.nan
                    )

                features = _build_feature_row(record)
                for col in FEATURE_COLUMNS:
                    df.at[idx, col] = features.get(col, 0)
                df.at[idx, "status"] = "OK"
                df.at[idx, "error"] = np.nan
                df.at[idx, "rate_limit_remaining"] = last_remaining

                saved_json = _save_json_backup(record)
                print(
                    f"[{pos}/{total}] 완료, 남은 호출 횟수={last_remaining}, JSON={saved_json.name}"
                )

            except Exception as e:
                for col in FEATURE_COLUMNS:
                    df.at[idx, col] = np.nan
                df.at[idx, "status"] = "FAILED"
                df.at[idx, "error"] = str(e)[:500]
                df.at[idx, "rate_limit_remaining"] = np.nan

                record["errors"].append(str(e))
                saved_json = _save_json_backup(record)
                print(f"[{pos}/{total}] 실패 -> NaN 처리, 오류={e}, JSON={saved_json.name}")

            # Row-by-row durable write.
            df.to_excel(OUTPUT_PATH, index=False)

    print(f"완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_main()
