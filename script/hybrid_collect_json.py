from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "raw" / "Sample Celebrity.xlsx"
JSON_DIR = BASE_DIR / "data" / "json_data"

SEARCH_ALL_URL = "https://api.x.com/2/tweets/search/all"
COUNTS_ALL_URL = "https://api.x.com/2/tweets/counts/all"


def get_bearer_token() -> str:
    token = (
        os.getenv("X_BEARER_TOKEN")
        or os.getenv("TWITTER_BEARER_TOKEN")
        or os.getenv("BEARER_TOKEN")
    )
    if not token:
        raise RuntimeError(
            "Bearer token not found. Set X_BEARER_TOKEN, TWITTER_BEARER_TOKEN, or BEARER_TOKEN."
        )
    return token


def _parse_start_date(value: Any) -> datetime:
    if hasattr(value, "to_pydatetime"):
        dt = value.to_pydatetime()
        return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
    if isinstance(value, datetime):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    date_text = str(value).split(" ")[0]
    return datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _iso_z(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_query(group_name: str) -> str:
    safe_name = str(group_name).replace('"', '\\"').strip()
    return f'"{safe_name}" lang:ko -is:retweet -is:nullcast'


def build_windows(start_date_value: Any) -> dict[str, str]:
    end = _parse_start_date(start_date_value)
    lag1_start = end - timedelta(days=7)
    lag2_start = end - timedelta(days=14)
    lag3_start = end - timedelta(days=21)
    return {
        "lag1_start_time": _iso_z(lag1_start),
        "lag1_end_time": _iso_z(end),
        "lag2_start_time": _iso_z(lag2_start),
        "lag2_end_time": _iso_z(lag1_start),
        "lag3_start_time": _iso_z(lag3_start),
        "lag3_end_time": _iso_z(lag2_start),
        "counts_start_time": _iso_z(lag3_start),
        "counts_end_time": _iso_z(end),
    }


def _safe_file_stem(name: str) -> str:
    stem = re.sub(r"[\\/:*?\"<>|]+", "_", str(name)).strip()
    return stem or "unknown"


def _request_json(
    session: requests.Session,
    url: str,
    headers: dict[str, str],
    params: dict[str, Any],
    timeout: int = 30,
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = session.get(url, headers=headers, params=params, timeout=timeout)
    remaining = response.headers.get("x-rate-limit-remaining")
    reset_time = response.headers.get("x-rate-limit-reset")
    print(f"남은 호출 횟수: {remaining}, 초기화 시간: {reset_time}")

    response.raise_for_status()
    return response.json(), {
        "x-rate-limit-remaining": remaining,
        "x-rate-limit-reset": reset_time,
    }


def fetch_counts_21d(
    session: requests.Session,
    headers: dict[str, str],
    query: str,
    start_time: str,
    end_time: str,
) -> dict[str, Any]:
    params = {
        "query": query,
        "start_time": start_time,
        "end_time": end_time,
        "granularity": "day",
    }
    payload, rate = _request_json(session, COUNTS_ALL_URL, headers, params)
    return {"request": params, "rate_limit": rate, "response": payload}


def fetch_search_week(
    session: requests.Session,
    headers: dict[str, str],
    query: str,
    start_time: str,
    end_time: str,
    max_results: int = 50,
    max_pages: int = 1,
    sort_order: str = "relevancy",
) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    collected_data: list[dict[str, Any]] = []
    next_token: str | None = None
    last_rate: dict[str, Any] = {}

    effective_pages = min(max_pages, 1)
    for _ in range(effective_pages):
        time.sleep(1.1)
        params = {
            "query": query,
            "start_time": start_time,
            "end_time": end_time,
            "max_results": max_results,
            "sort_order": sort_order,
            "tweet.fields": "public_metrics,created_at",
        }
        if next_token:
            params["next_token"] = next_token

        payload, rate = _request_json(session, SEARCH_ALL_URL, headers, params)
        last_rate = rate
        pages.append(payload)
        collected_data.extend(payload.get("data", []))
        next_token = payload.get("meta", {}).get("next_token")

        if not next_token:
            break

    return {
        "request": {
            "query": query,
            "start_time": start_time,
            "end_time": end_time,
            "max_results": max_results,
            "max_pages": effective_pages,
            "sort_order": sort_order,
            "tweet.fields": ["public_metrics", "created_at"],
        },
        "rate_limit": last_rate,
        "pages": pages,
        "count": len(collected_data),
        "has_next_token": bool(next_token),
    }


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(INPUT_PATH)
    if "group.nd.name" not in df.columns or "start_date" not in df.columns:
        raise ValueError("Input must include 'group.nd.name' and 'start_date'")

    token = get_bearer_token()
    headers = {"Authorization": f"Bearer {token}"}

    with requests.Session() as session:
        total = len(df)
        for i, row in df.iterrows():
            idx = i + 1
            name = str(row["group.nd.name"]).strip()
            start_date = row["start_date"]
            search_query = build_query(name)
            counts_query = build_query(name)
            windows = build_windows(start_date)

            print(f"[{idx}/{total}] 수집 시작: {name}")
            record: dict[str, Any] = {
                "name": name,
                "start_date": str(start_date).split(" ")[0],
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
            except Exception as e:
                record["errors"].append(f"counts_error: {e}")
                print(f"[{idx}/{total}] Counts 실패: {e}")

            for lag_key, s_key, e_key in [
                ("lag1w", "lag1_start_time", "lag1_end_time"),
                ("lag2w", "lag2_start_time", "lag2_end_time"),
                ("lag3w", "lag3_start_time", "lag3_end_time"),
            ]:
                try:
                    record["search"][lag_key] = fetch_search_week(
                        session=session,
                        headers=headers,
                        query=search_query,
                        start_time=windows[s_key],
                        end_time=windows[e_key],
                        max_results=50,
                        max_pages=1,
                    )
                except Exception as e:
                    record["errors"].append(f"{lag_key}_search_error: {e}")
                    print(f"[{idx}/{total}] {lag_key} Search 실패: {e}")

            out_file = JSON_DIR / f"{_safe_file_stem(name)}.json"
            out_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{idx}/{total}] 저장 완료: {out_file}")


if __name__ == "__main__":
    main()
