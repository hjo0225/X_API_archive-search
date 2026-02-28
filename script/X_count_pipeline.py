from __future__ import annotations

import os
import re
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "raw" / "Unique_Celebrity_Keyword.X.csv"
OUTPUT_PATH = BASE_DIR / "data" / "Celebrity_count.xlsx"
JSON_DIR = BASE_DIR / "data" / "json"
COUNTS_ALL_URL = "https://api.x.com/2/tweets/counts/all"
JSON_DIR.mkdir(parents=True, exist_ok=True)


def get_bearer_token() -> str:
    token = os.getenv("X_BEARER_TOKEN")
    if not token:
        raise RuntimeError("X_BEARER_TOKEN not found in environment variables.")
    return token


def safe_filename(value: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", str(value)).strip()
    return cleaned or "unknown"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_input_path(input_path: Path) -> Path:
    candidates = [
        input_path,
        BASE_DIR / "data" / "Unique_Celebrity_Keyword.X.xlsx",
        BASE_DIR / "data" / "Unique_Celebrity_Keyword.X.csv",
        BASE_DIR / "raw" / "Unique_Celebrity_Keyword.X.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Input file not found. Tried: " + ", ".join(str(p) for p in candidates)
    )


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def load_dataframe(input_path: Path, output_path: Path) -> tuple[pd.DataFrame, str]:
    if output_path.exists():
        df = pd.read_excel(output_path)
        source = "resume"
    else:
        resolved_input = _resolve_input_path(input_path)
        df = _read_table(resolved_input)
        source = "fresh"

    required_cols = ["variants_joined", "group.nd.name", "collab_start_week"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if source == "fresh":
        df["status"] = "PENDING"
        df["error_message"] = pd.NA
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_path, index=False)

    return df, source


def parse_iso_week(s: str) -> tuple[int, int]:
    text = str(s).strip()
    patterns = [
        r"^(\d{4})-W(\d{1,2})$",
        r"^(\d{4})W(\d{1,2})$",
        r"^(\d{4})-(\d{1,2})$",
        r"^(\d{4})(\d{2})$",
    ]
    for pattern in patterns:
        m = re.match(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        year = int(m.group(1))
        week = int(m.group(2))
        if 1 <= week <= 53:
            return year, week
        raise ValueError(f"Invalid ISO week number: {week}")
    raise ValueError(f"Invalid ISO week format: {s}")


def iso_week_monday(year: int, week: int) -> date:
    try:
        return datetime.fromisocalendar(int(year), int(week), 1).date()
    except ValueError as e:
        raise ValueError(f"Invalid ISO year/week: {year}-W{week:02d}") from e


def compute_collab_start_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "status" not in out.columns:
        out["status"] = "PENDING"
    if "error_message" not in out.columns:
        out["error_message"] = pd.NA
    if "collab_start_date" not in out.columns:
        out["collab_start_date"] = pd.NA

    for idx in out.index:
        status_val = str(out.at[idx, "status"]).strip().upper()
        if status_val == "OK":
            continue

        raw_week = out.at[idx, "collab_start_week"]
        try:
            year, week = parse_iso_week(str(raw_week))
            normalized_week = f"{year:04d}-W{week:02d}"
            monday = iso_week_monday(year, week)
            out.at[idx, "collab_start_week"] = normalized_week
            out.at[idx, "collab_start_date"] = monday.strftime("%Y-%m-%d")
        except Exception as e:
            out.at[idx, "status"] = "ERROR"
            out.at[idx, "error_message"] = str(e)

    return out


def build_time_window(collab_start_date_str: str) -> tuple[str, str]:
    try:
        collab_date = datetime.strptime(str(collab_start_date_str), "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"Invalid collab_start_date format: {collab_start_date_str}") from e

    start_date = collab_date + relativedelta(months=-6)
    end_date = collab_date + relativedelta(months=-2)
    start_iso = f"{start_date.strftime('%Y-%m-%d')}T00:00:00Z"
    end_iso = f"{end_date.strftime('%Y-%m-%d')}T00:00:00Z"
    return start_iso, end_iso


def fill_windows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "window_start" not in out.columns:
        out["window_start"] = pd.NA
    if "window_end" not in out.columns:
        out["window_end"] = pd.NA
    if "status" not in out.columns:
        out["status"] = "PENDING"
    if "error_message" not in out.columns:
        out["error_message"] = pd.NA

    for idx in out.index:
        status_val = str(out.at[idx, "status"]).strip().upper()
        if status_val == "OK":
            continue

        try:
            start_iso, end_iso = build_time_window(str(out.at[idx, "collab_start_date"]))
            out.at[idx, "window_start"] = start_iso
            out.at[idx, "window_end"] = end_iso
        except Exception as e:
            out.at[idx, "status"] = "ERROR"
            out.at[idx, "error_message"] = str(e)

    return out


def build_query(variants_joined: str) -> str:
    if pd.isna(variants_joined):
        raise ValueError("variants_joined is NaN")

    text = str(variants_joined).strip()
    if not text:
        raise ValueError("variants_joined is empty")

    # CSV stores variants as "a | b | c"; X query syntax requires "OR".
    parts = [p.strip().replace('"', "") for p in text.split("|")]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("No valid query variants after parsing variants_joined")

    # Deduplicate while preserving order.
    deduped = list(dict.fromkeys(parts))
    clause = deduped[0] if len(deduped) == 1 else f"({' OR '.join(deduped)})"
    query = f"{clause} lang:ko -is:retweet -is:nullcast"
    if len(query) > 1024:
        raise ValueError(f"Query too long: {len(query)} > 1024")

    return query


def fetch_counts_all_day(
    query: str,
    start_iso: str,
    end_iso: str,
    bearer_token: str,
    timeout: int = 30,
) -> tuple[dict, dict]:
    headers = {"Authorization": f"Bearer {bearer_token}"}
    all_data: list[dict] = []
    next_token: str | None = None
    page_count = 0
    last_meta: dict | None = None
    rate_meta = {"limit": None, "remaining": None, "reset": None}

    while True:
        params = {
            "query": query,
            "start_time": start_iso,
            "end_time": end_iso,
            "granularity": "day",
        }
        if next_token:
            params["next_token"] = next_token

        response = requests.get(COUNTS_ALL_URL, headers=headers, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()

        all_data.extend(payload.get("data", []) or [])
        last_meta = payload.get("meta", {}) or {}
        page_count += 1

        rate_meta = {
            "limit": response.headers.get("x-rate-limit-limit"),
            "remaining": response.headers.get("x-rate-limit-remaining"),
            "reset": response.headers.get("x-rate-limit-reset"),
        }

        next_token = last_meta.get("next_token")
        if not next_token:
            break

    combined_payload = {"data": all_data, "meta": last_meta, "page_count": page_count}
    return combined_payload, rate_meta


def aggregate_daily_to_iso_week(counts_data) -> dict[str, int]:
    weekly_counts: dict[str, int] = {}
    for row in counts_data or []:
        start_raw = row.get("start")
        if not start_raw:
            continue

        dt = pd.to_datetime(start_raw, utc=True, errors="coerce")
        if pd.isna(dt):
            continue

        iso = dt.date().isocalendar()
        week_key = f"{int(iso.year):04d}-W{int(iso.week):02d}"
        tweet_count = int(row.get("tweet_count", 0) or 0)
        weekly_counts[week_key] = weekly_counts.get(week_key, 0) + tweet_count

    return weekly_counts


def iso_week_str_to_year_week(iso_week_str: str) -> tuple[int, int]:
    m = re.match(r"^(\d{4})-W(\d{2})$", str(iso_week_str).strip())
    if not m:
        raise ValueError(f"Invalid iso week string: {iso_week_str}")
    return int(m.group(1)), int(m.group(2))


def year_week_to_iso_week_str(y: int, w: int) -> str:
    return f"{int(y):04d}-W{int(w):02d}"


def subtract_weeks(y: int, w: int, k: int) -> tuple[int, int]:
    monday = datetime.fromisocalendar(int(y), int(w), 1)
    shifted = monday - timedelta(weeks=int(k))
    iso = shifted.isocalendar()
    return int(iso.year), int(iso.week)


def build_lag_features(
    collab_start_week_str: str,
    weekly_counts_dict: dict[str, int],
) -> tuple[dict[str, int | None], int | None]:
    start_y, start_w = iso_week_str_to_year_week(collab_start_week_str)
    lag_features: dict[str, int | None] = {}

    values: list[int] = []
    for k in range(9, 27):
        y2, w2 = subtract_weeks(start_y, start_w, k)
        key = year_week_to_iso_week_str(y2, w2)
        value = weekly_counts_dict.get(key)
        lag_features[f"count_lag_{k}w"] = value if value is not None else None
        if value is not None:
            values.append(int(value))

    total_count_2to6m = sum(values) if values else None
    return lag_features, total_count_2to6m


def make_ip_payload(
    group_name,
    query,
    collab_start_week,
    collab_start_date,
    window_start,
    window_end,
    weekly_counts_dict,
    lag_features_dict,
    total_count_2to6m,
    rate_meta,
) -> dict:
    return {
        "group.nd.name": group_name,
        "query": query,
        "collab_start_week": collab_start_week,
        "collab_start_date": collab_start_date,
        "window_start": window_start,
        "window_end": window_end,
        "weekly_counts": weekly_counts_dict,
        "lag_features": lag_features_dict,
        "total_count_2to6m": total_count_2to6m,
        "rate_limit": rate_meta,
        "saved_at_utc": now_iso(),
    }


def save_ip_json(json_dir, group_name, payload) -> None:
    out_dir = Path(json_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_filename(group_name)}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_one_ip(df, idx, bearer_token) -> tuple[pd.DataFrame, bool]:
    should_stop_soon_flag = False
    row = df.loc[idx]

    group_name = row.get("group.nd.name")
    collab_n = row.get("collab_n")
    variants_joined = row.get("variants_joined")
    collab_start_week = row.get("collab_start_week")
    collab_start_date = row.get("collab_start_date")
    window_start = row.get("window_start")
    window_end = row.get("window_end")
    print(
        f"[START] idx={idx} group={group_name} collab_n={collab_n} "
        f"window={window_start}~{window_end}"
    )

    try:
        y, w = parse_iso_week(str(collab_start_week))
        normalized_collab_start_week = f"{y:04d}-W{w:02d}"
        df.at[idx, "collab_start_week"] = normalized_collab_start_week
        query = build_query(variants_joined)
        print(f"[REQUEST] idx={idx} counts/all query={query}")
        payload, rate_meta = fetch_counts_all_day(
            query=query,
            start_iso=str(window_start),
            end_iso=str(window_end),
            bearer_token=bearer_token,
        )
        weekly_counts_dict = aggregate_daily_to_iso_week(payload.get("data", []))
        lag_features_dict, total_count_2to6m = build_lag_features(
            collab_start_week_str=normalized_collab_start_week,
            weekly_counts_dict=weekly_counts_dict,
        )

        for k in range(9, 27):
            col = f"count_lag_{k}w"
            if col not in df.columns:
                df[col] = pd.NA
            df.at[idx, col] = lag_features_dict.get(col)

        if "total_count_2to6m" not in df.columns:
            df["total_count_2to6m"] = pd.NA
        if "status" not in df.columns:
            df["status"] = "PENDING"
        if "error_message" not in df.columns:
            df["error_message"] = pd.NA

        df.at[idx, "total_count_2to6m"] = total_count_2to6m
        df.at[idx, "status"] = "OK"
        df.at[idx, "error_message"] = ""

        ip_payload = make_ip_payload(
            group_name=group_name,
            query=query,
            collab_start_week=normalized_collab_start_week,
            collab_start_date=collab_start_date,
            window_start=window_start,
            window_end=window_end,
            weekly_counts_dict=weekly_counts_dict,
            lag_features_dict=lag_features_dict,
            total_count_2to6m=total_count_2to6m,
            rate_meta=rate_meta,
        )
        file_name = f"{group_name}({collab_n}회차)"
        save_ip_json(JSON_DIR, file_name, ip_payload)
        print(
            f"[OK] idx={idx} saved={safe_filename(file_name)}.json "
            f"total_count_2to6m={total_count_2to6m}"
        )

        remaining_raw = rate_meta.get("remaining")
        try:
            remaining = int(remaining_raw) if remaining_raw is not None else None
        except (TypeError, ValueError):
            remaining = None
        should_stop_soon_flag = bool(remaining is not None and remaining <= 10)
        if remaining is not None:
            print(f"[RATE] idx={idx} remaining={remaining}")
        if should_stop_soon_flag:
            print("[RATE] remaining <= 10, will stop soon.")

    except Exception as e:
        if "status" not in df.columns:
            df["status"] = "PENDING"
        if "error_message" not in df.columns:
            df["error_message"] = pd.NA

        error_line = re.sub(r"[\r\n]+", " ", str(e)).strip()
        df.at[idx, "status"] = "ERROR"
        df.at[idx, "error_message"] = error_line
        print(f"[ERROR] idx={idx} group={group_name} collab_n={collab_n} msg={error_line}")

        try:
            error_payload = {
                "group.nd.name": group_name,
                "status": "ERROR",
                "error_message": error_line,
                "saved_at_utc": now_iso(),
            }
            file_name = f"{group_name}({collab_n}회차)"
            save_ip_json(JSON_DIR, file_name, error_payload)
            print(f"[ERROR-SAVED] idx={idx} saved={safe_filename(file_name)}.json")
        except Exception:
            pass
    finally:
        time.sleep(1.1)

    return df, should_stop_soon_flag


def main() -> None:
    bearer_token = get_bearer_token()
    df, _source = load_dataframe(INPUT_PATH, OUTPUT_PATH)
    df = compute_collab_start_date(df)
    df = fill_windows(df)

    for k in range(9, 27):
        col = f"count_lag_{k}w"
        if col not in df.columns:
            df[col] = pd.NA
    if "total_count_2to6m" not in df.columns:
        df["total_count_2to6m"] = pd.NA
    if "status" not in df.columns:
        df["status"] = "PENDING"
    if "error_message" not in df.columns:
        df["error_message"] = pd.NA

    target_idxs = [idx for idx in df.index if str(df.at[idx, "status"]).strip().upper() != "OK"]
    print(
        f"[INIT] source={_source} total_rows={len(df)} pending_rows={len(target_idxs)} "
        f"output={OUTPUT_PATH}"
    )

    stop_soon = False
    stop_after_one_more = False

    for idx in target_idxs:
        break_after_this = stop_after_one_more

        df, should_stop_soon_flag = process_one_ip(df, idx, bearer_token)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(OUTPUT_PATH, index=False)
        print(f"[CHECKPOINT] idx={idx} workbook_saved={OUTPUT_PATH}")

        if break_after_this:
            print("[STOP] stopping after one more item due to low remaining rate limit.")
            break

        if should_stop_soon_flag:
            stop_soon = True
        if stop_soon:
            stop_after_one_more = True

    print("[DONE] pipeline finished.")


if __name__ == "__main__":
    main()
