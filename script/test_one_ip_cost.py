from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Allow importing script/X_count_pipeline.py when running from repo root.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import X_count_pipeline as xp


def load_df_for_test() -> tuple[pd.DataFrame, str]:
    try:
        return xp.load_dataframe(xp.INPUT_PATH, xp.OUTPUT_PATH)
    except FileNotFoundError:
        csv_path = xp.BASE_DIR / "raw" / "Unique_Celebrity_Keyword.X.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Neither input xlsx nor fallback csv exists. "
                f"Missing: {xp.INPUT_PATH} and {csv_path}"
            )
        df = pd.read_csv(csv_path)
        for col in ["status", "error_message"]:
            if col not in df.columns:
                df[col] = "PENDING" if col == "status" else pd.NA
        return df, "fresh_csv_fallback"


def pick_index(df, target_idx: int | None, target_name: str | None) -> int:
    if target_idx is not None:
        if target_idx not in df.index:
            raise ValueError(f"idx {target_idx} not found in dataframe index")
        return target_idx

    if target_name:
        matches = df.index[df["group.nd.name"].astype(str).str.strip() == target_name]
        if len(matches) == 0:
            raise ValueError(f"group.nd.name '{target_name}' not found")
        return int(matches[0])

    candidates = [i for i in df.index if str(df.at[i, "status"]).strip().upper() != "OK"]
    if candidates:
        return int(candidates[0])
    return int(df.index[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exactly one X counts/all request for one IP and print cost estimate."
    )
    parser.add_argument("--idx", type=int, default=None, help="DataFrame index to test")
    parser.add_argument("--name", type=str, default=None, help="group.nd.name to test")
    parser.add_argument(
        "--cost-per-request-usd",
        type=float,
        default=None,
        help="Optional unit price in USD for one counts/all request",
    )
    args = parser.parse_args()

    token = xp.get_bearer_token()
    df, source = load_df_for_test()
    df = xp.compute_collab_start_date(df)
    df = xp.fill_windows(df)

    idx = pick_index(df, args.idx, args.name)
    row = df.loc[idx]

    group_name = row.get("group.nd.name")
    query = xp.build_query(row.get("variants_joined"))
    start_iso = str(row.get("window_start"))
    end_iso = str(row.get("window_end"))

    t0 = time.time()
    payload, rate_meta = xp.fetch_counts_all_day(
        query=query,
        start_iso=start_iso,
        end_iso=end_iso,
        bearer_token=token,
    )
    elapsed = time.time() - t0

    weekly_counts = xp.aggregate_daily_to_iso_week(payload.get("data", []))
    lag_features, total_count_2to6m = xp.build_lag_features(
        str(row.get("collab_start_week")), weekly_counts
    )

    request_count = 1
    estimated_cost = None
    if args.cost_per_request_usd is not None:
        estimated_cost = request_count * args.cost_per_request_usd

    print("=== ONE-IP COST TEST ===")
    print(f"source={source}")
    print(f"idx={idx}")
    print(f"group.nd.name={group_name}")
    print(f"window_start={start_iso}")
    print(f"window_end={end_iso}")
    print(f"elapsed_sec={elapsed:.3f}")
    print(f"requests={request_count}")
    print(f"rate_limit={rate_meta}")
    print(f"daily_buckets={len(payload.get('data', []) or [])}")
    print(f"weekly_keys={len(weekly_counts)}")
    print(f"total_count_2to6m={total_count_2to6m}")
    print(f"sample_lag_9w={lag_features.get('count_lag_9w')}")
    if estimated_cost is None:
        print("estimated_cost_usd=None (pass --cost-per-request-usd to calculate)")
    else:
        print(f"estimated_cost_usd={estimated_cost:.6f}")


if __name__ == "__main__":
    main()
