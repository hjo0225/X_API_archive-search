from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "raw" / "Unique_Celebrity_Keyword.X.csv"
WORKBOOK_PATH = BASE_DIR / "data" / "Celebrity_count.xlsx"
JSON_DIR = BASE_DIR / "data" / "json"

LAG_COLUMNS = [f"count_lag_{k}w" for k in range(9, 27)]
RESTORE_COLUMNS = LAG_COLUMNS + [
    "total_count_2to6m",
    "collab_start_date",
    "window_start",
    "window_end",
    "status",
    "error_message",
]


def parse_json_filename(path: Path) -> tuple[str, int]:
    m = re.match(r"^(.*)\((\d+)회차\)\.json$", path.name)
    if not m:
        raise ValueError(f"Invalid json filename format: {path.name}")
    return m.group(1).strip(), int(m.group(2))


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


def load_or_create_workbook(workbook_path: Path) -> pd.DataFrame:
    if workbook_path.exists():
        return pd.read_excel(workbook_path)

    resolved_input = _resolve_input_path(INPUT_PATH)
    df = _read_table(resolved_input)
    df["status"] = "PENDING"
    df["error_message"] = pd.NA
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(workbook_path, index=False)
    print(f"[CREATE] workbook_created={workbook_path} source={resolved_input}")
    return df


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in RESTORE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def build_row_index(df: pd.DataFrame) -> dict[tuple[str, int], int]:
    index_map: dict[tuple[str, int], int] = {}
    for idx, row in df.iterrows():
        group_name = str(row.get("group.nd.name", "")).strip()
        collab_n_raw = row.get("collab_n")
        if not group_name or pd.isna(collab_n_raw):
            continue
        try:
            collab_n = int(collab_n_raw)
        except (TypeError, ValueError):
            continue
        index_map[(group_name, collab_n)] = idx
    return index_map


def restore_ok_payload(df: pd.DataFrame, idx: int, payload: dict) -> None:
    lag_features = payload.get("lag_features") or {}
    for col in LAG_COLUMNS:
        df.at[idx, col] = lag_features.get(col, pd.NA)

    df.at[idx, "total_count_2to6m"] = payload.get("total_count_2to6m", pd.NA)
    df.at[idx, "collab_start_date"] = payload.get(
        "collab_start_date",
        payload.get("collab_startdate", pd.NA),
    )
    df.at[idx, "window_start"] = payload.get("window_start", pd.NA)
    df.at[idx, "window_end"] = payload.get("window_end", pd.NA)
    df.at[idx, "status"] = "OK"
    df.at[idx, "error_message"] = ""


def restore_error_payload(df: pd.DataFrame, idx: int, payload: dict) -> None:
    df.at[idx, "status"] = payload.get("status", "ERROR")
    df.at[idx, "error_message"] = payload.get("error_message", "")


def values_equal(left, right) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return left == right


def verify_row(df: pd.DataFrame, idx: int, payload: dict) -> list[str]:
    issues: list[str] = []
    if payload.get("status") == "ERROR":
        expected_status = payload.get("status", "ERROR")
        expected_error = payload.get("error_message", "")
        if not values_equal(df.at[idx, "status"], expected_status):
            issues.append("status")
        if not values_equal(df.at[idx, "error_message"], expected_error):
            issues.append("error_message")
        return issues

    lag_features = payload.get("lag_features") or {}
    for col in LAG_COLUMNS:
        if not values_equal(df.at[idx, col], lag_features.get(col, pd.NA)):
            issues.append(col)
    if not values_equal(df.at[idx, "total_count_2to6m"], payload.get("total_count_2to6m", pd.NA)):
        issues.append("total_count_2to6m")
    expected_collab_start_date = payload.get(
        "collab_start_date",
        payload.get("collab_startdate", pd.NA),
    )
    if not values_equal(df.at[idx, "collab_start_date"], expected_collab_start_date):
        issues.append("collab_start_date")
    if not values_equal(df.at[idx, "window_start"], payload.get("window_start", pd.NA)):
        issues.append("window_start")
    if not values_equal(df.at[idx, "window_end"], payload.get("window_end", pd.NA)):
        issues.append("window_end")
    if not values_equal(df.at[idx, "status"], "OK"):
        issues.append("status")
    if not values_equal(df.at[idx, "error_message"], ""):
        issues.append("error_message")
    return issues


def main() -> None:
    if not JSON_DIR.exists():
        raise FileNotFoundError(f"JSON directory not found: {JSON_DIR}")

    df = load_or_create_workbook(WORKBOOK_PATH)
    df = ensure_columns(df)
    row_index = build_row_index(df)

    restored_ok = 0
    restored_error = 0
    skipped = 0
    unmatched = 0
    verified = 0
    mismatched = 0
    mismatch_samples: list[str] = []

    json_paths = sorted(JSON_DIR.glob("*.json"))
    for path in json_paths:
        try:
            group_name, collab_n = parse_json_filename(path)
        except ValueError:
            skipped += 1
            print(f"[SKIP] invalid filename: {path.name}")
            continue

        idx = row_index.get((group_name, collab_n))
        if idx is None:
            unmatched += 1
            print(f"[MISS] no matching row for {path.name}")
            continue

        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "ERROR":
            restore_error_payload(df, idx, payload)
            restored_error += 1
            print(f"[RESTORE-ERROR] idx={idx} file={path.name}")
        else:
            restore_ok_payload(df, idx, payload)
            restored_ok += 1
            print(f"[RESTORE-OK] idx={idx} file={path.name}")

        issues = verify_row(df, idx, payload)
        verified += 1
        if issues:
            mismatched += 1
            if len(mismatch_samples) < 10:
                mismatch_samples.append(f"{path.name}: {', '.join(issues)}")

    df.to_excel(WORKBOOK_PATH, index=False)
    print(
        f"[DONE] restored_ok={restored_ok} restored_error={restored_error} "
        f"unmatched={unmatched} skipped={skipped} saved={WORKBOOK_PATH}"
    )
    print(f"[VERIFY] verified={verified} mismatched={mismatched}")
    for sample in mismatch_samples:
        print(f"[VERIFY-MISMATCH] {sample}")


if __name__ == "__main__":
    main()

