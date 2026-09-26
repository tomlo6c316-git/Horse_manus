"""Inspect HKJC CSV columns and add leakage-safe horse history features.

Features added per runner:
- days_since_last_race
- horse_course_starts (same venue + exact distance, prior dates only)
- horse_course_win_rate (smoothed)
- horse_course_top3_rate (smoothed)

Rows with blank finishing results remain in the output for prediction but are
never used as historical outcomes. All results on the target calendar date are
excluded from as-of features to prevent same-day leakage.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_WIN_RATE = 0.10
BASE_TOP3_RATE = 0.30
PRIOR_STRENGTH = 10.0
UNKNOWN_DAYS_SINCE_RUN = 999.0

COLUMN_CANDIDATES = {
    "race_id": ["賽事編號", "賽事ID", "race_id", "raceId"],
    "race_date": ["賽事日期", "日期", "race_date", "date"],
    "horse_id": ["馬匹編號", "馬匹代號", "horse_id", "horse_code", "馬名", "horse_name"],
    "horse_no": ["馬號", "horse_no", "saddle_no"],
    "venue": ["racecourse_code", "賽馬場", "競馬場", "venue_code", "racecourse", "場地代碼", "venue", "course", "場地"],
    "distance": ["official_distance_m", "官方路程", "距離", "路程", "distance"],
    "rank": ["名次", "實際名次", "finish_position", "position", "rank"],
}


def detect_columns(df: pd.DataFrame, *, verbose: bool = True) -> dict[str, str | None]:
    """Print source columns and map common Chinese/English column names."""
    if verbose:
        print("CSV 欄位：")
        for col in df.columns:
            print(f"  - {col}")

    mapping = {
        field: next((candidate for candidate in candidates if candidate in df.columns), None)
        for field, candidates in COLUMN_CANDIDATES.items()
    }
    if verbose:
        print("\n欄位對應：")
        for field, col in mapping.items():
            print(f"  {field}: {col if col else '未找到'}")
    return mapping


def _race_date_from_id(value: object) -> pd.Timestamp:
    match = re.search(r"(20\d{6})", str(value))
    if not match:
        return pd.NaT
    return pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")


def _entity_key(frame: pd.DataFrame, columns: list[str]) -> list[tuple]:
    normalized = []
    for col in columns:
        values = frame[col].astype("string").str.strip()
        values = values.mask(values.isna() | values.eq(""), "__UNKNOWN__").astype(str)
        normalized.append(values.tolist())
    return list(zip(*normalized))


def _asof_lookup(
    target: pd.DataFrame,
    daily_history: pd.DataFrame,
    key_col: str,
    value_cols: list[str],
    defaults: dict[str, float],
) -> pd.DataFrame:
    """Lookup cumulative historical values strictly before each target date."""
    result = pd.DataFrame(index=target.index)
    for col, default in defaults.items():
        result[col] = float(default)
    if target.empty or daily_history.empty:
        return result

    grouped_history = {
        key: group.sort_values("_race_date")
        for key, group in daily_history.groupby(key_col, sort=False)
    }
    for key, positions in target.groupby(key_col, sort=False).groups.items():
        history = grouped_history.get(key)
        if history is None or history.empty:
            continue

        history_dates = history["_race_date"].to_numpy(dtype="datetime64[ns]")
        target_dates = target.loc[positions, "_race_date"].to_numpy(dtype="datetime64[ns]")
        prior_index = np.searchsorted(history_dates, target_dates, side="left") - 1
        valid = prior_index >= 0
        if not valid.any():
            continue

        target_positions = np.asarray(positions)[valid]
        history_positions = prior_index[valid]
        for col in value_cols:
            result.loc[target_positions, col] = history[col].to_numpy()[history_positions]
    return result


def add_horse_rest_and_course_features(
    source: pd.DataFrame,
    *,
    race_id_col: str | None = None,
    race_date_col: str | None = None,
    horse_col: str | None = None,
    venue_col: str | None = None,
    distance_col: str | None = None,
    rank_col: str | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Inspect columns and append past-only rest/course performance features.

    You may pass explicit column names if your CSV uses uncommon headers. The
    function returns (feature_dataframe, detected_column_mapping).
    """
    df = source.copy().reset_index(drop=True)
    detected = detect_columns(df, verbose=verbose)
    explicit = {
        "race_id": race_id_col,
        "race_date": race_date_col,
        "horse_id": horse_col,
        "venue": venue_col,
        "distance": distance_col,
        "rank": rank_col,
    }
    mapping = {key: explicit.get(key) or detected[key] for key in detected}

    required = ["horse_id", "venue", "distance", "rank"]
    missing = [field for field in required if not mapping.get(field)]
    if not mapping.get("race_id") and not mapping.get("race_date"):
        missing.append("race_id or race_date")
    if missing:
        raise ValueError(
            "缺少建立歷史特徵所需欄位："
            + ", ".join(missing)
            + "。請確認 CSV 欄位名稱，或在函式參數傳入對應欄名。"
        )

    if mapping["race_id"]:
        df["_race_date"] = df[mapping["race_id"]].map(_race_date_from_id)
    else:
        df["_race_date"] = pd.NaT
    if mapping["race_date"]:
        fallback = pd.to_datetime(df[mapping["race_date"]], errors="coerce")
        df["_race_date"] = df["_race_date"].fillna(fallback)
    if df["_race_date"].isna().any():
        bad_rows = int(df["_race_date"].isna().sum())
        raise ValueError(f"有 {bad_rows} 列無法解析賽事日期；需 YYYY-MM-DD 或賽事編號含 YYYYMMDD。")
    # Normalize timezone away and compare by calendar day.
    df["_race_date"] = pd.to_datetime(df["_race_date"], errors="coerce").dt.normalize()

    df["_horse_key"] = (
        df[mapping["horse_id"]].astype("string").str.strip()
        .mask(lambda s: s.isna() | s.eq(""), "__UNKNOWN__").astype(str)
    )
    df["_venue_key"] = (
        df[mapping["venue"]].astype("string").str.strip()
        .mask(lambda s: s.isna() | s.eq(""), "__UNKNOWN__").astype(str)
    )
    df["_distance_key"] = pd.to_numeric(df[mapping["distance"]], errors="coerce")
    if df["_distance_key"].isna().any():
        bad_distances = int(df["_distance_key"].isna().sum())
        if verbose:
            print(f"注意：{bad_distances} 列路程不是有效數字，這些列的同場地路程歷史率使用預設值。")

    # Blank rank indicates a pending/unrun entry. Any recorded non-numeric
    # outcome (e.g. DNF) is a start and a non-winner/non-top-three result.
    rank_text = df[mapping["rank"]].astype("string").str.strip()
    df["_has_result"] = rank_text.notna() & rank_text.ne("")
    df["_numeric_rank"] = pd.to_numeric(df[mapping["rank"]], errors="coerce").fillna(99.0)
    df["_win"] = (df["_numeric_rank"] == 1).astype(int)
    df["_top3"] = (df["_numeric_rank"] <= 3).astype(int)

    history = df.loc[
        df["_has_result"]
        & df["_distance_key"].notna()
        & df["_horse_key"].ne("__UNKNOWN__")
        & df["_race_date"].notna()
    ].copy()
    # Remove duplicated season exports before counting historical starts.
    dedupe_cols = [mapping["race_id"], mapping["horse_id"]] if mapping["race_id"] else ["_race_date", mapping["horse_id"]]
    dedupe_cols = [col for col in dedupe_cols if col]
    if dedupe_cols:
        history = history.drop_duplicates(subset=dedupe_cols, keep="last")

    # 1) Rest days: latest prior calendar date, excluding every race on current date.
    rest_daily = (
        history.groupby(["_horse_key", "_race_date"], observed=True, as_index=False)
        .size()
        .sort_values(["_horse_key", "_race_date"])
    )
    rest_daily["_previous_race_date"] = rest_daily.groupby(
        "_horse_key", sort=False
    )["_race_date"].shift(1)
    rest_lookup = {
        key: group.sort_values("_race_date")
        for key, group in rest_daily.groupby("_horse_key", sort=False)
    }
    df["days_since_last_race"] = UNKNOWN_DAYS_SINCE_RUN
    for key, positions in df.groupby("_horse_key", sort=False).groups.items():
        prior = rest_lookup.get(key)
        if prior is None or prior.empty:
            continue
        dates = prior["_race_date"].to_numpy(dtype="datetime64[ns]")
        target_dates = df.loc[positions, "_race_date"].to_numpy(dtype="datetime64[ns]")
        previous_index = np.searchsorted(dates, target_dates, side="left") - 1
        valid = previous_index >= 0
        if valid.any():
            idx = np.asarray(positions)[valid]
            last_date = prior["_race_date"].to_numpy(dtype="datetime64[ns]")[previous_index[valid]]
            current_date = df.loc[idx, "_race_date"].to_numpy(dtype="datetime64[ns]")
            df.loc[idx, "days_since_last_race"] = (
                (current_date - last_date) / np.timedelta64(1, "D")
            ).astype(float)

    # 2) Same-venue, exact-distance prior starts/win/top-3 rates.
    df["_course_key"] = list(zip(df["_horse_key"], df["_venue_key"], df["_distance_key"]))
    history["_course_key"] = list(zip(history["_horse_key"], history["_venue_key"], history["_distance_key"]))
    daily = (
        history.groupby(["_course_key", "_race_date"], observed=True, as_index=False)
        .agg(
            starts=("_win", "size"),
            wins=("_win", "sum"),
            top3=("_top3", "sum"),
        )
        .sort_values(["_course_key", "_race_date"])
    )
    for total_col, daily_col in [("prior_starts", "starts"), ("prior_wins", "wins"), ("prior_top3", "top3")]:
        daily[total_col] = daily.groupby("_course_key", sort=False)[daily_col].cumsum() - daily[daily_col]

    # For a target date, the latest strictly earlier daily row is fully known,
    # so include that row's starts/results (while still excluding target day).
    daily["horse_course_starts"] = daily["prior_starts"] + daily["starts"]
    daily["horse_course_win_rate"] = (
        daily["prior_wins"] + daily["wins"] + BASE_WIN_RATE * PRIOR_STRENGTH
    ) / (daily["prior_starts"] + daily["starts"] + PRIOR_STRENGTH)
    daily["horse_course_top3_rate"] = (
        daily["prior_top3"] + daily["top3"] + BASE_TOP3_RATE * PRIOR_STRENGTH
    ) / (daily["prior_starts"] + daily["starts"] + PRIOR_STRENGTH)

    df["horse_course_starts"] = 0.0
    df["horse_course_win_rate"] = BASE_WIN_RATE
    df["horse_course_top3_rate"] = BASE_TOP3_RATE
    course_daily_groups = {
        key: group.sort_values("_race_date")
        for key, group in daily.groupby("_course_key", sort=False)
    }
    for key, positions in df.groupby("_course_key", sort=False).groups.items():
        prior = course_daily_groups.get(key)
        if prior is None or prior.empty:
            continue
        dates = prior["_race_date"].to_numpy(dtype="datetime64[ns]")
        target_dates = df.loc[positions, "_race_date"].to_numpy(dtype="datetime64[ns]")
        prior_index = np.searchsorted(dates, target_dates, side="left") - 1
        valid = prior_index >= 0
        if not valid.any():
            continue
        idx = np.asarray(positions)[valid]
        prior_pos = prior_index[valid]
        for col in ["horse_course_starts", "horse_course_win_rate", "horse_course_top3_rate"]:
            df.loc[idx, col] = prior[col].to_numpy(dtype=float)[prior_pos]

    # Remove internal helper columns, keeping only the original CSV + features.
    helper_cols = [
        "_race_date", "_horse_key", "_venue_key", "_distance_key", "_has_result",
        "_numeric_rank", "_win", "_top3", "_course_key",
    ]
    df = df.drop(columns=helper_cols, errors="ignore")
    return df, mapping


def read_csv_robust(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp950", on_bad_lines="skip")


def main() -> None:
    parser = argparse.ArgumentParser(description="檢查 HKJC CSV 並新增賽前歷史特徵")
    parser.add_argument("input_csv", nargs="?", help="輸入 CSV 路徑")
    parser.add_argument("-o", "--output", default=None, help="輸出 CSV 路徑；預設為原檔名_features.csv")
    # Colab/Jupyter may inject kernel arguments such as `-f kernel.json`.
    # Ignore unknown arguments so notebook execution gives a helpful message.
    cli_args = sys.argv[1:]
    filtered_args = []
    skip_next = False
    for item in cli_args:
        if skip_next:
            skip_next = False
            continue
        if item == "-f":
            skip_next = True
            continue
        filtered_args.append(item)
    args, unknown_args = parser.parse_known_args(filtered_args)
    if unknown_args:
        print(f"忽略 Notebook/kernel 額外參數：{unknown_args}")
    if not args.input_csv:
        print(
            "請用以下任一方式提供 CSV 路徑：\n"
            "  Notebook 儲存格：!python /content/hkjc_extra_features.py /content/your.csv\n"
            "  Python 呼叫：df = read_csv_robust(path); add_horse_rest_and_course_features(df)"
        )
        return

    input_path = Path(args.input_csv)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_features.csv")
    source = read_csv_robust(input_path)
    result, mapping = add_horse_rest_and_course_features(source)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("\n新增特徵：days_since_last_race、horse_course_starts、horse_course_win_rate、horse_course_top3_rate")
    print(f"輸出完成：{output_path}")


if __name__ == "__main__":
    main()
