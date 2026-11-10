#!/usr/bin/env python3
"""Train a leakage-aware 15-feature HKJC PLACE model.

The feature order is intentionally aligned with the user's 15-feature Streamlit
app. Historical entity rates are rebuilt for every race date using earlier
calendar dates only. Races are split chronologically into train/validation/test.

Google Colab example:
  python train_hkjc_place_15feature_time_split.py \\
    --input /content/drive/MyDrive/HKJC_Model/hkjc_all_seasons_features_features_hkjc_enriched.csv \\
    --output-dir /content/drive/MyDrive/HKJC_Model/place_model_15_time_split

By default this matches the linked app's unchecked "dynamic form" option:
horse_last_rank=6.0 and horse surface/distance win rates=0.08. Add --dynamic-form
only if you will also tick that option in the app when predicting.

The input needs completed race rows with the required raw columns. Existing
precomputed feature columns are ignored and recalculated to prevent leakage.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

FEATURE_COLS_15 = [
    "market_implied_prob",
    "獨贏賠率",
    "odds_rank",
    "is_favorite",
    "排位檔位",
    "weight_diff",
    "weight_rank",
    "jockey_win_rate",
    "trainer_win_rate",
    "combo_win_rate",
    "horse_win_rate",
    "horse_last_rank",
    "距離",
    "horse_surface_win_rate",
    "horse_dist_win_rate",
]

REQUIRED_COLUMNS = [
    "賽事編號",
    "名次",
    "馬號",
    "馬名",
    "騎師",
    "練馬師",
    "實際負磅",
    "排位檔位",
    "獨贏賠率",
    "距離",
    "場地",
]

# These HKJC result codes indicate that the horse was withdrawn / did not take
# part, so it must not count as a starter or become a negative training label.
NON_STARTER_RE = re.compile(
    r"^(?:WR|WV|WX|TNP|WD|SCR|SCRATCHED|WITHDRAWN)|退出|退賽|未有出賽",
    flags=re.IGNORECASE,
)
DISQUALIFIED_RE = re.compile(r"DISQ|DISQUAL|取消資格", flags=re.IGNORECASE)
VOID_RACE_RE = re.compile(r"VOID|ABANDONED|取消賽事", flags=re.IGNORECASE)
RANK_RE = re.compile(r"^\s*(\d+)")

# Match the no-history fallbacks used by the current 15-feature predictor.
BASE_RATES = {
    "jockey_win_rate": 0.08,
    "trainer_win_rate": 0.08,
    "combo_win_rate": 0.05,
    "horse_win_rate": 0.05,
    "horse_surface_win_rate": 0.08,
    "horse_dist_win_rate": 0.08,
}

# HKJC local PLACE eligibility by the number of actual starters in that race.
# 4-6 starters: two place positions; 7+ starters: three place positions.
# Fewer than four starters do not create a PLACE training example.
PLACE_SLOTS_4_TO_6 = 2
PLACE_SLOTS_7_PLUS = 3

# Conservative LightGBM settings; no class_weight='balanced', because the model
# probability itself is used by the app and class weighting distorts calibration.
MODEL_PARAMS = {
    "objective": "binary",
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "max_depth": 6,
    "num_leaves": 15,
    "min_child_samples": 35,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": -1,
}


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read a CSV without silently skipping malformed rows."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp950", low_memory=False)


def normalize_key(series: pd.Series, strip_annotations: bool = True) -> pd.Series:
    """Normalize entity names in the same general way as the 15-feature app."""
    value = series.astype("string").fillna("").str.strip()
    if strip_annotations:
        # Jockey/horse names can carry an apprentice or code annotation.
        value = value.str.replace(r"\s*[（(].*$", "", regex=True)
    value = value.str.replace(r"\s+", "", regex=True).str.strip()
    return value.mask(value.eq(""), "__UNKNOWN__").astype(str)


def race_date_from_id(race_id: object) -> pd.Timestamp:
    match = re.search(r"(20\d{6})", str(race_id))
    if not match:
        return pd.NaT
    return pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")


def _add_prior_rate(
    frame: pd.DataFrame,
    entity_cols: list[str],
    feature_name: str,
    fallback: float | None,
) -> pd.DataFrame:
    """Attach a raw prior win rate using only dates earlier than each row."""
    group_cols = entity_cols + ["_race_date"]
    daily = (
        frame.groupby(group_cols, observed=True, dropna=False, as_index=False)
        .agg(_day_wins=("_is_win", "sum"), _day_starts=("_starter", "sum"))
        .sort_values(group_cols)
    )
    grouped = daily.groupby(entity_cols, observed=True, sort=False, dropna=False)
    daily["_wins_before"] = grouped["_day_wins"].cumsum() - daily["_day_wins"]
    daily["_starts_before"] = grouped["_day_starts"].cumsum() - daily["_day_starts"]
    default = np.nan if fallback is None else fallback
    daily[feature_name] = np.where(
        daily["_starts_before"] > 0,
        daily["_wins_before"] / daily["_starts_before"].replace(0, np.nan),
        default,
    )

    # Unknown or blank entities get the app's fixed fallback rather than a pooled
    # statistic for every blank name in the archive.
    lookup = daily[group_cols + [feature_name]]
    result = frame.drop(columns=[feature_name], errors="ignore").merge(
        lookup, on=group_cols, how="left", validate="many_to_one"
    )
    unknown = result[entity_cols].eq("__UNKNOWN__").any(axis=1)
    if fallback is not None:
        result.loc[unknown, feature_name] = fallback
        result[feature_name] = pd.to_numeric(result[feature_name], errors="coerce").fillna(fallback)
    else:
        result.loc[unknown, feature_name] = np.nan
        result[feature_name] = pd.to_numeric(result[feature_name], errors="coerce")
    return result


def _add_prior_last_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Use the latest numeric finishing position from a strictly earlier date."""
    numeric = frame.loc[frame["_numeric_rank"].notna()].copy()
    daily = (
        numeric.groupby(["_horse_key", "_race_date"], observed=True, as_index=False)
        .agg(_day_last_rank=("_numeric_rank", "min"))
        .sort_values(["_horse_key", "_race_date"])
    )
    daily["horse_last_rank"] = daily.groupby(
        "_horse_key", observed=True, sort=False
    )["_day_last_rank"].shift(1)
    lookup = daily[["_horse_key", "_race_date", "horse_last_rank"]]
    result = frame.drop(columns=["horse_last_rank"], errors="ignore").merge(
        lookup, on=["_horse_key", "_race_date"], how="left", validate="many_to_one"
    )
    result["horse_last_rank"] = pd.to_numeric(
        result["horse_last_rank"], errors="coerce"
    ).astype(float).fillna(6.5)
    result.loc[result["_horse_key"].eq("__UNKNOWN__"), "horse_last_rank"] = 6.5
    return result


def build_training_frame(
    raw: pd.DataFrame, dynamic_form: bool = False
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Rebuild target and all 15 features as-of each race date."""
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"輸入CSV缺少必要原始欄位：{missing}")

    df = raw.copy()
    # Some exported history files contain fully blank separator rows. These are
    # safe to remove; a row with any content but no parseable date still fails
    # validation below rather than being guessed or silently discarded.
    before_blank_drop = len(df)
    df = df.dropna(how="all").copy()
    blank_rows_removed = before_blank_drop - len(df)
    df["賽事編號"] = df["賽事編號"].astype("string").str.strip()
    df["馬號"] = df["馬號"].astype("string").str.strip()
    df["_race_date"] = df["賽事編號"].map(race_date_from_id)
    if df["_race_date"].isna().any():
        date_col = next(
            (c for c in ["賽事日期", "日期", "race_date", "date"] if c in df.columns),
            None,
        )
        if date_col:
            fallback = pd.to_datetime(df[date_col], errors="coerce")
            df["_race_date"] = df["_race_date"].fillna(fallback)
    if df["_race_date"].isna().any():
        bad = int(df["_race_date"].isna().sum())
        raise ValueError(f"有 {bad} 列不能解析賽事日期；請修正賽事編號或日期欄。")

    rank_text = df["名次"].astype("string").fillna("").str.strip()
    blank_rank = rank_text.eq("")
    raw_win_odds = pd.to_numeric(df["獨贏賠率"], errors="coerce")
    # The supplied history has three blank-rank rows with WIN odds 0.0. The
    # HKJC official result pages confirm these are WV withdrawals. Treat a
    # blank-rank row with no valid WIN price as a non-starter; if it has a real
    # price, stop instead of guessing its result or place in the field.
    ambiguous_blank = blank_rank & raw_win_odds.gt(1.0)
    if ambiguous_blank.any():
        raise ValueError(
            f"有 {int(ambiguous_blank.sum())} 列名次空白但獨贏賠率有效，無法判斷是否出賽；"
            "請補官方名次或出賽狀態後再訓練。"
        )
    rank_upper = rank_text.str.upper()
    df["_void_race_row"] = rank_upper.map(lambda x: bool(VOID_RACE_RE.search(str(x))))
    df["_blank_rank_no_valid_odds"] = blank_rank & ~raw_win_odds.gt(1.0)
    df["_nonstarter"] = rank_upper.map(lambda x: bool(NON_STARTER_RE.search(str(x)))) | df[
        "_blank_rank_no_valid_odds"
    ]
    df["_disqualified"] = rank_upper.map(lambda x: bool(DISQUALIFIED_RE.search(str(x))))
    rank_num = rank_text.str.extract(RANK_RE, expand=False)
    df["_numeric_rank"] = pd.to_numeric(rank_num, errors="coerce")
    df.loc[df["_disqualified"], "_numeric_rank"] = np.nan
    df["_starter"] = (~df["_nonstarter"]).astype(int)
    df["_is_win"] = df["_numeric_rank"].eq(1).fillna(False).astype(int)

    # Overlapping source exports must not duplicate a horse's race result.
    before_dedup = len(df)
    df = df.drop_duplicates(["賽事編號", "馬號"], keep="last").copy()
    duplicate_rows = before_dedup - len(df)
    void_races = set(df.loc[df["_void_race_row"], "賽事編號"].astype(str))
    df = df.loc[~df["賽事編號"].astype(str).isin(void_races)].copy()

    # Do not count official withdrawals/no-part entries as starters or train rows.
    starters = df.loc[df["_starter"].eq(1)].copy()
    if starters.empty:
        raise ValueError("排除退出及未出賽馬後沒有有效出賽資料。")

    # HKJC local fields contain at most 14 starters. Fail fast if a source export
    # has shifted horse bodyweight into the draw column (e.g. values around 1,000).
    draw_numeric = pd.to_numeric(starters["排位檔位"], errors="coerce")
    invalid_draw = draw_numeric.notna() & ~draw_numeric.between(1, 14)
    if invalid_draw.any():
        sample = starters.loc[invalid_draw, ["賽事編號", "馬號", "馬名", "排位檔位"]].head(8)
        raise ValueError(
            f"有效出賽馬匹有 {int(invalid_draw.sum())} 列檔位不在1至14；"
            "請檢查CSV欄位是否錯位。樣本："
            + sample.to_dict("records").__repr__()
        )

    starters["_starters_in_race"] = starters.groupby("賽事編號", observed=True)["馬號"].transform("size")
    starters["_place_slots"] = np.select(
        [starters["_starters_in_race"].between(4, 6), starters["_starters_in_race"].ge(7)],
        [PLACE_SLOTS_4_TO_6, PLACE_SLOTS_7_PLUS],
        default=0,
    ).astype(int)
    starters["target_place"] = (
        starters["_numeric_rank"].ge(1).fillna(False)
        & starters["_numeric_rank"].le(starters["_place_slots"]).fillna(False)
    ).astype(int)

    odds = pd.to_numeric(starters["獨贏賠率"], errors="coerce")
    starters["獨贏賠率"] = odds
    starters["_odds_valid"] = odds.gt(1.0)
    complete_odds_by_race = starters.groupby("賽事編號", observed=True)["_odds_valid"].transform("all")
    starters["_complete_win_odds_race"] = complete_odds_by_race

    # Historical rates use every completed actual start, even if a race is later
    # omitted from model evaluation because its WIN odds are incomplete.
    starters["_jockey_key"] = normalize_key(starters["騎師"])
    # Match app-manus.py: strip apprentice annotations from jockeys and horse
    # names, while trainer names are whitespace-normalized but otherwise kept.
    starters["_trainer_key"] = normalize_key(starters["練馬師"], strip_annotations=False)
    starters["_horse_key"] = normalize_key(starters["馬名"])
    starters["_surface_key"] = normalize_key(starters["場地"])
    starters["_distance_num"] = pd.to_numeric(starters["距離"], errors="coerce")
    starters["_distance_key"] = starters["_distance_num"].map(
        lambda x: str(int(x)) if pd.notna(x) else "__UNKNOWN__"
    )
    starters["_weight"] = pd.to_numeric(starters["實際負磅"], errors="coerce")
    starters["排位檔位"] = pd.to_numeric(starters["排位檔位"], errors="coerce")
    starters["距離"] = starters["_distance_num"]

    starters = _add_prior_rate(starters, ["_jockey_key"], "jockey_win_rate", BASE_RATES["jockey_win_rate"])
    starters = _add_prior_rate(starters, ["_trainer_key"], "trainer_win_rate", BASE_RATES["trainer_win_rate"])
    starters = _add_prior_rate(
        starters, ["_jockey_key", "_trainer_key"], "combo_win_rate", BASE_RATES["combo_win_rate"]
    )
    starters = _add_prior_rate(starters, ["_horse_key"], "horse_win_rate", BASE_RATES["horse_win_rate"])
    starters = _add_prior_rate(
        starters, ["_horse_key", "_surface_key"], "horse_surface_win_rate", None
    )
    starters = _add_prior_rate(
        starters, ["_horse_key", "_distance_key"], "horse_dist_win_rate", None
    )
    starters = _add_prior_last_rank(starters)
    # Dynamic mode matches the app's checked feature option. Its default is
    # fixed-form mode, so preserve the app's fixed inputs unless the user opts in.
    starters["horse_surface_win_rate"] = starters["horse_surface_win_rate"].fillna(
        starters["horse_win_rate"]
    ).fillna(BASE_RATES["horse_win_rate"])
    starters["horse_dist_win_rate"] = starters["horse_dist_win_rate"].fillna(
        starters["horse_win_rate"]
    ).fillna(BASE_RATES["horse_win_rate"])
    if dynamic_form:
        starters["horse_last_rank"] = starters["horse_last_rank"].fillna(6.5)
        starters["horse_surface_win_rate"] = starters["horse_surface_win_rate"].fillna(0.06)
        starters["horse_dist_win_rate"] = starters["horse_dist_win_rate"].fillna(0.06)
    else:
        starters["horse_last_rank"] = 6.0
        starters["horse_surface_win_rate"] = 0.08
        starters["horse_dist_win_rate"] = 0.08

    # Per-race market and weight features. A race is retained only when every
    # actual starter has a genuine WIN price; never fill a fake price such as 10.
    odds_raw = 1.0 / starters["獨贏賠率"].where(starters["_odds_valid"])
    prob_sum = odds_raw.groupby(starters["賽事編號"], observed=True).transform("sum")
    starters["market_implied_prob"] = (odds_raw / prob_sum).replace([np.inf, -np.inf], np.nan)
    starters["odds_rank"] = starters.groupby("賽事編號", observed=True)["獨贏賠率"].rank(
        method="min", ascending=True
    )
    starters["is_favorite"] = (starters["odds_rank"] == 1).astype(int)

    starters["排位檔位"] = starters["排位檔位"].fillna(7.0)
    # The live app fills missing horse weights with 120 lb before calculating
    # per-race differences and ranks; preserve this feature definition.
    starters["_weight"] = starters["_weight"].fillna(120.0)
    race_mean_weight = starters.groupby("賽事編號", observed=True)["_weight"].transform("mean")
    starters["weight_diff"] = starters["_weight"] - race_mean_weight
    starters["weight_rank"] = starters.groupby("賽事編號", observed=True)["_weight"].rank(
        ascending=False, method="min"
    )
    starters["距離"] = starters["距離"].fillna(1200.0)

    # Discard races with fewer than four starters or any missing/invalid WIN
    # price. This keeps the PLACE label and within-race market normalization valid.
    eligible = starters.loc[
        starters["_place_slots"].gt(0) & starters["_complete_win_odds_race"]
    ].copy()
    excluded_small = int(starters.loc[starters["_place_slots"].eq(0), "賽事編號"].nunique())
    excluded_odds = int(
        starters.loc[~starters["_complete_win_odds_race"], "賽事編號"].nunique()
    )
    if eligible.empty:
        raise ValueError("沒有符合PLACE規則且全場WIN賠率完整的賽事可訓練。")

    # Required predictor values must be finite; only numeric pre-race fields get
    # conservative fallbacks. Outcome labels and WIN odds are never fabricated.
    for col in FEATURE_COLS_15:
        eligible[col] = pd.to_numeric(eligible[col], errors="coerce")
    eligible[FEATURE_COLS_15] = eligible[FEATURE_COLS_15].replace([np.inf, -np.inf], np.nan)
    eligible["market_implied_prob"] = eligible["market_implied_prob"].fillna(0.0)
    eligible["排位檔位"] = eligible["排位檔位"].fillna(7.0)
    eligible["距離"] = eligible["距離"].fillna(1200.0)
    for col, fallback in BASE_RATES.items():
        eligible[col] = eligible[col].fillna(fallback)
    eligible["horse_last_rank"] = eligible["horse_last_rank"].fillna(6.5 if dynamic_form else 6.0)
    eligible["weight_diff"] = eligible["weight_diff"].fillna(0.0)
    eligible["weight_rank"] = eligible["weight_rank"].fillna(6.0)
    if not np.isfinite(eligible[FEATURE_COLS_15].to_numpy(dtype=float)).all():
        raise ValueError("特徵仍有非有限值；請檢查原始CSV。")

    stats = {
        "input_rows": int(len(raw)),
        "fully_blank_rows_removed": int(blank_rows_removed),
        "duplicate_rows_removed": int(duplicate_rows),
        "nonstarters_excluded": int(df["_nonstarter"].sum()),
        "blank_rank_no_valid_win_odds_excluded": int(df["_blank_rank_no_valid_odds"].sum()),
        "void_races_excluded": int(len(void_races)),
        "starter_rows_before_odds_filter": int(len(starters)),
        "eligible_rows": int(len(eligible)),
        "eligible_races": int(eligible["賽事編號"].nunique()),
        "races_excluded_under_4_starters": excluded_small,
        "races_excluded_incomplete_or_invalid_win_odds": excluded_odds,
        "place_positive_rows": int(eligible["target_place"].sum()),
    }
    return eligible.sort_values(["_race_date", "賽事編號", "馬號"]).reset_index(drop=True), stats


def chronological_masks(frame: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.Timestamp, pd.Timestamp]:
    """Split complete race dates into 70% train, 15% validation, 15% test."""
    dates = np.array(sorted(frame["_race_date"].dropna().unique()))
    if len(dates) < 10:
        raise ValueError(f"只有{len(dates)}個賽日；至少需要10個賽日作穩定時間切分。")
    train_end = max(1, int(len(dates) * 0.70))
    validation_end = max(train_end + 1, int(len(dates) * 0.85))
    validation_end = min(validation_end, len(dates) - 1)
    train_cut = pd.Timestamp(dates[train_end])
    validation_cut = pd.Timestamp(dates[validation_end])
    masks = {
        "train": frame["_race_date"] < train_cut,
        "validation": (frame["_race_date"] >= train_cut) & (frame["_race_date"] < validation_cut),
        "test": frame["_race_date"] >= validation_cut,
    }
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("時間切分後有空資料集；請增加歷史賽日或檢查資料日期。")
    return masks, train_cut, validation_cut


def safe_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | None]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None
    return {
        "auc": auc,
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def _race_selection_metrics(
    frame: pd.DataFrame,
    score_col: str,
    ascending: bool,
    payout_col: str | None,
) -> dict[str, Any]:
    """Top-1 and top-3 selection hit metrics plus optional official-dividend ROI."""
    # Build independent top-1 and top-3 selections without relying on original indices.
    top1_parts = []
    top3_parts = []
    for _, group in frame.groupby("賽事編號", sort=False, observed=True):
        ordered = group.sort_values([score_col, "馬號"], ascending=[ascending, True], kind="mergesort")
        top1_parts.append(ordered.head(1))
        top3_parts.append(ordered.head(min(3, len(ordered))))
    top1 = pd.concat(top1_parts, ignore_index=True)
    top3 = pd.concat(top3_parts, ignore_index=True)

    def hit_stats(selected: pd.DataFrame, is_top3: bool) -> dict[str, float]:
        bet_hit_rate = float(selected["target_place"].mean()) if len(selected) else float("nan")
        race_coverage = float(
            selected.groupby("賽事編號", observed=True)["target_place"].max().mean()
        ) if len(selected) else float("nan")
        return {
            "selected_horses": int(len(selected)),
            "bet_hit_rate": bet_hit_rate,
            "race_coverage": race_coverage,
        }

    def flat_roi(selected: pd.DataFrame) -> tuple[float | None, str, int, int]:
        if not payout_col or payout_col not in selected.columns:
            return None, "沒有官方位置派彩欄", 0, 0
        # Some older seasons in the source archive contain finishing results
        # but no dividend column values. Compute ROI only over races for which
        # every row has an official value and each placed horse has a positive
        # dividend; keep hit rates on the full test window above.
        values = pd.to_numeric(frame[payout_col], errors="coerce")
        payout_frame = frame[["賽事編號", "target_place"]].copy()
        payout_frame["_payout_value"] = values
        complete_races: set[str] = set()
        for race_id, race_group in payout_frame.groupby("賽事編號", sort=False, observed=True):
            race_values = race_group["_payout_value"]
            placed_values = race_group.loc[race_group["target_place"].eq(1), "_payout_value"]
            if (
                race_values.notna().all()
                and placed_values.notna().all()
                and placed_values.gt(0).all()
            ):
                complete_races.add(str(race_id))
        roi_bets = selected.loc[selected["賽事編號"].astype(str).isin(complete_races)].copy()
        if roi_bets.empty:
            return None, "測試期沒有具備完整官方位置派彩的場次", 0, 0
        winning = roi_bets.loc[roi_bets["target_place"].eq(1)]
        dividends = pd.to_numeric(winning[payout_col], errors="coerce")
        if dividends.isna().any() or dividends.le(0).any():
            return None, "有效派彩賽事中仍有入位注缺少派彩", 0, 0
        stake = 10.0 * len(roi_bets)
        payout = float(dividends.sum())
        return (
            float((payout / stake - 1.0) * 100.0),
            "每注HK$10；按官方每HK$10位置派彩；僅涵蓋派彩完整賽事",
            int(len(roi_bets)),
            int(roi_bets["賽事編號"].nunique()),
        )

    top1_stats = hit_stats(top1, False)
    top3_stats = hit_stats(top3, True)
    top1_roi, top1_roi_note, top1_roi_bets, top1_roi_races = flat_roi(top1)
    top3_roi, top3_roi_note, top3_roi_bets, top3_roi_races = flat_roi(top3)
    return {
        "top1_place_hit_rate": top1_stats["bet_hit_rate"],
        "top1_race_coverage": top1_stats["race_coverage"],
        "top1_flat_roi_pct": top1_roi,
        "top1_roi_note": top1_roi_note,
        "top1_roi_bets": top1_roi_bets,
        "top1_roi_races": top1_roi_races,
        "top3_ticket_hit_rate": top3_stats["bet_hit_rate"],
        "top3_at_least_one_place_per_race": top3_stats["race_coverage"],
        "top3_flat_roi_pct": top3_roi,
        "top3_roi_note": top3_roi_note,
        "top3_roi_bets": top3_roi_bets,
        "top3_roi_races": top3_roi_races,
        "top1_selected_horses": top1_stats["selected_horses"],
        "top3_selected_horses": top3_stats["selected_horses"],
    }


def evaluate_rankings(test: pd.DataFrame, model_probability: np.ndarray) -> dict[str, Any]:
    scored = test.copy()
    scored["model_place_probability"] = np.asarray(model_probability, dtype=float)
    scored["market_rank"] = scored.groupby("賽事編號", observed=True)["獨贏賠率"].rank(
        ascending=True, method="min"
    )
    scored["model_place_rank"] = scored.groupby("賽事編號", observed=True)[
        "model_place_probability"
    ].rank(ascending=False, method="first")
    payout_col = next(
        (c for c in ["位置派彩(每HK$10)", "位置派彩每HK$10", "位置派彩"] if c in scored.columns),
        None,
    )
    return {
        "row_metrics": safe_metrics(scored["target_place"].to_numpy(), model_probability),
        "model_ranking": _race_selection_metrics(
            scored, "model_place_probability", ascending=False, payout_col=payout_col
        ),
        "lowest_win_odds_baseline": _race_selection_metrics(
            scored, "獨贏賠率", ascending=True, payout_col=payout_col
        ),
        "payout_column": payout_col,
        "scored_test_rows": scored,
    }


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _mount_colab_drive_if_available() -> None:
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
    except ImportError:
        return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="以15項特徵訓練HKJC PLACE LightGBM，使用past-only特徵與時間切分。"
    )
    parser.add_argument("--input", default=None, help="已完成賽事歷史CSV路徑")
    parser.add_argument("--output-dir", default=None, help="模型、指標及測試預測輸出目錄")
    parser.add_argument("--no-mount-drive", action="store_true", help="在Colab中不自動掛載Google Drive")
    parser.add_argument(
        "--dynamic-form", action="store_true",
        help="訓練動態上仗／同場地／同路程特徵；預測時須勾選GitHub app同名選項",
    )
    parser.add_argument(
        "--both-forms", action="store_true",
        help="以同一份history分別訓練固定與動態form；輸出到fixed_form和dynamic_form子資料夾",
    )
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        # Colab/IPython may append a kernel connection flag such as
        # ``-f /path/to/kernel-....json`` when a script is launched via %run.
        allowed_kernel_args = all(
            token == "-f" or token.startswith(("/", "--HistoryManager."))
            for token in unknown_args
        )
        if allowed_kernel_args:
            print(f"忽略Colab/Jupyter kernel參數：{unknown_args}")
        else:
            parser.error(f"不認識的參數：{unknown_args}")

    if not args.no_mount_drive:
        _mount_colab_drive_if_available()

    if args.input:
        input_path = Path(args.input).expanduser()
    else:
        candidates = [
            Path("/content/drive/MyDrive/HKJC_Model/hkjc_enriched_updated_through_20261014.csv"),
            Path("/content/drive/MyDrive/HKJC_Model/hkjc_all_seasons_features_features_hkjc_enriched.csv"),
            Path("/content/drive/MyDrive/HKJC_Model/hkjc_all_seasons_features_features.csv"),
            Path("/content/drive/MyDrive/HKJC_Model/hkjc_all_seasons_features.csv"),
        ]
        input_path = next((p for p in candidates if p.exists()), candidates[0])
    if not input_path.exists():
        raise FileNotFoundError(
            f"找不到輸入歷史CSV：{input_path}\n"
            "請用 --input 指定你實際的Google Drive CSV路徑。"
        )

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else input_path.parent / "place_model_15_time_split"
    if args.both_forms and args.dynamic_form:
        parser.error("--both-forms 和 --dynamic-form 請勿同時使用。")
    if args.both_forms:
        output_dir.mkdir(parents=True, exist_ok=True)
        script_path = Path(__file__).resolve()
        for mode, extra_args in (
            ("fixed_form", []),
            ("dynamic_form", ["--dynamic-form"]),
        ):
            mode_dir = output_dir / mode
            command = [
                sys.executable,
                str(script_path),
                "--input",
                str(input_path),
                "--output-dir",
                str(mode_dir),
                "--no-mount-drive",
                *extra_args,
            ]
            print(f"\n{'=' * 18} 開始訓練 {mode} {'=' * 18}", flush=True)
            subprocess.run(command, check=True)
        print(f"\n兩種模式均完成；模型分別保存在：{output_dir / 'fixed_form'} 和 {output_dir / 'dynamic_form'}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"讀取：{input_path}")
    raw = read_csv_robust(input_path)
    print(f"原始資料：{len(raw):,}列、{raw['賽事編號'].nunique():,}場")
    data, cleaning_stats = build_training_frame(raw, dynamic_form=args.dynamic_form)
    masks, train_cut, validation_cut = chronological_masks(data)
    print("清洗摘要：", json.dumps(cleaning_stats, ensure_ascii=False))
    print(
        "時間切分：",
        f"Train < {train_cut.date()} ({int(masks['train'].sum()):,}列),",
        f"Validation {train_cut.date()}–{(validation_cut - pd.Timedelta(days=1)).date()} ({int(masks['validation'].sum()):,}列),",
        f"Test >= {validation_cut.date()} ({int(masks['test'].sum()):,}列)",
    )

    train = data.loc[masks["train"]].copy()
    validation = data.loc[masks["validation"]].copy()
    test = data.loc[masks["test"]].copy()
    X_train, y_train = train[FEATURE_COLS_15], train["target_place"].astype(int)
    X_val, y_val = validation[FEATURE_COLS_15], validation["target_place"].astype(int)
    X_test, y_test = test[FEATURE_COLS_15], test["target_place"].astype(int)
    if y_train.nunique() < 2 or y_val.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Train、Validation及Test都必須同時包含入位和未入位標籤。")

    eval_model = lgb.LGBMClassifier(**MODEL_PARAMS)
    eval_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, first_metric_only=True, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )
    best_iteration = int(eval_model.best_iteration_ or MODEL_PARAMS["n_estimators"])
    # Importance is taken from the chronological pre-holdout evaluation model,
    # not from the final refit that has seen the holdout rows.
    importance = pd.DataFrame({
        "feature": FEATURE_COLS_15,
        "gain_importance": eval_model.booster_.feature_importance(
            importance_type="gain", iteration=best_iteration
        ),
        "split_importance": eval_model.booster_.feature_importance(
            importance_type="split", iteration=best_iteration
        ),
    })
    gain_total = float(importance["gain_importance"].sum())
    importance["gain_share_pct"] = (
        importance["gain_importance"] / gain_total * 100.0 if gain_total > 0 else 0.0
    )
    importance = importance.sort_values(
        ["gain_importance", "split_importance"], ascending=False
    ).reset_index(drop=True)
    importance_path = output_dir / "place_feature_importance_15feature.csv"
    importance.to_csv(importance_path, index=False, encoding="utf-8-sig")
    test_probability = eval_model.predict_proba(X_test, num_iteration=best_iteration)[:, 1]
    evaluation = evaluate_rankings(test, test_probability)

    print("\n獨立時間留出 Test 評估（未參與訓練或早停）：")
    print(json.dumps(_clean_json({
        "test_dates": [test["_race_date"].min(), test["_race_date"].max()],
        "row_metrics": evaluation["row_metrics"],
        "model_ranking": evaluation["model_ranking"],
        "lowest_win_odds_baseline": evaluation["lowest_win_odds_baseline"],
        "best_iteration": best_iteration,
        "payout_column": evaluation["payout_column"],
        "feature_importance_top_5_by_gain": importance.head(5).to_dict("records"),
    }), ensure_ascii=False, indent=2))

    # Export row-level holdout predictions for independent inspection.
    scored = evaluation["scored_test_rows"].copy()
    export_cols = [
        "賽事編號", "_race_date", "馬號", "馬名", "獨贏賠率", "名次",
        "_starters_in_race", "_place_slots", "target_place", "model_place_probability",
        "model_place_rank", "market_rank",
    ] + FEATURE_COLS_15
    for payout_col in ["位置派彩(每HK$10)", "位置派彩每HK$10", "位置派彩"]:
        if payout_col in scored.columns:
            export_cols.append(payout_col)
            break
    export_cols = list(dict.fromkeys(c for c in export_cols if c in scored.columns))
    scored[export_cols].to_csv(
        output_dir / "place_test_predictions_15feature.csv", index=False, encoding="utf-8-sig"
    )

    # After the holdout evaluation is frozen, train the deployable candidate on
    # all available historical data using the validation-selected tree count.
    final_params = dict(MODEL_PARAMS)
    final_params["n_estimators"] = best_iteration
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(data[FEATURE_COLS_15], data["target_place"].astype(int))
    # Exact filename expected by the linked app.py: PLACE_MODEL_PATH.
    model_path = output_dir / "my_hkjc_place_model.pkl"
    joblib.dump(final_model, model_path)

    metadata = {
        "model_type": "LightGBM binary PLACE classifier",
        "target": "target_place: official numeric finishing rank within HKJC local PLACE slots",
        "features": FEATURE_COLS_15,
        "feature_count": len(FEATURE_COLS_15),
        "feature_history_rule": "all entity win rates and last numeric rank use race dates strictly earlier than each target date; same-day results excluded",
        "dynamic_form": bool(args.dynamic_form),
        "app_setting_match": (
            "tick the app's dynamic-form checkbox" if args.dynamic_form
            else "leave the app's dynamic-form checkbox unticked; fixed feature values match its default"
        ),
        "place_slots": {"4_to_6_actual_starters": 2, "7_or_more_actual_starters": 3, "fewer_than_4": "excluded"},
        "excluded_nonstarter_prefixes": ["WV", "WX", "TNP", "WD", "SCR", "SCRATCHED", "WITHDRAWN"],
        "class_weight": None,
        "split": {
            "train_end_exclusive": train_cut,
            "validation_end_exclusive": validation_cut,
            "test_start_inclusive": validation_cut,
            "test_dates": [test["_race_date"].min(), test["_race_date"].max()],
            "method": "chronological by full race date; 70% train, 15% validation, 15% holdout test",
        },
        "metrics_on_pre_final_refit_test_model": {
            "row_metrics": evaluation["row_metrics"],
            "model_ranking": evaluation["model_ranking"],
            "lowest_win_odds_baseline": evaluation["lowest_win_odds_baseline"],
        },
        "training_rows_final_model": int(len(data)),
        "training_dates_final_model": [data["_race_date"].min(), data["_race_date"].max()],
        "test_rows": int(len(test)),
        "test_races": int(test["賽事編號"].nunique()),
        "best_iteration_selected_on_validation": best_iteration,
        "params": final_params,
        "input_csv": str(input_path.resolve()),
        "note": "Test metrics belong to the model fit on train and evaluated after validation-based early stopping. The saved deployment candidate is subsequently refit on all rows; its test metrics are an estimate of the training procedure, not a fresh test of that refitted file.",
    }
    metrics_path = output_dir / "place_training_metrics_15feature.json"
    metrics_path.write_text(
        json.dumps(_clean_json(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n候選模型：{model_path}")
    print(f"測試逐馬預測：{output_dir / 'place_test_predictions_15feature.csv'}")
    print(f"指標與切分資訊：{metrics_path}")
    print(f"特徵重要性：{importance_path}")
    print("注意：此候選模型輸入為15大；正式預測app必須選擇與模型相同的form模式。")


if __name__ == "__main__":
    main()
