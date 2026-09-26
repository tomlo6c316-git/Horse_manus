import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import requests
import re
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from hkjc_extra_features import add_horse_rest_and_course_features

# 頁面基本設定
st.set_page_config(
    page_title="HKJC AI 智能賽馬預測系統",
    page_icon="🐎",
    layout="wide"
)

st.title("🐎 HKJC 19 項特徵 AI 預測系統 (支援即時賠率)")
st.markdown("---")

# 設定模型路徑
MODEL_PATH = 'my_hkjc_model.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

if model is None:
    st.error(f"⚠️ 找不到 AI 模型檔 `{MODEL_PATH}`！請確認模型是否已上傳至正確目錄。")
    st.stop()

# 側邊欄：檔案上傳區與參數設定
st.sidebar.header("📂 資料載入")
uploaded_file = st.sidebar.file_uploader("請上傳賽事資料 (CSV)", type=['csv'])
history_file = st.sidebar.file_uploader(
    "上傳歷史 enriched CSV（用於計算賽前歷史特徵）",
    type=['csv'],
    key='historical_results_upload',
    help='請選擇 hkjc_all_seasons_features_features_hkjc_enriched.csv；不提供時，4項歷史特徵會使用預設值。',
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 投注策略參數設定")
min_ev = st.sidebar.slider("最小期望值 (EV 門檻)", 0.0, 1.5, 0.0, 0.05)
min_odds = st.sidebar.number_input("最低獨贏賠率", min_value=1.0, max_value=50.0, value=3.0)
max_odds = st.sidebar.number_input("最高獨贏賠率", min_value=1.0, max_value=100.0, value=20.0)

# ==========================================
# HKJC 新版 GraphQL：一次取得當日所有場次 WIN / PLACE 賠率
# 查詢字串需保持原樣；這是網站使用的唯讀查詢，不會下注。
# ==========================================
GRAPHQL_URL = "https://info.cld.hkjc.com/graphql/base/"
ODDS_QUERY = """query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
  raceMeetings(date: $date, venueCode: $venueCode) {
    pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
      id
      status
      sellStatus
      oddsType
      lastUpdateTime
      guarantee
      minTicketCost
      name_en
      name_ch
      leg {
        number
        races
      }
      cWinSelections {
        composite
        name_ch
        name_en
        starters
      }
      oddsNodes {
        combString
        oddsValue
        hotFavourite
        oddsDropValue
        bankerOdds {
          combString
          oddsValue
        }
      }
    }
  }
}"""

GRAPHQL_HEADERS = {
    "Accept": "*/*",
    "Content-Type": "application/json",
    "Origin": "https://bet.hkjc.com",
    "Referer": "https://bet.hkjc.com/",
    "User-Agent": "Mozilla/5.0",
}

def normalize_horse_no(value):
    """把 01、1、1.0 統一成字串 1，方便與 CSV 馬號比對。"""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except (ValueError, TypeError):
        return text


def extract_race_no(value):
    """由類似 20260927-01 的賽事編號提取場次。"""
    match = re.search(r"-(\d+)\s*$", str(value).strip())
    return int(match.group(1)) if match else None


def fetch_live_odds(date_str, venue):
    """一次抓取指定日期／場地所有場次的即時 WIN / PLACE 賠率。

    回傳 (賠率資料, 最近更新時間, 錯誤訊息)。賠率資料格式：
    {場次: {"WIN": {馬號: 賠率}, "PLA": {馬號: 賠率}}}
    """
    body = {
        "operationName": "racing",
        "variables": {
            "date": date_str,
            "venueCode": venue,
            "raceNo": None,  # null 表示查詢該賽日所有場次
            "oddsTypes": ["WIN", "PLA"],
        },
        "query": ODDS_QUERY,
    }
    try:
        resp = requests.post(
            GRAPHQL_URL,
            headers=GRAPHQL_HEADERS,
            json=body,
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("errors"):
            return None, None, f"GraphQL 錯誤：{payload['errors']}"

        meetings = (payload.get("data") or {}).get("raceMeetings") or []
        if not meetings:
            return None, None, "找不到該日期／場地的賽事資料。"

        odds_by_race = {}
        timestamps = []
        for pool in meetings[0].get("pmPools") or []:
            pool_type = str(pool.get("oddsType", "")).upper()
            if pool_type not in ("WIN", "PLA"):
                continue
            if pool.get("lastUpdateTime"):
                timestamps.append(pool["lastUpdateTime"])

            race_numbers = (pool.get("leg") or {}).get("races") or []
            for race_number in race_numbers:
                race_number = int(race_number)
                market = odds_by_race.setdefault(
                    race_number, {"WIN": {}, "PLA": {}}
                )
                for node in pool.get("oddsNodes") or []:
                    horse_no = normalize_horse_no(node.get("combString"))
                    odds_value = node.get("oddsValue")
                    if horse_no and odds_value not in (None, ""):
                        market[pool_type][horse_no] = float(odds_value)

        if not odds_by_race:
            return None, None, "賽事有回應，但目前尚無可用 WIN／PLACE 賠率。"
        return odds_by_race, (max(timestamps) if timestamps else None), None

    except requests.RequestException as exc:
        return None, None, f"連線錯誤：{exc}"
    except (ValueError, TypeError, KeyError) as exc:
        return None, None, f"解析回應錯誤：{exc}"


# Must match FEATURE_COLS and the historical-rate settings in the training script.
FEATURE_COLS = [
    'market_implied_prob', '獨贏賠率', 'odds_rank', 'is_favorite',
    '排位檔位', 'weight_diff', 'weight_rank',
    'jockey_win_rate', 'trainer_win_rate', 'combo_win_rate',
    'horse_win_rate', 'horse_last_rank', '距離',
    'horse_surface_win_rate', 'horse_dist_win_rate',
    'days_since_last_race', 'horse_course_starts',
    'horse_course_win_rate', 'horse_course_top3_rate',
]
BASE_WIN_RATE = 0.10
PRIOR_STRENGTH = 10.0


def make_entity_keys(frame, columns):
    """Build collision-safe tuple keys from the selected entity columns."""
    if not columns:
        return [('__UNKNOWN__',)] * len(frame)
    normalized = []
    for col in columns:
        values = frame[col].astype('string').str.strip()
        values = values.mask(values.isna() | values.eq(''), '__UNKNOWN__').astype(str)
        normalized.append(values.tolist())
    return list(zip(*normalized))


def assign_prior_rate(history, target, entity_cols, base_rate):
    """Smoothed win rates using outcomes strictly before each target calendar date."""
    result = pd.Series(base_rate, index=target.index, dtype=float)
    if history.empty or target.empty:
        return result

    h = history.loc[history['_race_date'].notna()].copy()
    t = target.loc[target['_race_date'].notna()].copy()
    if h.empty or t.empty:
        return result

    h['_entity_key'] = make_entity_keys(h, entity_cols)
    t['_entity_key'] = make_entity_keys(t, entity_cols)
    daily = (
        h.groupby(['_entity_key', '_race_date'], sort=True, observed=True)
        .agg(wins=('target_win', 'sum'), starts=('target_win', 'size'))
        .reset_index()
        .sort_values(['_entity_key', '_race_date'])
    )
    daily['cum_wins'] = daily.groupby('_entity_key', sort=False)['wins'].cumsum()
    daily['cum_starts'] = daily.groupby('_entity_key', sort=False)['starts'].cumsum()
    lookup = {key: group for key, group in daily.groupby('_entity_key', sort=False)}

    for key, positions in t.groupby('_entity_key', sort=False).groups.items():
        past = lookup.get(key)
        if past is None or past.empty:
            continue
        dates = past['_race_date'].to_numpy(dtype='datetime64[ns]')
        target_dates = t.loc[positions, '_race_date'].to_numpy(dtype='datetime64[ns]')
        insertion = np.searchsorted(dates, target_dates, side='left') - 1
        valid = insertion >= 0
        if not valid.any():
            continue
        idx = np.asarray(positions)[valid]
        pidx = insertion[valid]
        wins = past['cum_wins'].to_numpy(dtype=float)[pidx]
        starts = past['cum_starts'].to_numpy(dtype=float)[pidx]
        result.loc[idx] = (wins + base_rate * PRIOR_STRENGTH) / (starts + PRIOR_STRENGTH)
    return result


def assign_prior_last_rank(history, target):
    """Most recent horse result from a strictly earlier date; 99 represents DNF."""
    result = pd.Series(6.0, index=target.index, dtype=float)
    if history.empty or target.empty:
        return result
    h = history.loc[history['_race_date'].notna()].copy()
    t = target.loc[target['_race_date'].notna()].copy()
    if h.empty or t.empty:
        return result
    h['_horse_key'] = make_entity_keys(h, ['_horse_key'])
    t['_horse_key'] = make_entity_keys(t, ['_horse_key'])
    daily = (
        h.groupby(['_horse_key', '_race_date'], observed=True, as_index=False)
        .agg(day_rank=('numeric_rank', 'min'))
        .sort_values(['_horse_key', '_race_date'])
    )
    lookup = {key: group for key, group in daily.groupby('_horse_key', sort=False)}
    for key, positions in t.groupby('_horse_key', sort=False).groups.items():
        past = lookup.get(key)
        if past is None or past.empty:
            continue
        dates = past['_race_date'].to_numpy(dtype='datetime64[ns]')
        target_dates = t.loc[positions, '_race_date'].to_numpy(dtype='datetime64[ns]')
        insertion = np.searchsorted(dates, target_dates, side='left') - 1
        valid = insertion >= 0
        if valid.any():
            idx = np.asarray(positions)[valid]
            result.loc[idx] = past['day_rank'].to_numpy(dtype=float)[insertion[valid]]
    return result


def add_training_compatible_features(raw):
    """Calculate all 19 training features for historical and upcoming rows.

    Unfinished rows are kept for prediction but never contribute to historical
    win-rate or last-rank statistics. Every historical statistic excludes all
    results on the prediction row's own date, matching the training pipeline.
    """
    if '賽事編號' not in raw.columns:
        raise ValueError('CSV 必須包含「賽事編號」欄位。')
    if '馬號' not in raw.columns:
        raise ValueError('CSV 必須包含「馬號」欄位。')

    df = raw.copy().reset_index(drop=True)
    df['賽事編號'] = df['賽事編號'].astype(str).str.strip()
    date_match = df['賽事編號'].str.extract(r'(20\d{6})', expand=False)
    race_dates = pd.to_datetime(date_match, format='%Y%m%d', errors='coerce')
    if race_dates.isna().any():
        date_col = next((c for c in ['賽事日期', '日期', 'race_date', 'date'] if c in df.columns), None)
        if date_col:
            fallback = pd.to_datetime(df[date_col], errors='coerce')
            race_dates = race_dates.fillna(fallback)
    df['_race_date'] = race_dates
    if df['_race_date'].isna().any():
        bad_count = int(df['_race_date'].isna().sum())
        raise ValueError(f'{bad_count} 列無法從賽事編號／日期欄解析日期，請確認日期格式。')

    if '獨贏賠率' not in df.columns:
        df['獨贏賠率'] = np.nan
    raw_win_odds = pd.to_numeric(df['獨贏賠率'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    df['_valid_training_odds'] = raw_win_odds.notna() & (raw_win_odds > 1.0)
    df['獨贏賠率'] = raw_win_odds.where(raw_win_odds > 1.0, 10.0).fillna(10.0)
    if '位置賠率' not in df.columns:
        df['位置賠率'] = 1.0 + (df['獨贏賠率'] - 1.0) / 3.2

    # Blank result = upcoming/unrun; non-numeric recorded result = non-winner (99).
    if '名次' in df.columns:
        rank_text = df['名次'].astype('string').str.strip()
        df['_has_result'] = rank_text.notna() & rank_text.ne('')
        df['numeric_rank'] = pd.to_numeric(df['名次'], errors='coerce').fillna(99.0)
    else:
        df['_has_result'] = False
        df['numeric_rank'] = 99.0
    df['target_win'] = np.where(df['_has_result'], (df['numeric_rank'] == 1).astype(int), np.nan)

    # Deduplicate only the history contribution, while keeping app rows intact.
    horse_id_col = next(
        (c for c in ['馬匹編號', '馬匹代號', 'horse_id', 'horse_code'] if c in df.columns),
        '馬名' if '馬名' in df.columns else '馬號',
    )
    df['_horse_key'] = df[horse_id_col].astype('string').str.strip()
    df['_horse_key'] = df['_horse_key'].mask(df['_horse_key'].isna() | df['_horse_key'].eq(''), '__UNKNOWN__').astype(str)
    for col, key_col in [('騎師', '_jockey_key'), ('練馬師', '_trainer_key'), ('場地', '_venue_key')]:
        if col in df.columns:
            values = df[col].astype('string').str.strip()
            df[key_col] = values.mask(values.isna() | values.eq(''), '__UNKNOWN__').astype(str)
        else:
            df[key_col] = '__UNKNOWN__'

    history = df.loc[
        df['_has_result'] & df['target_win'].notna() & df['_valid_training_odds']
    ].copy()
    history = history.drop_duplicates(subset=['賽事編號', '馬號'], keep='last')

    # Current race market features (updated live odds are already in df).
    df['market_prob_raw'] = 1.0 / df['獨贏賠率']
    prob_sum = df.groupby('賽事編號', observed=True)['market_prob_raw'].transform('sum')
    df['market_implied_prob'] = (df['market_prob_raw'] / prob_sum.replace(0, np.nan)).fillna(0.0)
    df['odds_rank'] = df.groupby('賽事編號', observed=True)['獨贏賠率'].rank(method='min', ascending=True)
    df['is_favorite'] = (df['odds_rank'] == 1).astype(int)

    if '排位檔位' in df.columns:
        df['排位檔位'] = pd.to_numeric(df['排位檔位'], errors='coerce').fillna(7.0)
    else:
        df['排位檔位'] = 7.0
    if '距離' in df.columns:
        df['距離'] = pd.to_numeric(df['距離'], errors='coerce').fillna(1200.0)
    else:
        df['距離'] = 1200.0

    if '實際負磅' in df.columns:
        df['實際負磅'] = pd.to_numeric(df['實際負磅'], errors='coerce')
        race_mean = df.groupby('賽事編號', observed=True)['實際負磅'].transform('mean')
        df['實際負磅'] = df['實際負磅'].fillna(race_mean).fillna(120.0)
        race_mean = df.groupby('賽事編號', observed=True)['實際負磅'].transform('mean')
        df['weight_diff'] = df['實際負磅'] - race_mean
        df['weight_rank'] = df.groupby('賽事編號', observed=True)['實際負磅'].rank(ascending=False, method='min')
    else:
        df['weight_diff'] = 0.0
        df['weight_rank'] = 6.0

    # Calculate rates against outcome-known historical rows only.
    df['jockey_win_rate'] = assign_prior_rate(history, df, ['_jockey_key'], 0.10)
    df['trainer_win_rate'] = assign_prior_rate(history, df, ['_trainer_key'], 0.10)
    df['combo_win_rate'] = (df['jockey_win_rate'] + df['trainer_win_rate']) / 2.0
    df['horse_win_rate'] = assign_prior_rate(history, df, ['_horse_key'], 0.10)
    df['horse_last_rank'] = assign_prior_last_rank(history, df)
    df['horse_surface_win_rate'] = assign_prior_rate(history, df, ['_horse_key', '_venue_key'], 0.08)
    df['_dist_group'] = pd.cut(
        df['距離'], bins=[0, 1200, 1600, 2000, 2400, 10000],
        labels=['<=1200', '1201-1600', '1601-2000', '2001-2400', '>2400'],
        include_lowest=True,
    ).astype('string').fillna('unknown')
    # The training feature's distance groups use the normalized distance feature.
    history['_dist_group'] = pd.cut(
        pd.to_numeric(history.get('距離', pd.Series(1200.0, index=history.index)), errors='coerce').fillna(1200.0),
        bins=[0, 1200, 1600, 2000, 2400, 10000],
        labels=['<=1200', '1201-1600', '1601-2000', '2001-2400', '>2400'],
        include_lowest=True,
    ).astype('string').fillna('unknown')
    df['horse_dist_win_rate'] = assign_prior_rate(history, df, ['_horse_key', '_dist_group'], 0.08)

    # Same shared implementation used to prepare the consolidated training CSV.
    try:
        df, _ = add_horse_rest_and_course_features(df, verbose=False)
    except Exception as exc:
        raise ValueError(f'休賽／同場地路程特徵計算失敗：{exc}') from exc
    df['days_since_last_race'] = pd.to_numeric(df['days_since_last_race'], errors='coerce').fillna(999.0)
    df['horse_course_starts'] = pd.to_numeric(df['horse_course_starts'], errors='coerce').fillna(0.0)
    df['horse_course_win_rate'] = pd.to_numeric(df['horse_course_win_rate'], errors='coerce').fillna(0.10)
    df['horse_course_top3_rate'] = pd.to_numeric(df['horse_course_top3_rate'], errors='coerce').fillna(0.30)
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Catch model/schema drift rather than silently feeding reordered features.
    raw_model_features = getattr(model, 'feature_name_', None)
    if raw_model_features is None or len(raw_model_features) == 0:
        raw_model_features = getattr(model, 'feature_names_in_', None)
    model_features = [str(name) for name in raw_model_features] if raw_model_features is not None else []
    if model_features and model_features != FEATURE_COLS:
        raise ValueError(
            '模型特徵清單與 app.py 不一致。請用相同 FEATURE_COLS 重訓／部署模型。'
        )
    return df

if uploaded_file is not None:
    # 雙編碼容錯讀取
    try:
        df_raw = pd.read_csv(uploaded_file, encoding='utf-8-sig')
    except:
        uploaded_file.seek(0)
        df_raw = pd.read_csv(uploaded_file, encoding='cp950')

    history_raw = None
    if history_file is not None:
        try:
            history_raw = pd.read_csv(history_file, encoding='utf-8-sig')
        except UnicodeDecodeError:
            history_file.seek(0)
            history_raw = pd.read_csv(history_file, encoding='cp950')
        required_history = ['賽事編號', '馬名', '名次', 'racecourse_code', 'official_distance_m']
        missing_history = [c for c in required_history if c not in history_raw.columns]
        if missing_history:
            st.error(f"歷史 enriched CSV 缺少必要欄位：{missing_history}")
            st.stop()
        st.sidebar.success(f"已載入歷史賽果：{len(history_raw):,} 筆")
        
    st.success("✅ 賽事資料載入成功！")
    
    # 確保賠率欄位存在；沒有實際 PLACE 賠率時以估算值作初始值
    if '獨贏賠率' not in df_raw.columns:
        df_raw['獨贏賠率'] = 10.0
    if '位置賠率' not in df_raw.columns:
        win_values = pd.to_numeric(df_raw['獨贏賠率'], errors='coerce').fillna(10.0)
        df_raw['位置賠率'] = 1.0 + (win_values - 1.0) / 3.2

    # 使用 Session State 保留已抓取的賠率
    if 'df_data' not in st.session_state or st.session_state.get('uploaded_filename') != uploaded_file.name:
        st.session_state['df_data'] = df_raw.copy()
        st.session_state['uploaded_filename'] = uploaded_file.name
        st.session_state['odds_editor_version'] = 0

    st.markdown("---")
    st.subheader("⚡ 臨場賠率更新中心")

    col_v, col_d, col_r = st.columns([1.5, 2, 1.5])
    venue_input = col_v.selectbox("賽事場地", ["HV (跑馬地)", "ST (沙田)"])
    venue_code = "HV" if venue_input.startswith("HV") else "ST"

    # 從賽事編號取日期（例如 20260927-01）；格式為 YYYY-MM-DD
    sample_id = str(df_raw['賽事編號'].iloc[0]).strip() if len(df_raw) else ""
    date_match = re.search(r"(20\d{6})", sample_id)
    if date_match:
        raw_date = date_match.group(1)
        auto_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
    else:
        auto_date = datetime.now().strftime("%Y-%m-%d")
    api_date = col_d.text_input("API 查詢日期 (YYYY-MM-DD)", value=auto_date)

    race_series = df_raw['賽事編號'].map(extract_race_no)
    races_available = sorted(int(x) for x in race_series.dropna().unique())
    if races_available:
        target_race = col_r.selectbox("查看場次", races_available)
    else:
        target_race = col_r.number_input("查看場次", min_value=1, max_value=15, value=1, step=1)

    auto_col, interval_col, button_col = st.columns([1.5, 1.5, 2.5])
    auto_refresh = auto_col.checkbox("自動更新全部場次", value=False)
    refresh_seconds = interval_col.selectbox(
        "更新間隔（秒）", [30, 60, 120, 300], index=1, disabled=not auto_refresh
    )
    manual_refresh = button_col.button(
        "🔄 立即更新全部場次賠率", use_container_width=True
    )

    if auto_refresh:
        st_autorefresh(
            interval=int(refresh_seconds * 1000),
            key="hkjc_live_odds_autorefresh",
        )

    if auto_refresh or manual_refresh:
        with st.spinner("正在讀取當日所有場次的 WIN／PLACE 即時賠率…"):
            live_by_race, server_updated_at, error = fetch_live_odds(api_date, venue_code)

        if live_by_race:
            df_temp = st.session_state['df_data'].copy()
            row_races = df_temp['賽事編號'].map(extract_race_no)
            row_horses = df_temp['馬號'].map(normalize_horse_no)
            updated_rows = 0

            for race_no, markets in live_by_race.items():
                race_mask = row_races == int(race_no)
                for horse_no, value in markets['WIN'].items():
                    horse_mask = race_mask & (row_horses == horse_no)
                    updated_rows += int(horse_mask.sum())
                    df_temp.loc[horse_mask, '獨贏賠率'] = value
                for horse_no, value in markets['PLA'].items():
                    horse_mask = race_mask & (row_horses == horse_no)
                    df_temp.loc[horse_mask, '位置賠率'] = value

            st.session_state['df_data'] = df_temp
            st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version', 0) + 1
            st.session_state['odds_last_status'] = (
                f"成功取得 {len(live_by_race)} 場賠率；更新到 CSV 的馬匹資料列：{updated_rows}。"
            )
            st.session_state['odds_server_time'] = server_updated_at
            st.session_state['odds_last_error'] = None
        else:
            st.session_state['odds_last_error'] = error

    if st.session_state.get('odds_last_error'):
        st.warning(st.session_state['odds_last_error'])
    elif st.session_state.get('odds_last_status'):
        st.success(st.session_state['odds_last_status'])
        if st.session_state.get('odds_server_time'):
            st.caption(f"馬會資料最後更新時間：{st.session_state['odds_server_time']}")
    if auto_refresh:
        st.caption(f"自動刷新已開啟，每 {refresh_seconds} 秒查詢一次；每次僅送出一個唯讀請求。")

    # ==========================================
    # 互動式臨場賠率輸入面板 (綁定 Session State)
    # ==========================================
    st.info("👇 賠率已自動載入下方表格。你也可以直接點擊表格進行手動微調。")
    
    edit_columns = ['賽事編號', '馬號', '馬名', '排位檔位', '獨贏賠率', '位置賠率']
    df_editable = st.session_state['df_data'][edit_columns].copy()
    
    edited_df = st.data_editor(
        df_editable,
        disabled=['賽事編號', '馬號', '馬名', '排位檔位'], 
        use_container_width=True,
        hide_index=True,
        key=f"odds_editor_{st.session_state.get('odds_editor_version', 0)}"
    )
    
    # 將手動編輯或 API 抓取的結果套用到主要 DataFrame
    df = st.session_state['df_data'].copy()
    df['獨贏賠率'] = pd.to_numeric(edited_df['獨贏賠率'], errors='coerce').fillna(10.0)
    df['位置賠率'] = pd.to_numeric(edited_df['位置賠率'], errors='coerce').fillna(1.5)
    
    # Build the training-time features using historical completed races as context.
    try:
        prediction_race_ids = set(df['賽事編號'].astype(str).str.strip())
        df['_is_prediction_row'] = True
        if history_raw is not None:
            history_context = history_raw.copy()
            history_context['_is_prediction_row'] = False
            combined = pd.concat([history_context, df], ignore_index=True, sort=False)
            featured = add_training_compatible_features(combined)
            df = featured.loc[featured['_is_prediction_row'].eq(True)].copy()
            df = df.drop(columns=['_is_prediction_row'], errors='ignore')
            st.caption('已使用歷史賽果計算賽前特徵；歷史資料本身不會列入預測結果。')
        else:
            df = add_training_compatible_features(df)
            st.warning(
                '尚未上傳歷史 enriched CSV。模型仍可運行，但歷史表現特徵會使用預設值，'
                '不適合作為完整的19特徵公平測試。'
            )
    except Exception as exc:
        st.error(f"特徵計算失敗：{type(exc).__name__}: {exc}")
        st.stop()

    X_predict = df[FEATURE_COLS]

    df['pred_win_prob'] = model.predict_proba(X_predict)[:, 1]
    df['ev'] = df['pred_win_prob'] * df['獨贏賠率']

    st.markdown("---")
    tab1, tab2 = st.tabs(["🎯 各場次預測推薦", "📈 歷史回測 (獨贏/位置/位置Q)"])

    # ---------------- 分頁 1: 賽前預測 ----------------
    with tab1:
        st.subheader("🎯 各場次 AI 智慧投注推薦清單")

        recommendations = []
        for race_id, group in df.groupby('賽事編號'):
            filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
            sorted_group = filtered_group.sort_values(by='ev', ascending=False).reset_index(drop=True)

            if len(sorted_group) >= 2:
                top1 = sorted_group.iloc[0]
                top2 = sorted_group.iloc[1]
                top3 = sorted_group.iloc[2] if len(sorted_group) >= 3 else top2

                win_pick = f"馬號 {top1['馬號']} ({top1['馬名']}) [勝率:{top1['pred_win_prob']*100:.1f}%, EV:{top1['ev']:.2f}]"
                q_pick = f"{top1['馬號']} + {top2['馬號']} ({top1['馬名']} / {top2['馬名']})"
                qp_pick = f"{top1['馬號']} + {top2['馬號']} 或 {top1['馬號']} + {top3['馬號']}"

                recommendations.append({
                    '賽事編號': race_id,
                    '🎯 獨贏推薦': win_pick,
                    '🔗 連贏推薦 (Q)': q_pick,
                    '🔗 位置Q推薦 (QP)': qp_pick
                })

        rec_df = pd.DataFrame(recommendations)
        if rec_df.empty:
            st.warning("⚠️ 沒有符合當前篩選條件的馬匹，請試著放寬左側欄的 EV 或賠率限制。")
        else:
            st.dataframe(rec_df, use_container_width=True)

    # ---------------- 分頁 2: 賽後回測 ----------------
    with tab2:
        st.subheader("📊 多彩種策略回測總覽")
        if (df['numeric_rank'] == 99).all():
            st.warning("⚠️ 系統偵測到目前的 CSV 中沒有真實的『名次』紀錄 (或全部為空值)。請上傳已完賽並包含名次結果的歷史檔案來執行回測。")
        else:
            sub_tab1, sub_tab2, sub_tab3 = st.tabs(["🥇 獨贏 (Win)", "🥈 位置 (Place)", "🔗 位置Q (QP)"])
            BET_AMOUNT = 100 
            
            with sub_tab1:
                st.markdown("#### 🥇 獨贏 (Win) 策略回測")
                win_invested, win_return, win_bets, win_hits = 0, 0, 0, 0
                win_records = []
                for race_id, group in df.groupby('賽事編號'):
                    filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
                    sorted_group = filtered_group.sort_values(by='ev', ascending=False).reset_index(drop=True)
                    if len(sorted_group) > 0:
                        pick = sorted_group.iloc[0]
                        win_bets += 1
                        win_invested += BET_AMOUNT
                        is_hit = (pick['numeric_rank'] == 1)
                        if is_hit:
                            win_hits += 1
                            payout = BET_AMOUNT * pick['獨贏賠率']
                            win_return += payout
                            result_str = "✅ 贏"
                        else:
                            payout = 0
                            result_str = "❌ 輸"
                        win_records.append({
                            '賽事編號': race_id,
                            '投注馬號': f"{pick['馬號']} ({pick['馬名']})",
                            '實際名次': str(pick['名次']).replace('.0', ''),
                            '獨贏賠率': pick['獨贏賠率'],
                            'EV': round(pick['ev'], 2),
                            '結果': result_str,
                            '派彩': f"${payout:.1f}",
                            '淨盈虧': f"${payout - BET_AMOUNT:.1f}"
                        })
                if win_bets > 0:
                    roi = ((win_return - win_invested) / win_invested) * 100
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("投注場數", f"{win_bets} 場")
                    col2.metric("命中場數", f"{win_hits} 場", f"勝率: {win_hits/win_bets*100:.1f}%")
                    col3.metric("總成本", f"${win_invested}")
                    col4.metric("總回收", f"${win_return:.1f}", f"ROI: {roi:.2f}%")
                    st.markdown("##### 📝 獨贏明細")
                    st.dataframe(pd.DataFrame(win_records), use_container_width=True)
                else:
                    st.info("💡 目前設定下沒有符合獨贏出手的場次。")

            with sub_tab2:
                st.markdown("#### 🥈 位置 (Place) 策略回測")
                place_invested, place_return, place_bets, place_hits = 0, 0, 0, 0
                place_records = []
                for race_id, group in df.groupby('賽事編號'):
                    filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
                    sorted_group = filtered_group.sort_values(by='ev', ascending=False).reset_index(drop=True)
                    if len(sorted_group) > 0:
                        pick = sorted_group.iloc[0]
                        place_bets += 1
                        place_invested += BET_AMOUNT
                        is_hit = (pick['numeric_rank'] <= 3)
                        p_odds = float(pick['位置賠率']) if pd.notna(pick['位置賠率']) and float(pick['位置賠率']) > 1.0 else 1.5
                        if is_hit:
                            place_hits += 1
                            payout = BET_AMOUNT * p_odds
                            place_return += payout
                            result_str = "✅ 命中位置"
                        else:
                            payout = 0
                            result_str = "❌ 未入前三"
                        place_records.append({
                            '賽事編號': race_id,
                            '投注馬號': f"{pick['馬號']} ({pick['馬名']})",
                            '實際名次': str(pick['名次']).replace('.0', ''),
                            '位置賠率': round(p_odds, 2),
                            'EV': round(pick['ev'], 2),
                            '結果': result_str,
                            '派彩': f"${payout:.1f}",
                            '淨盈虧': f"${payout - BET_AMOUNT:.1f}"
                        })
                if place_bets > 0:
                    roi = ((place_return - place_invested) / place_invested) * 100
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("投注場數", f"{place_bets} 場")
                    col2.metric("命中位置場數", f"{place_hits} 場", f"位置勝率: {place_hits/place_bets*100:.1f}%")
                    col3.metric("總成本", f"${place_invested}")
                    col4.metric("總回收", f"${place_return:.1f}", f"ROI: {roi:.2f}%")
                    st.markdown("##### 📝 位置明細")
                    st.dataframe(pd.DataFrame(place_records), use_container_width=True)
                else:
                    st.info("💡 目前設定下沒有符合位置出手的場次。")

            with sub_tab3:
                st.markdown("#### 🔗 位置Q (QP) 策略回測")
                qp_invested, qp_return, qp_bets, qp_hits = 0, 0, 0, 0
                qp_records = []
                for race_id, group in df.groupby('賽事編號'):
                    filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
                    sorted_group = filtered_group.sort_values(by='ev', ascending=False).reset_index(drop=True)
                    if len(sorted_group) >= 2:
                        top1 = sorted_group.iloc[0]
                        top2 = sorted_group.iloc[1]
                        qp_bets += 1
                        qp_invested += BET_AMOUNT
                        is_hit = (top1['numeric_rank'] <= 3) and (top2['numeric_rank'] <= 3)
                        p1_odds = float(top1['位置賠率']) if pd.notna(top1['位置賠率']) else 1.5
                        p2_odds = float(top2['位置賠率']) if pd.notna(top2['位置賠率']) else 1.5
                        estimated_qp_odds = round(p1_odds * p2_odds * 1.8, 1) 
                        if is_hit:
                            qp_hits += 1
                            payout = BET_AMOUNT * estimated_qp_odds
                            qp_return += payout
                            result_str = "✅ 命中位置Q"
                        else:
                            payout = 0
                            result_str = "❌ 落空"
                        qp_records.append({
                            '賽事編號': race_id,
                            'QP 組合': f"{top1['馬號']} + {top2['馬號']} ({top1['馬名']} / {top2['馬名']})",
                            '實際名次': f"首選:第{str(top1['名次']).replace('.0','')}名 | 次選:第{str(top2['名次']).replace('.0','')}名",
                            '估算QP賠率': estimated_qp_odds,
                            '結果': result_str,
                            '派彩': f"${payout:.1f}",
                            '淨盈虧': f"${payout - BET_AMOUNT:.1f}"
                        })
                if qp_bets > 0:
                    roi = ((qp_return - qp_invested) / qp_invested) * 100
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("投注場數", f"{qp_bets} 場")
                    col2.metric("命中 QP 場數", f"{qp_hits} 場", f"QP 命中率: {qp_hits/qp_bets*100:.1f}%")
                    col3.metric("總成本", f"${qp_invested}")
                    col4.metric("總回收", f"${qp_return:.1f}", f"ROI: {roi:.2f}%")
                    st.markdown("##### 📝 位置Q (QP) 明細")
                    st.dataframe(pd.DataFrame(qp_records), use_container_width=True)
                else:
                    st.info("💡 目前設定下沒有符合位置 Q 出手的場次 (需同場至少有 2 匹馬符合 EV/賠率門檻)。")
else:
    st.info("👈 請在左側上傳賽事 CSV 檔案來啟動系統！")
