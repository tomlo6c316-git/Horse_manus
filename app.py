import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import requests
import re
import time
import html
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh

# 頁面基本設定
st.set_page_config(
    page_title="HKJC AI 智能賽馬預測系統",
    page_icon="🐎",
    layout="wide"
)

st.title("🐎 HKJC 旗艦 15 大特徵 AI 預測系統 (支援即時賠率)")
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

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 投注策略參數設定")
min_ev = st.sidebar.slider("最小期望值 (EV 門檻)", 0.0, 1.5, 0.0, 0.05)
min_odds = st.sidebar.number_input("最低獨贏賠率", min_value=1.0, max_value=50.0, value=3.0)
max_odds = st.sidebar.number_input("最高獨贏賠率", min_value=1.0, max_value=100.0, value=20.0)

# ==========================================
# HKJC 新版 GraphQL：一次取得當日所有場次 WIN / PLACE 賠率
# ==========================================
GRAPHQL_URL = "https://info.cld.hkjc.com/graphql/base/"
ODDS_QUERY = """query racing($date: String,$venueCode: String, $oddsTypes: [OddsType],$raceNo: Int) {
  raceMeetings(date: $date, venueCode:$venueCode) {
    pmPools(oddsTypes: $oddsTypes, raceNo:$raceNo) {
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

HK_TZ = ZoneInfo("Asia/Hong_Kong")


@st.cache_resource
def get_odds_executor():
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="hkjc-readonly-monitor")


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_race_start_time(date_str, venue, race_no):
    """Read the scheduled off time from the official HKJC racecard page."""
    try:
        race_date = datetime.strptime(str(date_str), "%Y-%m-%d")
        response = requests.get(
            "https://racing.hkjc.com/zh-hk/local/information/racecard",
            params={
                "racedate": race_date.strftime("%Y/%m/%d"),
                "Racecourse": venue,
                "RaceNo": int(race_no),
            },
            headers={"User-Agent": GRAPHQL_HEADERS["User-Agent"]},
            timeout=12,
        )
        response.raise_for_status()
        page_text = html.unescape(re.sub(r"<[^>]+>", " ", response.text))
        page_text = re.sub(r"\s+", " ", page_text)
        match = re.search(
            rf"第\s*{int(race_no)}\s*場.{{0,350}}?((?:[01]?\d|2[0-3]):[0-5]\d)",
            page_text,
        )
        return datetime.strptime(match.group(1), "%H:%M").time() if match else None
    except (requests.RequestException, ValueError, TypeError):
        return None


def classify_pool_status(pool):
    """Fail closed: only the official START_SELL/START_SELL pair is treated as open."""
    status = str((pool or {}).get("status") or "").strip().upper()
    sell_status = str((pool or {}).get("sellStatus") or "").strip().upper()
    codes = {status, sell_status}
    if status == "START_SELL" and sell_status == "START_SELL":
        return "受注中"
    if codes.intersection({"SUSPENDED", "SELLINGSTOPPED", "HOLD"}):
        return "暫停受注"
    if codes.intersection({"STOP_SELL", "CLOSED", "PAYOUT"}):
        return "停止受注／已封盤"
    return "狀態未知"

def normalize_horse_no(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except (ValueError, TypeError):
        return text

def extract_race_no(value):
    match = re.search(r"-(\d+)\s*$", str(value).strip())
    return int(match.group(1)) if match else None

def fetch_live_odds(date_str, venue):
    body = {
        "operationName": "racing",
        "variables": {
            "date": date_str,
            "venueCode": venue,
            "raceNo": None,
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
            return None, None, None, f"GraphQL 錯誤：{payload['errors']}"

        meetings = (payload.get("data") or {}).get("raceMeetings") or []
        if not meetings:
            return None, None, None, "找不到該日期／場地的賽事資料。"

        odds_by_race = {}
        status_by_race = {}
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
                status_by_race.setdefault(race_number, {})[pool_type] = {
                    "status": pool.get("status"),
                    "sellStatus": pool.get("sellStatus"),
                    "lastUpdateTime": pool.get("lastUpdateTime"),
                }
                market = odds_by_race.setdefault(
                    race_number, {"WIN": {}, "PLA": {}}
                )
                for node in pool.get("oddsNodes") or []:
                    horse_no = normalize_horse_no(node.get("combString"))
                    odds_value = node.get("oddsValue")
                    if horse_no and odds_value not in (None, ""):
                        market[pool_type][horse_no] = float(odds_value)

        if not status_by_race:
            return None, None, None, "賽事有回應，但目前尚無可用彩池狀態。"
        return (
            odds_by_race,
            status_by_race,
            (max(timestamps) if timestamps else None),
            None,
        )

    except requests.RequestException as exc:
        return None, None, None, f"連線錯誤：{exc}"
    except (ValueError, TypeError, KeyError) as exc:
        return None, None, None, f"解析回應錯誤：{exc}"

if uploaded_file is not None:
    # 雙編碼容錯讀取
    try:
        df_raw = pd.read_csv(uploaded_file, encoding='utf-8-sig')
    except:
        uploaded_file.seek(0)
        df_raw = pd.read_csv(uploaded_file, encoding='cp950')
        
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

    # ==========================================
    # 互動式臨場賠率輸入面板 (移至上方，讓流程更順暢)
    # ==========================================
    st.info("👇 以下是當前載入的賽事資料。你可以點擊表格進行手動微調，或透過下方的更新面板抓取即時賠率。")
    
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
    
    # 1. 特徵工程
    df['market_prob'] = 1 / df['獨贏賠率']
    prob_sum = df.groupby('賽事編號')['market_prob'].transform('sum')
    df['market_implied_prob'] = df['market_prob'] / prob_sum

    df['odds_rank'] = df.groupby('賽事編號')['獨贏賠率'].rank(method='min')
    df['is_favorite'] = (df['odds_rank'] == 1).astype(int)
    df['排位檔位'] = pd.to_numeric(df['排位檔位'], errors='coerce').fillna(7)

    if '實際負磅' in df.columns:
        df['實際負磅'] = pd.to_numeric(df['實際負磅'], errors='coerce').fillna(120)
        avg_weight = df.groupby('賽事編號')['實際負磅'].transform('mean')
        df['weight_diff'] = df['實際負磅'] - avg_weight
        df['weight_rank'] = df.groupby('賽事編號')['實際負磅'].rank(ascending=False, method='min')
    else:
        df['weight_diff'] = 0.0
        df['weight_rank'] = 6.0

    rank_source = df['名次'] if '名次' in df.columns else pd.Series(99, index=df.index)
    df['numeric_rank'] = pd.to_numeric(rank_source, errors='coerce').fillna(99)

    df['jockey_win_rate'] = df.get('jockey_win_rate', 0.12)
    df['trainer_win_rate'] = df.get('trainer_win_rate', 0.12)
    df['combo_win_rate'] = df.get('combo_win_rate', 0.10)
    df['horse_win_rate'] = df.get('horse_win_rate', 0.10)
    df['horse_last_rank'] = df.get('horse_last_rank', 6.0)

    if '距離' not in df.columns: df['距離'] = 1200
    if 'horse_surface_win_rate' not in df.columns: df['horse_surface_win_rate'] = 0.08
    if 'horse_dist_win_rate' not in df.columns: df['horse_dist_win_rate'] = 0.08

    # 2. 提取 15 大特徵並預測勝率與 EV
    feature_cols = [
        'market_implied_prob', '獨贏賠率', 'odds_rank', 'is_favorite',
        '排位檔位', 'weight_diff', 'weight_rank',
        'jockey_win_rate', 'trainer_win_rate', 'combo_win_rate',
        'horse_win_rate', 'horse_last_rank',
        '距離', 'horse_surface_win_rate', 'horse_dist_win_rate'
    ]

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X_predict = df[feature_cols].fillna(0)
    df['pred_win_prob'] = model.predict_proba(X_predict)[:, 1]
    df['ev'] = df['pred_win_prob'] * df['獨贏賠率']

    st.markdown("---")
    tab1, tab2 = st.tabs(["🎯 各場次預測推薦", "📈 歷史回測 (獨贏/位置/位置Q)"])

    # ---------------- 分頁 1: 賽前預測 ----------------
    with tab1:
        # ==========================================
     # ==========================================
        # ⚡ 臨場賠率更新中心 (移至推薦表格正上方)
        # ==========================================
        st.subheader("⚡ 臨場賠率更新中心與實戰計時")
        
        # 1. 這裡改成 4 個欄位，分配寬度比例
        col_v, col_d, col_r, col_t = st.columns([1.2, 1.8, 1.2, 1.8])
        venue_input = col_v.selectbox("賽事場地", ["HV (跑馬地)", "ST (沙田)"])
        venue_code = "HV" if venue_input.startswith("HV") else "ST"

        # 從賽事編號取日期（例如 20260927-01）
        sample_id = str(df_raw['賽事編號'].iloc[0]).strip() if len(df_raw) else ""
        date_match = re.search(r"(20\d{6})", sample_id)
        if date_match:
            raw_date = date_match.group(1)
            auto_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
        else:
            auto_date = datetime.now(HK_TZ).strftime("%Y-%m-%d")
        api_date = col_d.text_input("API 查詢日期 (YYYY-MM-DD)", value=auto_date)

        race_series = df_raw['賽事編號'].map(extract_race_no)
        races_available = sorted(int(x) for x in race_series.dropna().unique())
        if races_available:
            target_race = col_r.selectbox("查看場次", races_available)
        else:
            target_race = col_r.number_input("查看場次", min_value=1, max_value=15, value=1, step=1)

        # Official scheduled time is the default; the time input remains an explicit override.
        official_time = fetch_race_start_time(api_date, venue_code, int(target_race))
        fallback_time = datetime.strptime("14:30", "%H:%M").time()
        target_time = col_t.time_input(
            "開跑時間（香港時間，可手動覆蓋）",
            value=official_time or fallback_time,
            key=f"race_time_{api_date}_{venue_code}_{int(target_race)}",
        )
        if official_time:
            st.caption(f"已從 HKJC 官方排位表讀取開跑時間：{official_time.strftime('%H:%M')}（香港時間）。")
        else:
            st.warning("未能讀取官方排位時間；請核對或手動設定。此倒數只供提醒，不代表投注池仍接受投注。")

        reminder_seconds = st.select_slider(
            "賽前提醒提前時間（秒）", options=[30, 60, 90, 120, 180], value=60
        )
        try:
            target_date = datetime.strptime(api_date, "%Y-%m-%d").date()
            target_datetime = datetime.combine(target_date, target_time, tzinfo=HK_TZ)
            now_hkt = datetime.now(HK_TZ)
            time_diff = (target_datetime - now_hkt).total_seconds()
        except ValueError:
            time_diff = None
            st.error("日期格式無效，請使用 YYYY-MM-DD；倒數暫時停用。")

        if time_diff is not None and time_diff > 0:
            remaining = math.ceil(time_diff)
            hours, remainder = divmod(remaining, 3600)
            mins, secs = divmod(remainder, 60)
            if hours:
                countdown = f"{hours} 小時 {mins} 分 {secs} 秒"
            else:
                countdown = f"{mins} 分 {secs} 秒"
            st.info(f"⏳ 距離排定開跑：**{countdown}**")
            if time_diff <= reminder_seconds:
                st.warning(
                    f"賽前提醒：距排定開跑約 {math.ceil(time_diff)} 秒。受注可能提前停止；請以官方彩池狀態為準。"
                )
        elif time_diff is not None and time_diff > -300:
            st.warning("已到或略過排定開跑時間；若賽事延誤，仍須以官方受注狀態為準。")
        elif time_diff is not None:
            st.info("已超過排定開跑時間；不可只憑倒數推斷投注池仍開放。")

        # Timer ticks every second, while the read-only API request is rate-limited separately.
        auto_col, interval_col, button_col = st.columns([2.0, 1.2, 2.1])
        auto_refresh = auto_col.checkbox("啟用秒級倒數及受注狀態監測（只讀）", value=True)
        refresh_seconds = interval_col.selectbox(
            "官方狀態查詢間隔（秒）", [5, 10, 15, 30], index=1, disabled=not auto_refresh
        )
        manual_refresh = button_col.button(
            "🔄 立即更新賠率／受注狀態", use_container_width=True
        )

        poll_source_key = f"{api_date}|{venue_code}"
        if st.session_state.get("odds_monitor_source_key") != poll_source_key:
            st.session_state["odds_monitor_source_key"] = poll_source_key
            st.session_state["odds_last_poll_monotonic"] = 0.0
            st.session_state["odds_last_success_monotonic"] = None
            st.session_state["odds_pool_status_by_race"] = {}
            st.session_state["odds_last_error"] = None
            st.session_state["odds_last_status"] = None
            st.session_state["odds_server_time"] = None
            st.session_state["odds_checked_at_hkt"] = None
            st.session_state["pool_watch_key"] = None
            st.session_state["pool_transition_notice"] = None

        # Apply a completed background request without blocking the one-second countdown.
        fetch_future = st.session_state.get("odds_fetch_future")
        if fetch_future is not None and fetch_future.done():
            future_source_key = st.session_state.get("odds_fetch_future_source_key")
            try:
                live_by_race, pool_status_by_race, server_updated_at, error = fetch_future.result()
            except Exception as exc:
                live_by_race, pool_status_by_race, server_updated_at = None, None, None
                error = f"背景查詢失敗：{exc}"
            st.session_state["odds_fetch_future"] = None
            st.session_state["odds_fetch_future_source_key"] = None

            if future_source_key == poll_source_key:
                if error:
                    st.session_state["odds_last_error"] = error
                else:
                    pool_status_by_race = pool_status_by_race or {}
                    st.session_state["odds_pool_status_by_race"] = pool_status_by_race
                    st.session_state["odds_last_success_monotonic"] = time.monotonic()
                    st.session_state["odds_checked_at_hkt"] = datetime.now(HK_TZ).strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state["odds_server_time"] = server_updated_at
                    st.session_state["odds_last_error"] = None

                    if live_by_race:
                        df_temp = st.session_state['df_data'].copy()
                        row_races = df_temp['賽事編號'].map(extract_race_no)
                        row_horses = df_temp['馬號'].map(normalize_horse_no)
                        updated_rows = 0
                        odds_changed = False

                        for race_no, markets in live_by_race.items():
                            race_mask = row_races == int(race_no)
                            for horse_no, value in markets['WIN'].items():
                                horse_mask = race_mask & (row_horses == horse_no)
                                updated_rows += int(horse_mask.sum())
                                if horse_mask.any():
                                    old = pd.to_numeric(df_temp.loc[horse_mask, '獨贏賠率'], errors='coerce').fillna(-1).to_numpy()
                                    if not np.all(old == float(value)):
                                        odds_changed = True
                                        df_temp.loc[horse_mask, '獨贏賠率'] = value
                            for horse_no, value in markets['PLA'].items():
                                horse_mask = race_mask & (row_horses == horse_no)
                                if horse_mask.any():
                                    old = pd.to_numeric(df_temp.loc[horse_mask, '位置賠率'], errors='coerce').fillna(-1).to_numpy()
                                    if not np.all(old == float(value)):
                                        odds_changed = True
                                        df_temp.loc[horse_mask, '位置賠率'] = value

                        if odds_changed:
                            st.session_state['df_data'] = df_temp
                            st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version', 0) + 1
                        st.session_state['odds_last_status'] = (
                            f"成功取得 {len(live_by_race)} 場賠率及官方彩池狀態；匹配到 CSV 馬匹資料列：{updated_rows}。"
                        )
                    else:
                        st.session_state['odds_last_status'] = "已取得官方彩池狀態；目前沒有可更新的賠率數值。"

                    target_pool = (pool_status_by_race.get(int(target_race)) or {}).get("WIN")
                    if target_pool:
                        current_pool_state = classify_pool_status(target_pool)
                        watch_key = f"{api_date}|{venue_code}|{int(target_race)}|WIN"
                        previous_key = st.session_state.get("pool_watch_key")
                        previous_state = st.session_state.get("pool_watch_state")
                        if previous_key == watch_key and previous_state and previous_state != current_pool_state:
                            st.session_state["pool_transition_notice"] = (
                                f"WIN 受注狀態變更：{previous_state} → {current_pool_state} "
                                f"（{st.session_state['odds_checked_at_hkt']} 香港時間）。"
                            )
                        elif previous_key != watch_key:
                            st.session_state["pool_transition_notice"] = None
                        st.session_state["pool_watch_key"] = watch_key
                        st.session_state["pool_watch_state"] = current_pool_state

        now_monotonic = time.monotonic()
        last_poll = st.session_state.get("odds_last_poll_monotonic", 0.0)
        poll_due = manual_refresh or (
            auto_refresh and now_monotonic - last_poll >= refresh_seconds
        )
        fetch_future = st.session_state.get("odds_fetch_future")
        if poll_due and (fetch_future is None or fetch_future.done()):
            st.session_state["odds_last_poll_monotonic"] = now_monotonic
            st.session_state["odds_fetch_future_source_key"] = poll_source_key
            st.session_state["odds_fetch_future"] = get_odds_executor().submit(
                fetch_live_odds, api_date, venue_code
            )
            fetch_future = st.session_state["odds_fetch_future"]

        if auto_refresh or fetch_future is not None:
            st_autorefresh(interval=1000, key="hkjc_countdown_tick")

        if fetch_future is not None and not fetch_future.done():
            st.caption("正在背景讀取官方賠率／彩池狀態；倒數仍會每秒更新。")

        if st.session_state.get('odds_last_error'):
            st.warning(st.session_state['odds_last_error'])
        elif st.session_state.get('odds_last_status'):
            st.success(st.session_state['odds_last_status'])

        if st.session_state.get('odds_server_time'):
            st.caption(f"馬會彩池資料最後更新時間：{st.session_state['odds_server_time']}")

        status_by_race = st.session_state.get("odds_pool_status_by_race", {})
        target_pool = (status_by_race.get(int(target_race)) or {}).get("WIN")
        last_success = st.session_state.get("odds_last_success_monotonic")
        data_age = (time.monotonic() - last_success) if last_success is not None else None
        fresh_limit = max(10, int(refresh_seconds) * 2)
        raw_pool_state = classify_pool_status(target_pool) if target_pool else "狀態未知"
        pool_state = raw_pool_state if data_age is not None and data_age <= fresh_limit else "狀態未知／資料過期"

        st.markdown("#### 🟢 官方 WIN 受注狀態（唯讀）")
        st.metric("第 {} 場 WIN 彩池".format(int(target_race)), pool_state)
        if target_pool:
            st.caption(
                f"原始 status={target_pool.get('status') or '—'}；"
                f"sellStatus={target_pool.get('sellStatus') or '—'}；"
                f"彩池更新時間={target_pool.get('lastUpdateTime') or '—'}；"
                f"本機查詢={st.session_state.get('odds_checked_at_hkt') or '—'}。"
            )
        if pool_state == "受注中":
            st.success("官方資料目前回報 WIN 彩池開放；仍須以馬會成功受注回執為準。")
        elif pool_state in ("暫停受注", "停止受注／已封盤"):
            st.error("官方資料回報目前暫停或停止受注；此時不會提交任何投注。")
        else:
            st.warning("受注狀態未知或資料過期；請勿把舊狀態當作仍可受注。")

        current_watch_key = f"{api_date}|{venue_code}|{int(target_race)}|WIN"
        if (
            st.session_state.get("pool_transition_notice")
            and st.session_state.get("pool_watch_key") == current_watch_key
        ):
            st.info(st.session_state["pool_transition_notice"])

        if auto_refresh:
            st.caption(
                f"倒數每秒刷新；官方狀態每 {refresh_seconds} 秒最多查詢一次。" 
                "狀態僅供提醒，頁面不會提交投注；請保持此頁開啟。"
            )

        # ==========================================
        st.markdown("---")
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
