import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import joblib
import requests
import re
import time
from datetime import datetime, timedelta, timezone
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components

# 頁面基本設定
st.set_page_config(
    page_title="HKJC AI 智能賽馬預測系統",
    page_icon="🐎",
    layout="wide"
)

st.title("🐎 HKJC 旗艦 15 大特徵 AI 預測系統 (支援即時賠率)")
st.markdown("---")

# 設定 repo 內的模型及預測 CSV 路徑
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / 'my_hkjc_model.pkl'
PREDICTION_CSV_PATH = APP_DIR / 'prediction.csv'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

if model is None:
    st.error(f"⚠️ 找不到 AI 模型檔 `{MODEL_PATH}`！請確認模型是否已上傳至正確目錄。")
    st.stop()

# 預測卡從 GitHub repo 同目錄固定讀取；賽後結果另行上傳作回測
st.sidebar.header("📂 預測資料")
st.sidebar.caption(f"預測 CSV：{PREDICTION_CSV_PATH.name}（由 GitHub repo 讀取）")
backtest_file = st.sidebar.file_uploader(
    "回測用：上傳已完成賽事 CSV（需要賽事編號、馬號、名次）",
    type=['csv'],
    key='backtest_results_upload',
    help='預測賽卡固定從 GitHub repo 讀取；這個上傳欄只用來匯入賽後名次及可選的官方派彩賠率。',
)

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

def normalize_race_id(value):
    """Normalize race ids such as 20261001-1 and 20261001-01 to one key."""
    text = str(value).strip()
    match = re.search(r"(20\d{6}).*?(\d+)\s*$", text)
    if match:
        return f"{match.group(1)}-{int(match.group(2)):02d}"
    return text

def fetch_live_odds(date_str, venue):
    body = {
        "operationName": "racing",
        "variables": {
            "date": date_str,
            "venueCode": venue,
            "raceNo": race_no,
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

if PREDICTION_CSV_PATH.is_file():
    # 預測賽卡直接從已部署的 GitHub repo 讀取，不需每次在 Streamlit 上傳。
    try:
        df_raw = pd.read_csv(PREDICTION_CSV_PATH, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df_raw = pd.read_csv(PREDICTION_CSV_PATH, encoding='cp950')

    if df_raw.empty:
        st.error(f"預測 CSV 是空檔：{PREDICTION_CSV_PATH.name}")
        st.stop()
    st.sidebar.success(f"已從 repo 載入預測卡：{PREDICTION_CSV_PATH.name}（{len(df_raw)} 匹）")
    st.success("✅ 已從 GitHub repo 載入預測賽事資料！")
    
    # 確保賠率欄位存在；沒有實際 PLACE 賠率時以估算值作初始值
    if '獨贏賠率' not in df_raw.columns:
        df_raw['獨贏賠率'] = 10.0
    if '位置賠率' not in df_raw.columns:
        win_values = pd.to_numeric(df_raw['獨贏賠率'], errors='coerce').fillna(10.0)
        df_raw['位置賠率'] = 1.0 + (win_values - 1.0) / 3.2

    # 使用 Session State 保留已抓取的賠率；repo CSV 更新後依檔案修改時間重新載入
    prediction_signature = f"{PREDICTION_CSV_PATH.name}:{PREDICTION_CSV_PATH.stat().st_mtime_ns}:{PREDICTION_CSV_PATH.stat().st_size}"
    if 'df_data' not in st.session_state or st.session_state.get('prediction_signature') != prediction_signature:
        st.session_state['df_data'] = df_raw.copy()
        st.session_state['prediction_signature'] = prediction_signature
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

    # 回測使用獨立上傳的完賽資料：按賽事編號＋馬號將名次、可選官方賠率合併到預測快照。
    df_backtest = df.copy()
    backtest_ready = False
    if backtest_file is not None:
        try:
            try:
                result_raw = pd.read_csv(backtest_file, encoding='utf-8-sig')
            except UnicodeDecodeError:
                backtest_file.seek(0)
                result_raw = pd.read_csv(backtest_file, encoding='cp950')
            required_result_cols = ['賽事編號', '馬號', '名次']
            missing_result_cols = [c for c in required_result_cols if c not in result_raw.columns]
            if missing_result_cols:
                st.sidebar.error(f"回測 CSV 缺少欄位：{missing_result_cols}")
            else:
                result_map = result_raw.copy()
                result_map['_race_key'] = result_map['賽事編號'].map(normalize_race_id)
                result_map['_horse_key'] = result_map['馬號'].map(normalize_horse_no)
                rename_map = {'名次': '_result_rank'}
                if '獨贏賠率' in result_map.columns:
                    rename_map['獨贏賠率'] = '_result_win_odds'
                if '位置賠率' in result_map.columns:
                    rename_map['位置賠率'] = '_result_place_odds'
                result_map = result_map.drop_duplicates(['_race_key', '_horse_key'], keep='last')
                result_map = result_map[['_race_key', '_horse_key'] + [c for c in rename_map if c in result_map.columns]]
                result_map = result_map.rename(columns=rename_map)

                df_backtest['_race_key'] = df_backtest['賽事編號'].map(normalize_race_id)
                df_backtest['_horse_key'] = df_backtest['馬號'].map(normalize_horse_no)
                df_backtest = df_backtest.merge(
                    result_map, on=['_race_key', '_horse_key'], how='left', validate='many_to_one', indicator='_result_match'
                )
                df_backtest['名次'] = df_backtest['_result_rank']
                df_backtest['numeric_rank'] = pd.to_numeric(df_backtest['名次'], errors='coerce').fillna(99)
                if '_result_win_odds' in df_backtest.columns:
                    df_backtest['回測獨贏賠率'] = pd.to_numeric(df_backtest['_result_win_odds'], errors='coerce')
                if '_result_place_odds' in df_backtest.columns:
                    df_backtest['回測位置賠率'] = pd.to_numeric(df_backtest['_result_place_odds'], errors='coerce')
                has_official_result = df_backtest['_result_match'].eq('both') & df_backtest['_result_rank'].notna()
                matched = int(has_official_result.sum())
                backtest_ready = matched > 0
                if backtest_ready:
                    # 排除沒有出現在官方賽果中的取消／退出馬匹；非數字完賽狀態會保留並視為未勝出。
                    df_backtest = df_backtest.loc[has_official_result].copy()
                df_backtest.drop(columns=['_race_key', '_horse_key', '_result_rank', '_result_win_odds', '_result_place_odds', '_result_match'], errors='ignore', inplace=True)
                if backtest_ready:
                    st.sidebar.success(f"回測賽果已配對：{matched} 匹")
                else:
                    st.sidebar.warning("未能按賽事編號＋馬號配對到名次；請檢查兩份 CSV 的識別欄位。")
        except Exception as exc:
            st.sidebar.error(f"讀取回測 CSV 失敗：{type(exc).__name__}: {exc}")

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
            auto_date = datetime.now().strftime("%Y-%m-%d")
        api_date = col_d.text_input("API 查詢日期 (YYYY-MM-DD)", value=auto_date)

        race_series = df_raw['賽事編號'].map(extract_race_no)
        races_available = sorted(int(x) for x in race_series.dropna().unique())
        if races_available:
            target_race = col_r.selectbox("查看場次", races_available)
        else:
            target_race = col_r.number_input("查看場次", min_value=1, max_value=15, value=1, step=1)

        # 2. 加入預定開跑時間輸入框
        target_time = col_t.time_input("預定開跑時間", value=datetime.strptime("14:30", "%H:%M").time())
        
        # 3. 獨立倒數計時器：只在瀏覽器內繪製，不呼叫 HKJC，也不觸發 Streamlit rerun。
        # 賽日使用 API 日期，香港時區固定為 UTC+8。
        try:
            race_date = datetime.strptime(api_date.strip(), "%Y-%m-%d").date()
        except ValueError:
            st.error("API 日期格式必須是 YYYY-MM-DD，請先修正日期再使用計時器或更新賠率。")
            st.stop()

        hk_timezone = timezone(timedelta(hours=8))
        target_datetime = datetime.combine(race_date, target_time).replace(tzinfo=hk_timezone)
        target_iso = target_datetime.isoformat()

        timer_html = r"""
        <div id="race-countdown" style="font: bold 30px monospace; text-align: center; padding: 8px;">
          正在啟動倒數計時器…
        </div>
        <script>
        const targetEpochMs = Date.parse("__TARGET_ISO__");
        const timerElement = document.getElementById("race-countdown");
        const startWallMs = Date.now();
        const startMonoMs = performance.now();

        function updateCountdown() {
          // performance.now() 提供平滑的單調計時；不會每秒重新查賠率。
          const elapsedMs = performance.now() - startMonoMs;
          const remainingMs = targetEpochMs - (startWallMs + elapsedMs);

          if (remainingMs > 0) {
            const remainingUs = Math.ceil(remainingMs * 1000);
            const totalSeconds = Math.floor(remainingUs / 1000000);
            const microseconds = remainingUs % 1000000;
            const hours = Math.floor(totalSeconds / 3600);
            const minutes = Math.floor((totalSeconds % 3600) / 60);
            const seconds = totalSeconds % 60;

            timerElement.style.color = "#1769aa";
            timerElement.textContent =
              `${String(hours).padStart(2, "0")}:` +
              `${String(minutes).padStart(2, "0")}:` +
              `${String(seconds).padStart(2, "0")}.` +
              `${String(microseconds).padStart(6, "0")}`;
            requestAnimationFrame(updateCountdown);
          } else if (remainingMs > -300000) {
            timerElement.style.color = "#b26a00";
            timerElement.textContent = "已到預定開跑時間（賽事可能延遲，請留意官方資訊）";
            requestAnimationFrame(updateCountdown);
          } else {
            timerElement.style.color = "#b00020";
            timerElement.textContent = "賽事可能已開跑或結束";
          }
        }

        requestAnimationFrame(updateCountdown);
        </script>
        """.replace("__TARGET_ISO__", target_iso)

        components.html(timer_html, height=70, scrolling=False)

        # 官方賽日收音機：嵌入 HKJC 網頁播放器，不抓取或轉播音訊串流。
        with st.expander("🔊 HKJC 官方賽日收音機（現場評述）", expanded=False):
            st.caption("選擇官方頁面上的語言並按播放。瀏覽器不允許網頁自動播放時，請手動按播放鍵。")
            st.markdown("[在新分頁開啟 HKJC 官方賽日收音機](https://racing.hkjc.com/zh-hk/showcase/live)")
            components.iframe(
                "https://racing.hkjc.com/zh-hk/showcase/live",
                height=620,
                scrolling=True,
            )

        # 下方原本的更新按鈕保持不變
        auto_col, interval_col, button_col = st.columns([1.5, 1.5, 2.5])
        auto_refresh = auto_col.checkbox("自動更新全部場次", value=False)
        refresh_seconds = interval_col.selectbox(
            "更新間隔（秒）", [10,30, 60, 120, 300], index=1, disabled=not auto_refresh
        )
        manual_refresh = button_col.button(
            "🔄 立即更新全部場次賠率", use_container_width=True
        )

        if auto_refresh:
            st_autorefresh(
                interval=int(refresh_seconds * 1000),
                key="hkjc_live_odds_autorefresh",
            )

        # 執行賠率抓取邏輯
        if auto_refresh or manual_refresh:
            with st.spinner(f"正在讀取第 {target_race} 場即時賠率…"):
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
                
                # 強制頁面重新載入，讓上方的表格與下方的 AI 推薦瞬間更新
                # st.rerun()
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
        if not backtest_ready:
            st.warning("請在左側上傳已完賽的結果 CSV。需有『賽事編號』、『馬號』及『名次』；官方獨贏／位置賠率欄位可選。")
        else:
            sub_tab1, sub_tab2, sub_tab3 = st.tabs(["🥇 獨贏 (Win)", "🥈 位置 (Place)", "🔗 位置Q (QP)"])
            BET_AMOUNT = 100 
            
            with sub_tab1:
                st.markdown("#### 🥇 獨贏 (Win) 策略回測")
                win_invested, win_return, win_bets, win_hits = 0, 0, 0, 0
                win_records = []
                for race_id, group in df_backtest.groupby('賽事編號'):
                    filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
                    sorted_group = filtered_group.sort_values(by='ev', ascending=False).reset_index(drop=True)
                    if len(sorted_group) > 0:
                        pick = sorted_group.iloc[0]
                        win_bets += 1
                        win_invested += BET_AMOUNT
                        is_hit = (pick['numeric_rank'] == 1)
                        settlement_win_odds = pick.get('回測獨贏賠率', np.nan)
                        if pd.isna(settlement_win_odds) or settlement_win_odds <= 1:
                            settlement_win_odds = pick['獨贏賠率']
                        if is_hit:
                            win_hits += 1
                            payout = BET_AMOUNT * settlement_win_odds
                            win_return += payout
                            result_str = "✅ 贏"
                        else:
                            payout = 0
                            result_str = "❌ 輸"
                        win_records.append({
                            '賽事編號': race_id,
                            '投注馬號': f"{pick['馬號']} ({pick['馬名']})",
                            '實際名次': str(pick['名次']).replace('.0', ''),
                            '獨贏賠率 (派彩用)': settlement_win_odds,
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
                for race_id, group in df_backtest.groupby('賽事編號'):
                    filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
                    sorted_group = filtered_group.sort_values(by='ev', ascending=False).reset_index(drop=True)
                    if len(sorted_group) > 0:
                        pick = sorted_group.iloc[0]
                        place_bets += 1
                        place_invested += BET_AMOUNT
                        is_hit = (pick['numeric_rank'] <= 3)
                        settlement_place_odds = pick.get('回測位置賠率', np.nan)
                        if pd.isna(settlement_place_odds) or settlement_place_odds <= 1:
                            settlement_place_odds = pick['位置賠率']
                        p_odds = float(settlement_place_odds) if pd.notna(settlement_place_odds) and float(settlement_place_odds) > 1.0 else 1.5
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
                for race_id, group in df_backtest.groupby('賽事編號'):
                    filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
                    sorted_group = filtered_group.sort_values(by='ev', ascending=False).reset_index(drop=True)
                    if len(sorted_group) >= 2:
                        top1 = sorted_group.iloc[0]
                        top2 = sorted_group.iloc[1]
                        qp_bets += 1
                        qp_invested += BET_AMOUNT
                        is_hit = (top1['numeric_rank'] <= 3) and (top2['numeric_rank'] <= 3)
                        p1_settlement = top1.get('回測位置賠率', np.nan)
                        p2_settlement = top2.get('回測位置賠率', np.nan)
                        p1_odds = float(p1_settlement if pd.notna(p1_settlement) and p1_settlement > 1.0 else top1['位置賠率'])
                        p2_odds = float(p2_settlement if pd.notna(p2_settlement) and p2_settlement > 1.0 else top2['位置賠率'])
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
    st.error(f"找不到 repo 預測 CSV：{PREDICTION_CSV_PATH.name}。請把 futurecard CSV 放到 app.py 同一個 GitHub 資料夾，並命名為 prediction.csv。")
