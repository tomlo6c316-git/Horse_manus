import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import joblib
import requests
import re
import time
import hashlib
from io import StringIO
from io import BytesIO
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse
from bs4 import BeautifulSoup
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components

# 頁面基本設定
st.set_page_config(
    page_title="HKJC AI 智能賽馬預測系統",
    page_icon="🐎",
    layout="wide"
)

st.title("🐎 HKJC 旗艦 15 大特徵 AI 預測系統 (雙渦輪模型引擎版)")
st.markdown("---")

# 初始化模擬投注紀錄與單場凍結名單
if 'simulated_bets' not in st.session_state:
    st.session_state['simulated_bets'] = pd.DataFrame(columns=[
        '下注時間', '賽事編號', '場次', '馬號', '馬名', '玩法', '買入賠率', '注碼'
    ])
if 'frozen_races' not in st.session_state:
    st.session_state['frozen_races'] = {} 

# 設定 repo 內的模型及預測 CSV 路徑
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / 'my_hkjc_model.pkl'
PLACE_MODEL_PATH = APP_DIR / 'my_hkjc_place_model.pkl' # 🎯 新增位置模型路徑

FEATURE_COLS_15 = [
    'market_implied_prob', '獨贏賠率', 'odds_rank', 'is_favorite',
    '排位檔位', 'weight_diff', 'weight_rank',
    'jockey_win_rate', 'trainer_win_rate', 'combo_win_rate',
    'horse_win_rate', 'horse_last_rank',
    '距離', 'horse_surface_win_rate', 'horse_dist_win_rate',
]

@st.cache_resource
def load_models():
    m_win = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    m_pla = joblib.load(PLACE_MODEL_PATH) if os.path.exists(PLACE_MODEL_PATH) else None
    return m_win, m_pla

model_win, model_place = load_models()

if model_win is None:
    st.error(f"⚠️ 找不到獨贏模型檔 `{MODEL_PATH}`！請確認模型是否已上傳。")
    st.stop()
if model_place is None:
    st.warning("⚠️ 尚未偵測到位置專屬模型 `my_hkjc_place_model.pkl`，位置相關彩池將暫時使用獨贏模型代算。")

@st.cache_data
def load_historical_stats(history_csv_path='hkjc_all_seasons_features.csv'):
    csv_path = Path(__file__).resolve().parent / history_csv_path
    if not os.path.exists(csv_path): return None
    try:
        hist_df = pd.read_csv(csv_path, low_memory=False)
        hist_df['numeric_rank'] = pd.to_numeric(hist_df['名次'], errors='coerce')
        hist_df['is_win'] = (hist_df['numeric_rank'] == 1).astype(int)
        
        hist_df['_clean_jockey'] = hist_df['騎師'].astype(str).str.replace(r'\(.*?\)', '', regex=True).str.replace(' ', '')
        hist_df['_clean_trainer'] = hist_df['練馬師'].astype(str).str.replace(' ', '')
        hist_df['_clean_horse'] = hist_df['馬名'].astype(str).str.replace(r'\(.*?\)', '', regex=True).str.replace(' ', '')
        
        jockey_stats = hist_df.groupby('_clean_jockey', as_index=False)['is_win'].mean().rename(columns={'is_win': 'hist_jockey_win_rate'})
        trainer_stats = hist_df.groupby('_clean_trainer', as_index=False)['is_win'].mean().rename(columns={'is_win': 'hist_trainer_win_rate'})
        horse_stats = hist_df.groupby('_clean_horse', as_index=False)['is_win'].mean().rename(columns={'is_win': 'hist_horse_win_rate'})
        combo_stats = hist_df.groupby(['_clean_jockey', '_clean_trainer'], as_index=False)['is_win'].mean().rename(columns={'is_win': 'hist_combo_win_rate'})
        
        valid_ranks_df = hist_df.dropna(subset=['numeric_rank']).copy()
        if '賽事編號' in valid_ranks_df.columns:
            valid_ranks_df = valid_ranks_df.sort_values('賽事編號')
        horse_last_rank_stats = (
            valid_ranks_df.groupby('_clean_horse', as_index=False)
            .last()[['_clean_horse', 'numeric_rank']]
            .rename(columns={'numeric_rank': 'hist_horse_last_rank'})
        )
        if '場地' in hist_df.columns:
            hist_df['_clean_surface'] = hist_df['場地'].astype(str).str.strip()
            horse_surface_stats = hist_df.groupby(['_clean_horse', '_clean_surface'], as_index=False)['is_win'].mean().rename(columns={'is_win': 'hist_horse_surface_win_rate'})
        else: horse_surface_stats = pd.DataFrame(columns=['_clean_horse', '_clean_surface', 'hist_horse_surface_win_rate'])
        if '距離' in hist_df.columns:
            hist_df['_clean_dist'] = pd.to_numeric(hist_df['距離'], errors='coerce').fillna(1200).astype(int)
            horse_dist_stats = hist_df.groupby(['_clean_horse', '_clean_dist'], as_index=False)['is_win'].mean().rename(columns={'is_win': 'hist_horse_dist_win_rate'})
        else: horse_dist_stats = pd.DataFrame(columns=['_clean_horse', '_clean_dist', 'hist_horse_dist_win_rate'])
        return (jockey_stats, trainer_stats, horse_stats, combo_stats, horse_last_rank_stats, horse_surface_stats, horse_dist_stats)
    except Exception as e:
        st.error(f"載入歷史數據失敗：{e}")
        return None

def find_racecard_table(html_text):
    try: tables = pd.read_html(StringIO(html_text))
    except Exception: return None
    for table in tables:
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = ['_'.join(map(str, col)) for col in table.columns]
        table.columns = [str(col).replace('\n', ' ').strip() for col in table.columns]
        header_text = ' '.join(table.columns)
        if '馬名' in header_text and '騎師' in header_text: return table
    return None

def find_column(columns, keywords, fallback=None):
    return next((col for col in columns if any(word in str(col) for word in keywords)), fallback)

def fetch_hkjc_racecard(race_date):
    date_text = race_date.strftime('%Y/%m/%d')
    race_day_id = race_date.strftime('%Y%m%d')
    season_start = race_date.year if race_date.month >= 9 else race_date.year - 1
    season = f"{season_start % 100:02d}/{(season_start + 1) % 100:02d}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    session = requests.Session()
    rows = []
    race_venue = None
    first_race = None
    for venue_code in ('ST', 'HV'):
        url = f'https://racing.hkjc.com/racing/information/Chinese/Racing/RaceCard.aspx?racedate={date_text}&Racecourse={venue_code}&RaceNo=1'
        response = session.get(url, headers=headers, timeout=12)
        response.encoding = 'utf-8'
        first_race_table = find_racecard_table(response.text)
        if first_race_table is not None:
            race_venue = venue_code
            first_race = (first_race_table, response.text)
            break
    if first_race is None: raise ValueError('HKJC 暫未提供該日期的賽卡。')

    for race_no in range(1, 16):
        if race_no == 1: race_table, race_html = first_race
        else:
            url = f'https://racing.hkjc.com/racing/information/Chinese/Racing/RaceCard.aspx?racedate={date_text}&Racecourse={race_venue}&RaceNo={race_no}'
            response = session.get(url, headers=headers, timeout=12)
            response.encoding = 'utf-8'
            race_html = response.text
            race_table = find_racecard_table(race_html)
            if race_table is None: break
        distance_match = re.search(r'(\d{4})\s*米', race_html)
        distance = int(distance_match.group(1)) if distance_match else 1200
        surface = '泥地' if ('泥地' in race_html or '全天候' in race_html) else '草地'
        columns = list(race_table.columns)
        horse_no_col = find_column(columns, ('馬號', '編號'), columns[0] if columns else None)
        horse_name_col = find_column(columns, ('馬名',))
        weight_col = find_column(columns, ('負磅', '磅'))
        jockey_col = find_column(columns, ('騎師',))
        trainer_col = find_column(columns, ('練馬師',))
        draw_col = find_column(columns, ('檔位',))
        if not all((horse_no_col, horse_name_col, weight_col, jockey_col, trainer_col, draw_col)): continue

        for _, row in race_table.iterrows():
            horse_no = str(row[horse_no_col]).strip()
            if re.fullmatch(r'\d+\.0', horse_no): horse_no = horse_no[:-2]
            if not horse_no.isdigit(): continue
            horse_name = re.sub(r'\(.*?\)', '', str(row[horse_name_col])).strip()
            if not horse_name or horse_name.lower() == 'nan': continue
            rows.append({
                '馬季': season, '賽事編號': f'{race_day_id}-{race_no:02d}', '名次': '', '馬號': horse_no, '馬名': horse_name,
                '騎師': str(row[jockey_col]).strip(), '練馬師': str(row[trainer_col]).strip(),
                '實際負磅': str(row[weight_col]).strip(), '排位檔位': str(row[draw_col]).strip(),
                '獨贏賠率': 10.0, '位置賠率': 1.5, '距離': distance, '場地': surface, 'racecourse_code': race_venue,
            })
        time.sleep(0.25)
    if not rows: raise ValueError('HKJC 沒有回傳排位表。')
    return pd.DataFrame(rows).drop_duplicates(['賽事編號', '馬號'], keep='last').reset_index(drop=True)

# ==========================================
# 📂 預測資料與側邊欄設定
# ==========================================
st.sidebar.header("📂 預測資料")
available_csvs = sorted([f.name for f in APP_DIR.glob("prediction*.csv")])
selected_csv_name = st.sidebar.selectbox("📜 選擇預測賽卡", available_csvs) if available_csvs else "prediction.csv"

if st.session_state.get('active_csv_name') != selected_csv_name:
    st.session_state['active_csv_name'] = selected_csv_name
    st.session_state.pop('fetched_prediction_df', None)
    st.session_state.pop('df_data', None)
    st.session_state.pop('prediction_signature', None)
    st.session_state['frozen_races'] = {}
    st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version', 0) + 1

SELECTED_CSV_PATH = APP_DIR / selected_csv_name

race_date_choice = st.sidebar.date_input('選擇賽事日期 (抓新排位用)', value=datetime.now().date())
if st.sidebar.button('🏇 抓取排位並載入預測', use_container_width=True):
    try:
        with st.spinner(f'正在抓取 HKJC {race_date_choice:%Y-%m-%d} 排位表…'):
            fetched_card = fetch_hkjc_racecard(race_date_choice)
        st.session_state['fetched_prediction_df'] = fetched_card
        st.session_state['fetched_prediction_signature'] = f"hkjc:{race_date_choice:%Y%m%d}:{len(fetched_card)}"
        st.session_state['fetched_prediction_date'] = race_date_choice
        st.session_state['df_data'] = fetched_card.copy()
        st.session_state['prediction_signature'] = st.session_state['fetched_prediction_signature']
        st.session_state['pending_auto_odds_fetch'] = True
        st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version', 0) + 1
        st.session_state['frozen_races'] = {}
        st.sidebar.success(f'已載入 {len(fetched_card)} 匹馬。')
    except Exception as exc: st.sidebar.error(f'抓取排位失敗：{exc}')

current_fetched_card = st.session_state.get('fetched_prediction_df')
if current_fetched_card is not None:
    if st.sidebar.button('🔄 放棄抓取資料，改用上方選單 CSV', use_container_width=True):
        st.session_state.pop('fetched_prediction_df', None)
        st.session_state.pop('df_data', None)
        st.session_state.pop('prediction_signature', None)
        st.session_state['frozen_races'] = {}
        st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version', 0) + 1
        st.rerun()

backtest_file = st.sidebar.file_uploader("回測用：上傳已完成賽事 CSV", type=['csv'], key='backtest_results_upload')

st.sidebar.markdown("---")
st.sidebar.header("⚙ 投注策略與算分引擎")
strategy_mode = st.sidebar.radio("1️⃣ 推薦排序核心邏輯", ["⚡ 傳統純 EV 排序 (高賠率優先)", "🛡️ 抗噪穩定 (基本面優先 + 排除回飛)", "🎯 純 AI 勝率最高 (命中率優先)"], index=0)
use_dynamic_form = st.sidebar.checkbox("2️⃣ 將「上仗名次與同程勝率」納入 AI 算分", value=False)
min_ev = st.sidebar.slider("最小期望值 (EV 門檻)", 0.0, 6.0, 0.0, 0.1)
min_odds = st.sidebar.number_input("最低獨贏賠率", min_value=1.0, max_value=50.0, value=3.0)
max_odds = st.sidebar.number_input("最高獨贏賠率", min_value=1.0, max_value=100.0, value=20.0)

st.sidebar.markdown("---")
st.sidebar.header("💰 資金管理 (凱利注碼精算)")
total_bankroll = st.sidebar.number_input("💵 單日總本金預算 (HKD)", min_value=100, max_value=1000000, value=5000, step=500)
kelly_fraction = st.sidebar.slider("🎛️ 凱利保守係數", 0.05, 1.0, 0.15, 0.05)

# ==========================================
# 🛠️ 歷史賽果抓取器
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("🛠️ 歷史賽果抓取器")
tool_date_choice = st.sidebar.date_input("選擇目標賽事日期", value=datetime.now().date(), key="tool_date_input")
tool_season = st.sidebar.text_input("輸入馬季", value="26/27")

if st.sidebar.button("📥 抓取並產出完賽 CSV", use_container_width=True, key="run_tool_button"):
    date_str_tool = tool_date_choice.strftime('%Y/%m/%d')
    date_for_id_tool = tool_date_choice.strftime('%Y%m%d')
    all_races_data_tool = []
    headers_tool = {'User-Agent': 'Mozilla/5.0'}
    progress_bar = st.sidebar.progress(0)
    for race_no in range(1, 13):
        url = f"https://racing.hkjc.com/racing/information/Chinese/Racing/LocalResults.aspx?RaceDate={date_str_tool}&RaceNo={race_no}"
        try:
            resp = requests.get(url, headers=headers_tool, timeout=15)
            resp.encoding = 'utf-8' 
            html_text = resp.text
            if "沒有相關資料" in html_text: continue
            dist_match = re.search(r'(\d{4})\s*米', html_text)
            distance = int(dist_match.group(1)) if dist_match else 1200
            surface_type = "草地" if "草地" in html_text else "泥地"
            tables = pd.read_html(StringIO(html_text))
            target_df = None
            for t in tables:
                if isinstance(t.columns, pd.MultiIndex): t.columns = ['_'.join(map(str, col)) for col in t.columns]
                check_str = "".join([str(c) for c in t.columns])
                for row_idx in range(min(5, len(t))): check_str += "".join([str(c) for c in t.iloc[row_idx].values])
                if '名次' in check_str and '馬號' in check_str and '獨贏' in check_str:
                    if not any('名次' in str(c) for c in t.columns):
                        for row_idx in range(min(5, len(t))):
                            row_str = "".join([str(c) for c in t.iloc[row_idx].values])
                            if '名次' in row_str and '馬號' in row_str:
                                t.columns = [str(c) for c in t.iloc[row_idx]]
                                t = t.drop(range(row_idx + 1)).reset_index(drop=True) 
                                break
                    target_df = t
                    break
            if target_df is not None and not target_df.empty:
                cols = target_df.columns.astype(str)
                rank_col, horse_no_col, odds_col = next((c for c in cols if '名次' in c), None), next((c for c in cols if '馬號' in c or '編號' in c), None), next((c for c in cols if '獨贏' in c), None)
                if rank_col and horse_no_col:
                    for idx, row in target_df.iterrows():
                        horse_no = str(row.get(horse_no_col, '')).strip()
                        if horse_no.endswith('.0'): horse_no = horse_no[:-2]
                        if horse_no.isdigit():
                            all_races_data_tool.append({
                                '馬季': tool_season, '賽事編號': f"{date_for_id_tool}-{race_no:02d}", '名次': str(row.get(rank_col, '')).strip(),
                                '馬號': horse_no, '馬名': str(row.get(next((c for c in cols if '馬名' in c), ''), '')).split('(')[0].strip(),
                                '騎師': str(row.get(next((c for c in cols if '騎師' in c), ''), '')).strip(), '練馬師': str(row.get(next((c for c in cols if '練馬師' in c), ''), '')).strip(),
                                '實際負磅': str(row.get(next((c for c in cols if '負磅' in c or '實際負磅' in c), ''), '')).strip(),
                                '排位檔位': str(row[next((c for c in cols if '檔位' in c), cols[0])]).strip() if any('檔位' in c for c in cols) else '7',
                                '獨贏賠率': row[odds_col] if odds_col and pd.notna(row[odds_col]) else 0.0, '距離': distance, '場地': surface_type,
                                'horse_surface_win_rate': 0.08, 'horse_dist_win_rate': 0.08
                            })
            progress_bar.progress(race_no / 12)
            time.sleep(1.0)
        except Exception: continue
    progress_bar.empty()
    if all_races_data_tool:
        df_tool = pd.DataFrame(all_races_data_tool)
        csv_data = df_tool.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.sidebar.success("🎉 抓取成功！")
        st.sidebar.download_button("⬇️ 下載完賽 CSV", data=csv_data, file_name=f"hkjc_data_15features_{date_for_id_tool}.csv", mime="text/csv", use_container_width=True)

# ==========================================
# HKJC 完整版 GraphQL 賠率抓取 (保證不報錯)
# ==========================================
GRAPHQL_URL = "https://info.cld.hkjc.com/graphql/base/"
ODDS_QUERY = """query racing($date: String,$venueCode: String, $oddsTypes: [OddsType],$raceNo: Int) {
  raceMeetings(date: $date, venueCode:$venueCode) {
    pmPools(oddsTypes: $oddsTypes, raceNo:$raceNo) {
      id status sellStatus oddsType lastUpdateTime guarantee minTicketCost name_en name_ch
      leg { number races }
      cWinSelections { composite name_ch name_en starters }
      oddsNodes { combString oddsValue hotFavourite oddsDropValue bankerOdds { combString oddsValue } }
    }
  }
}"""
GRAPHQL_HEADERS = { "Accept": "*/*", "Content-Type": "application/json", "Origin": "https://bet.hkjc.com", "Referer": "https://bet.hkjc.com/", "User-Agent": "Mozilla/5.0"}

def normalize_horse_no(value):
    try: return str(int(float(str(value).strip())))
    except: return str(value).strip()

def extract_race_no(value):
    match = re.search(r"-(\d+)\s*$", str(value).strip())
    return int(match.group(1)) if match else None

def normalize_race_id(value):
    match = re.search(r"(20\d{6}).*?(\d+)\s*$", str(value).strip())
    return f"{match.group(1)}-{int(match.group(2)):02d}" if match else str(value).strip()

def fetch_live_odds(date_str, venue):
    try:
        resp = requests.post(GRAPHQL_URL, headers=GRAPHQL_HEADERS, json={"operationName": "racing", "variables": {"date": date_str, "venueCode": venue, "raceNo": None, "oddsTypes": ["WIN", "PLA"]}, "query": ODDS_QUERY}, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("errors"): return None, None, f"GraphQL 伺服器拒絕：{payload['errors']}"
        meetings = (payload.get("data") or {}).get("raceMeetings") or []
        if not meetings: return None, None, "無資料（可能尚未開盤或日期錯誤）"
        odds_by_race, timestamps = {}, []
        for pool in meetings[0].get("pmPools") or []:
            pool_type = str(pool.get("oddsType", "")).upper()
            if pool_type not in ("WIN", "PLA"): continue
            if pool.get("lastUpdateTime"): timestamps.append(pool["lastUpdateTime"])
            for race_number in (pool.get("leg") or {}).get("races") or []:
                market = odds_by_race.setdefault(int(race_number), {"WIN": {}, "PLA": {}})
                for node in pool.get("oddsNodes") or []:
                    horse_no = normalize_horse_no(node.get("combString"))
                    odds_val = str(node.get("oddsValue")).strip()
                    if horse_no and odds_val not in ("None", "", "SCR"):
                        try: market[pool_type][horse_no] = float(odds_val)
                        except: pass
        return odds_by_race, (max(timestamps) if timestamps else None), None
    except Exception as exc: return None, None, f"連線錯誤：{exc}"

# ==========================================
# 載入並計算 15 大特徵與雙渦輪引擎分數
# ==========================================
fetched_prediction_df = st.session_state.get('fetched_prediction_df')
hk_timezone = timezone(timedelta(hours=8))

if fetched_prediction_df is not None:
    df_raw = fetched_prediction_df.copy()
    prediction_signature = st.session_state.get('fetched_prediction_signature', "HKJC_fetch")
elif SELECTED_CSV_PATH.is_file():
    try: df_raw = pd.read_csv(SELECTED_CSV_PATH, encoding='utf-8-sig')
    except: df_raw = pd.read_csv(SELECTED_CSV_PATH, encoding='cp950')
    prediction_signature = f"{SELECTED_CSV_PATH.name}:{SELECTED_CSV_PATH.stat().st_mtime_ns}"
    st.sidebar.success(f"✅ 載入：`{SELECTED_CSV_PATH.name}` ({len(df_raw)}匹)")
else: st.stop()

if '獨贏賠率' not in df_raw.columns: df_raw['獨贏賠率'] = 10.0
if '位置賠率' not in df_raw.columns: df_raw['位置賠率'] = 1.0 + (pd.to_numeric(df_raw['獨贏賠率'], errors='coerce').fillna(10.0) - 1.0) / 3.2

if 'df_data' not in st.session_state or st.session_state.get('prediction_signature') != prediction_signature:
    st.session_state['df_data'] = df_raw.copy()
    st.session_state['prediction_signature'] = prediction_signature
    st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version', 0) + 1
    st.session_state['baseline_odds_map'] = {f"{normalize_race_id(r['賽事編號'])}_{normalize_horse_no(r['馬號'])}": float(pd.to_numeric(r['獨贏賠率'], errors='coerce')) for _,r in df_raw.iterrows()}
    st.session_state['baseline_locked_time'] = "初始載入值"

df_editable = st.session_state['df_data'][['賽事編號', '馬號', '馬名', '排位檔位', '獨贏賠率', '位置賠率']].copy()
edited_df = st.data_editor(
    df_editable, disabled=['賽事編號', '馬號', '馬名', '排位檔位'], 
    use_container_width=True, hide_index=True, key=f"odds_editor_{st.session_state['odds_editor_version']}"
)
st.session_state['df_data']['獨贏賠率'] = pd.to_numeric(edited_df['獨贏賠率'], errors='coerce').fillna(10.0)
st.session_state['df_data']['位置賠率'] = pd.to_numeric(edited_df['位置賠率'], errors='coerce').fillna(1.5)

df = st.session_state['df_data'].copy()
df['_horse_uid'] = df.apply(lambda r: f"{normalize_race_id(r['賽事編號'])}_{normalize_horse_no(r['馬號'])}", axis=1)
df['基準賠率'] = pd.to_numeric(df['_horse_uid'].map(st.session_state.get('baseline_odds_map', {})).fillna(df['獨贏賠率']), errors='coerce')
df['odds_drop_pct'] = ((df['基準賠率'] - df['獨贏賠率']) / df['基準賠率'].replace(0, np.nan)).fillna(0.0) * 100.0
df['資金流向'] = df['odds_drop_pct'].apply(lambda pct: f"🔥 大落飛 (-{pct:.0f}%)" if pct >= 20 else (f"📉 落飛 (-{pct:.0f}%)" if pct >= 10 else (f"🥶 回飛 (+{abs(pct):.0f}%)" if pct <= -20 else "➖ 穩定")))

# 特徵補全
df['market_prob'] = 1 / df['獨贏賠率']
df['market_implied_prob'] = df['market_prob'] / df.groupby('賽事編號')['market_prob'].transform('sum')
df['odds_rank'] = df.groupby('賽事編號')['獨贏賠率'].rank(method='min')
df['is_favorite'] = (df['odds_rank'] == 1).astype(int)
df['排位檔位'] = pd.to_numeric(df['排位檔位'], errors='coerce').fillna(7)
df['實際負磅'] = pd.to_numeric(df.get('實際負磅', 120), errors='coerce').fillna(120)
df['weight_diff'] = df['實際負磅'] - df.groupby('賽事編號')['實際負磅'].transform('mean')
df['weight_rank'] = df.groupby('賽事編號')['實際負磅'].rank(ascending=False, method='min')
df['距離'] = pd.to_numeric(df.get('距離', 1200), errors='coerce').fillna(1200).astype(int)
df['場地'] = df.get('場地', '草地')
df['numeric_rank'] = pd.to_numeric(df.get('名次', 99), errors='coerce').fillna(99)

hist_stats = load_historical_stats()
if hist_stats is not None:
    jockey_stats, trainer_stats, horse_stats, combo_stats, hr_stats, sur_stats, dist_stats = hist_stats
    df['_clean_jockey'] = df['騎師'].astype(str).str.replace(r'\(.*?\)', '', regex=True).str.replace(' ', '')
    df['_clean_trainer'] = df['練馬師'].astype(str).str.replace(' ', '')
    df['_clean_horse'] = df['馬名'].astype(str).str.replace(r'\(.*?\)', '', regex=True).str.replace(' ', '')
    df['_clean_surface'] = df['場地'].astype(str).str.strip()
    df['_clean_dist'] = df['距離'].astype(int)
    
    df = df.merge(jockey_stats, on='_clean_jockey', how='left').merge(trainer_stats, on='_clean_trainer', how='left')
    df = df.merge(horse_stats, on='_clean_horse', how='left').merge(combo_stats, on=['_clean_jockey', '_clean_trainer'], how='left')
    df = df.merge(hr_stats, on='_clean_horse', how='left').merge(sur_stats, on=['_clean_horse', '_clean_surface'], how='left').merge(dist_stats, on=['_clean_horse', '_clean_dist'], how='left')
    
    df['jockey_win_rate'] = df['hist_jockey_win_rate'].fillna(0.08)
    df['trainer_win_rate'] = df['hist_trainer_win_rate'].fillna(0.08)
    df['horse_win_rate'] = df['hist_horse_win_rate'].fillna(0.05)
    df['combo_win_rate'] = df['hist_combo_win_rate'].fillna(0.05)
    df['real_last_rank'] = df['hist_horse_last_rank'].fillna(6.5)
    
    if use_dynamic_form:
        df['horse_last_rank'] = df['real_last_rank']
        df['horse_surface_win_rate'] = df['hist_horse_surface_win_rate'].fillna(df['horse_win_rate']).fillna(0.06)
        df['horse_dist_win_rate'] = df['hist_horse_dist_win_rate'].fillna(df['horse_win_rate']).fillna(0.06)
    else:
        df['horse_last_rank'] = 6.0
        df['horse_surface_win_rate'] = 0.08
        df['horse_dist_win_rate'] = 0.08
else:
    for c, v in [('jockey_win_rate',0.08),('trainer_win_rate',0.08),('combo_win_rate',0.05),('horse_win_rate',0.05),('horse_last_rank',6.0),('horse_surface_win_rate',0.08),('horse_dist_win_rate',0.08),('real_last_rank',6.5)]:
        df[c] = df.get(c, v)

# 🚀 雙引擎同時預測 (Win + Place)
X_predict = df[FEATURE_COLS_15].fillna(0)
df['pred_win_prob'] = model_win.predict_proba(X_predict)[:, 1]

if model_place is not None:
    df['pred_place_prob'] = model_place.predict_proba(X_predict)[:, 1]
else:
    df['pred_place_prob'] = df['pred_win_prob'] # 若無位置模型則退回使用獨贏模型

df['win_ev'] = df['pred_win_prob'] * df['獨贏賠率']
df['place_ev'] = df['pred_place_prob'] * df['位置賠率']

drop_bonus = np.clip(df['odds_drop_pct'] / 100.0, -0.35, 0.30)
df['smart_score_win'] = df['pred_win_prob'] * (1.0 + drop_bonus)
df['smart_score_place'] = df['pred_place_prob'] * (1.0 + drop_bonus)

# 💰 雙重凱利注碼精算
safe_win_odds = np.maximum(df['獨贏賠率'], 1.01)
df['kelly_pct_win'] = np.clip((df['win_ev'] - 1.0) / (safe_win_odds - 1.0), 0.0, 1.0)
df['suggested_stake_win'] = np.floor((total_bankroll * df['kelly_pct_win'] * kelly_fraction) / 10) * 10

safe_place_odds = np.maximum(df['位置賠率'], 1.01)
df['kelly_pct_place'] = np.clip((df['place_ev'] - 1.0) / (safe_place_odds - 1.0), 0.0, 1.0)
df['suggested_stake_place'] = np.floor((total_bankroll * df['kelly_pct_place'] * kelly_fraction) / 10) * 10

# 🚀 核心雙軌過濾邏輯
def filter_and_sort_group(group, mode, min_ev_val, min_odds_val, max_odds_val, bet_type="win"):
    if bet_type == "win":
        cond = (group['win_ev'] >= min_ev_val) & (group['獨贏賠率'] >= min_odds_val) & (group['獨贏賠率'] <= max_odds_val)
        sort_col = 'smart_score_win' if mode.startswith("🛡️") else ('pred_win_prob' if mode.startswith("🎯") else 'win_ev')
    else:
        # 位置賠率較低，賠率門檻等比例除以3，確保不會把位置好馬濾掉
        cond = (group['place_ev'] >= min_ev_val) & (group['位置賠率'] >= max(1.01, min_odds_val / 3.0))
        sort_col = 'smart_score_place' if mode.startswith("🛡️") else ('pred_place_prob' if mode.startswith("🎯") else 'place_ev')
        
    if mode.startswith("🛡️"): cond = cond & (group['odds_drop_pct'] > -25.0)
    return group[cond].sort_values(by=sort_col, ascending=False).reset_index(drop=True)

# 準備回測資料
df_backtest = df.copy()
backtest_ready = False
if backtest_file is not None:
    try:
        try: result_raw = pd.read_csv(backtest_file, encoding='utf-8-sig')
        except:
            backtest_file.seek(0)
            result_raw = pd.read_csv(backtest_file, encoding='cp950')
        result_map = result_raw.copy()
        result_map['_race_key'] = result_map['賽事編號'].map(normalize_race_id)
        result_map['_horse_key'] = result_map['馬號'].map(normalize_horse_no)
        rename_map = {'名次': '_result_rank'}
        if '獨贏賠率' in result_map.columns: rename_map['獨贏賠率'] = '_result_win_odds'
        if '位置賠率' in result_map.columns: rename_map['位置賠率'] = '_result_place_odds'
        result_map = result_map.drop_duplicates(['_race_key', '_horse_key'], keep='last').rename(columns=rename_map)

        df_backtest['_race_key'] = df_backtest['賽事編號'].map(normalize_race_id)
        df_backtest['_horse_key'] = df_backtest['馬號'].map(normalize_horse_no)
        df_backtest = df_backtest.merge(result_map[['_race_key', '_horse_key'] + [c for c in rename_map.values() if c in result_map.columns]], on=['_race_key', '_horse_key'], how='left')
        df_backtest['名次'] = df_backtest['_result_rank']
        df_backtest['numeric_rank'] = pd.to_numeric(df_backtest['名次'], errors='coerce').fillna(99)
        if '_result_win_odds' in df_backtest.columns: df_backtest['回測獨贏賠率'] = pd.to_numeric(df_backtest['_result_win_odds'], errors='coerce')
        if '_result_place_odds' in df_backtest.columns: df_backtest['回測位置賠率'] = pd.to_numeric(df_backtest['_result_place_odds'], errors='coerce')
        df_backtest = df_backtest[df_backtest['_result_rank'].notna()].copy()
        if not df_backtest.empty:
            backtest_ready = True
            st.sidebar.success(f"✅ 成功配對回測賽果：共 {len(df_backtest)} 匹")
    except Exception as exc: st.sidebar.error(f"回測讀取失敗：{exc}")

tab1, tab2 = st.tabs(["🎯 各場次預測推薦與單場凍結", "📈 歷史回測 (買時選馬 vs 最終派彩)"])

with tab1:
    col_v, col_d, col_r, col_t = st.columns([1.2, 1.8, 1.2, 1.8])
    try:
        race_id_prefix = str(df_raw['賽事編號'].iloc[0]).split('-')[0]
        auto_api_date = f"{race_id_prefix[:4]}-{race_id_prefix[4:6]}-{race_id_prefix[6:]}"
    except: auto_api_date = datetime.now().strftime("%Y-%m-%d")
        
    venue_code = "HV" if col_v.selectbox("場地", ["HV (跑馬地)", "ST (沙田)"], index=1 if str(df_raw.get('racecourse_code',pd.Series([''])).iloc[0])=='ST' else 0).startswith("HV") else "ST"
    api_date = col_d.text_input("API 日期", value=auto_api_date)
    races_available = sorted(int(x) for x in df_raw['賽事編號'].map(extract_race_no).dropna().unique())
    target_race = col_r.selectbox("操作場次", races_available) if races_available else col_r.number_input("場次", 1, 15, 1)
    
    target_time = col_t.time_input("預定開跑時間", value=datetime.strptime("18:45", "%H:%M").time())
    try: race_date = datetime.strptime(api_date.strip(), "%Y-%m-%d").date()
    except ValueError: st.error("API 日期格式必須是 YYYY-MM-DD。"); st.stop()

    target_datetime = datetime.combine(race_date, target_time).replace(tzinfo=hk_timezone)
    timer_html = r"""<div id="race-countdown" style="font: bold 30px monospace; text-align: center; padding: 8px;">倒數計時器…</div>
    <script>
    const targetMs = Date.parse("__TARGET_ISO__"); const el = document.getElementById("race-countdown");
    const startW = Date.now(); const startM = performance.now();
    function update() {
      const remain = targetMs - (startW + performance.now() - startM);
      if (remain > 0) {
        const secs = Math.floor(remain / 1000);
        el.style.color = "#1769aa";
        el.textContent = `${String(Math.floor(secs/3600)).padStart(2,"0")}:${String(Math.floor((secs%3600)/60)).padStart(2,"0")}:${String(secs%60).padStart(2,"0")}`;
        requestAnimationFrame(update);
      } else { el.style.color = "#b00020"; el.textContent = "已到預定開跑時間"; }
    } requestAnimationFrame(update);
    </script>""".replace("__TARGET_ISO__", target_datetime.isoformat())
    components.html(timer_html, height=70, scrolling=False)

    auto_col, interval_col, button_col, lock_col = st.columns([1.3, 1.3, 2.0, 2.0])
    auto_refresh = auto_col.checkbox("自動更新未凍結")
    refresh_seconds = interval_col.selectbox("間隔(秒)", [10, 30, 60], index=1, disabled=not auto_refresh)
    manual_refresh = button_col.button("🔄 立即更新", use_container_width=True)
    if lock_col.button("📌 鎖定早盤基準", use_container_width=True):
        st.session_state['baseline_odds_map'] = {f"{normalize_race_id(r['賽事編號'])}_{normalize_horse_no(r['馬號'])}": float(r['獨贏賠率']) for _,r in df.iterrows()}
        st.session_state['baseline_locked_time'] = datetime.now(hk_timezone).strftime("%H:%M:%S")
        st.rerun()

    f_col1, f_col2, f_col3 = st.columns([2.2, 1.8, 2.6])
    if int(target_race) not in st.session_state['frozen_races']:
        if f_col1.button(f"🔒 凍結第 {target_race} 場賠率", use_container_width=True, type="primary"):
            st.session_state['frozen_races'][int(target_race)] = datetime.now(hk_timezone).strftime("%H:%M:%S")
            st.rerun()
    else:
        if f_col1.button(f"🔓 解除第 {target_race} 場凍結", use_container_width=True):
            st.session_state['frozen_races'].pop(int(target_race), None)
            st.rerun()
    if f_col2.button("🔓 解除全場凍結", use_container_width=True): st.session_state['frozen_races'] = {}; st.rerun()
    f_col3.info("🔒 已凍結: " + (", ".join([f"R{r}" for r in sorted(st.session_state['frozen_races'].keys())]) if st.session_state['frozen_races'] else "無"))

    if auto_refresh: st_autorefresh(interval=int(refresh_seconds * 1000), key="odds_refresh")
    
    if auto_refresh or manual_refresh:
        with st.spinner("正在讀取當日 WIN／PLACE 即時賠率…"):
            live_by_race, server_updated_at, error = fetch_live_odds(api_date, venue_code)

        if live_by_race:
            df_temp = st.session_state['df_data'].copy()
            row_races = df_temp['賽事編號'].map(extract_race_no)
            row_horses = df_temp['馬號'].map(normalize_horse_no)
            updated_races_count, skipped_frozen = 0, []

            for race_no, markets in live_by_race.items():
                r_int = int(race_no)
                if r_int in st.session_state['frozen_races']:
                    skipped_frozen.append(f"R{r_int}")
                    continue
                updated_races_count += 1
                race_mask = row_races == r_int
                for horse_no, value in markets['WIN'].items(): df_temp.loc[race_mask & (row_horses == horse_no), '獨贏賠率'] = value
                for horse_no, value in markets['PLA'].items(): df_temp.loc[race_mask & (row_horses == horse_no), '位置賠率'] = value

            if st.session_state.get('baseline_locked_time') == "初始載入值":
                st.session_state['baseline_odds_map'] = {f"{normalize_race_id(r['賽事編號'])}_{normalize_horse_no(r['馬號'])}": float(r['獨贏賠率']) for _,r in df_temp.iterrows()}
                st.session_state['baseline_locked_time'] = f"首次鎖定 ({datetime.now(hk_timezone).strftime('%H:%M:%S')})"

            st.session_state['df_data'] = df_temp
            st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version', 0) + 1
            st.session_state['odds_last_status'] = f"✅ 成功更新 {updated_races_count} 場最新賠率"
            st.session_state['odds_last_error'] = None
            if manual_refresh: st.rerun()
        else:
            st.session_state['odds_last_error'] = f"⚠️ 無法取得賠率：{error}"
            st.session_state['odds_last_status'] = None
            if manual_refresh: st.rerun()

    # 狀態列
    locked_time_str = st.session_state.get('baseline_locked_time', '初始載入值')
    form_mode_str = "🔥 真實近況與同程算分" if use_dynamic_form else "🎯 1004 爆冷模式 (固定常數算分)"
    st.caption(f"📌 早盤基準：`{locked_time_str}` ｜ 排序：`{strategy_mode}` ｜ 算分引擎：`{form_mode_str}`")

    if st.session_state.get('odds_last_error'): st.warning(st.session_state['odds_last_error'])
    elif st.session_state.get('odds_last_status'): st.success(st.session_state['odds_last_status'])

    # 🎯 推薦清單 (雙引擎分流)
    recommendations = []
    for race_id, group in df.groupby('賽事編號'):
        r_no = extract_race_no(race_id)
        freeze_badge = f"🔒 已凍結 ({st.session_state['frozen_races'][r_no]})" if r_no in st.session_state['frozen_races'] else "🟢 即時跳動中"
        
        sg_win = filter_and_sort_group(group, strategy_mode, min_ev, min_odds, max_odds, bet_type="win")
        sg_place = filter_and_sort_group(group, strategy_mode, min_ev, min_odds, max_odds, bet_type="place")
        
        win_pick = "⚠️ 獨贏無達標馬匹"
        q_pick = "⚠️ 位置/Q無達標馬匹"
        qp_pick = "⚠️ 位置/QP無達標馬匹"
        
        if len(sg_win) >= 1:
            t1 = sg_win.iloc[0]
            last_r_str = f"上仗:{int(t1['real_last_rank'])}" if t1['real_last_rank'] != 6.5 else "上仗:無"
            win_pick = f"{t1['馬號']} ({t1['馬名']}) [AI勝率:{t1['pred_win_prob']*100:.1f}% | 獨贏EV:{t1['win_ev']:.2f}] {t1['資金流向']} 💰${t1['suggested_stake_win']:.0f}"
            
        if len(sg_place) >= 2:
            p1, p2 = sg_place.iloc[0], sg_place.iloc[1]
            q_pick = f"{p1['馬號']} + {p2['馬號']} ({p1['馬名']}/{p2['馬名']}) [位置EV: {p1['place_ev']:.2f} & {p2['place_ev']:.2f}]"
            if len(sg_place) >= 3:
                p3 = sg_place.iloc[2]
                qp_pick = f"{p1['馬號']}+{p2['馬號']} 或 {p1['馬號']}+{p3['馬號']}"
            else:
                qp_pick = f"{p1['馬號']}+{p2['馬號']} (僅一注)"
                
        if len(sg_win) > 0 or len(sg_place) > 0:
            recommendations.append({
                '賽事編號': race_id,
                '狀態': freeze_badge,
                '🎯 獨贏首選 (Win Engine)': win_pick,
                '🥉 位置/Q首選 (Place Engine)': q_pick,
                '🔗 位置QP (Place Engine)': qp_pick
            })
            
    st.subheader("🎯 雙渦輪引擎推薦清單")
    if recommendations: st.dataframe(pd.DataFrame(recommendations), hide_index=True)
    
    # 單場雷達表 (顯示雙機率與雙 EV)
    rdf = df[df['賽事編號'].map(extract_race_no) == target_race].copy()
    st.markdown(f"#### 🔍 第 {target_race} 場 深度雷達")
    if not rdf.empty:
        sort_col = 'smart_score_win' if strategy_mode.startswith("🛡️") else ('pred_win_prob' if strategy_mode.startswith("🎯") else 'win_ev')
        rdf = rdf.sort_values(by=sort_col, ascending=False)
        st.dataframe(pd.DataFrame({
            '馬號': rdf['馬號'], '馬名': rdf['馬名'], '檔位': rdf['排位檔位'], 
            'AI勝率': (rdf['pred_win_prob']*100).round(1).astype(str)+'%',
            'AI入位率': (rdf['pred_place_prob']*100).round(1).astype(str)+'%',
            '獨賠': rdf['獨贏賠率'].round(1), '位賠': rdf['位置賠率'].round(1),
            '獨贏EV': rdf['win_ev'].round(2), '位置EV': rdf['place_ev'].round(2),
            '流向': rdf['資金流向'], '獨注碼': rdf['suggested_stake_win'].apply(lambda x: f"${x:.0f}")
        }), use_container_width=True, hide_index=True)

# ---------------- 分頁 2: 歷史回測 ----------------
with tab2:
    st.subheader("📊 雙引擎策略回測總覽 (真實凱利注碼 vs 官方派彩)")
    if not backtest_ready:
        st.warning("請在左側上傳已完賽的結果 CSV。需有『賽事編號』、『馬號』及『名次』。")
    else:
        st.markdown("---")
        if st.button("🪄 一鍵 AI 參數尋優 (獨贏最高 ROI)", type="primary", use_container_width=True):
            with st.spinner("🤖 正在運算上百種參數組合..."):
                best_roi = -999
                best_params = None
                for test_ev in np.arange(1.0, 4.1, 0.5):
                    for test_odds in np.arange(2.0, 10.1, 1.0):
                        t_inv, t_ret, bets_cnt = 0, 0, 0
                        for race_id, group in df_backtest.groupby('賽事編號'):
                            sg = filter_and_sort_group(group, strategy_mode, test_ev, test_odds, max_odds, bet_type="win")
                            if len(sg) > 0:
                                pick = sg.iloc[0]
                                stake = max(10, float(pick.get('suggested_stake_win', 10)) if pd.notna(pick.get('suggested_stake_win')) else 10)
                                t_inv += stake
                                bets_cnt += 1
                                if pick['numeric_rank'] == 1:
                                    s_odds = pick.get('回測獨贏賠率', np.nan)
                                    s_odds = pick['獨贏賠率'] if pd.isna(s_odds) or s_odds <= 1 else s_odds
                                    t_ret += stake * float(s_odds)
                        if t_inv > 0 and bets_cnt >= 2:
                            roi = ((t_ret - t_inv) / t_inv) * 100
                            if roi > best_roi: best_roi, best_params = roi, {'ev': test_ev, 'odds': test_odds, 'bets': bets_cnt, 'roi': roi, 'profit': t_ret - t_inv}
                if best_params and best_params['roi'] > 0:
                    st.success(f"🎉 **尋優完成！** 建議 EV `{best_params['ev']:.1f}` + 最低賠率 `{best_params['odds']:.1f}`。預期 ROI: **{best_params['roi']:.2f}%**")
                else: st.warning("⚠️ 找不到能產生穩定正利潤的參數組合。")
        st.markdown("---")

        sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs(["🥇 獨贏 (Win)", "🥈 單匹位置 (Place)", "🔗 位置Q (QP)", "🥉 3匹位置包抄", "🛒 模擬結算"])

        with sub_tab1:
            win_invested, win_return, win_bets, win_hits, win_records = 0, 0, 0, 0, []
            for race_id, group in df_backtest.groupby('賽事編號'):
                sorted_group = filter_and_sort_group(group, strategy_mode, min_ev, min_odds, max_odds, bet_type="win")
                if len(sorted_group) > 0:
                    pick = sorted_group.iloc[0]
                    stake = max(10, float(pick.get('suggested_stake_win', 10)) if pd.notna(pick.get('suggested_stake_win')) else 10)
                    win_bets += 1; win_invested += stake
                    is_hit = (pick['numeric_rank'] == 1)
                    s_odds = pick.get('回測獨贏賠率', np.nan)
                    s_odds = pick['獨贏賠率'] if pd.isna(s_odds) or s_odds <= 1 else s_odds
                    payout = stake * float(s_odds) if is_hit else 0
                    if is_hit: win_hits += 1; win_return += payout
                    win_records.append({'賽事編號': race_id, '馬名': pick['馬名'], '實際名次': str(pick['名次']).replace('.0', ''), '買時賠率': pick['獨贏賠率'], '派彩賠率': float(s_odds), 'AI勝率': f"{pick['pred_win_prob']*100:.1f}%", 'EV': round(pick['win_ev'], 2), '結果': "✅ 贏" if is_hit else "❌ 輸", '淨盈虧': f"${payout - stake:.1f}"})
            if win_bets > 0:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("投注場數", f"{win_bets} 場"); c2.metric("勝率", f"{win_hits/win_bets*100:.1f}%"); c3.metric("總成本", f"${win_invested:,.0f}"); c4.metric("總回收", f"${win_return:,.1f}", f"ROI: {((win_return - win_invested) / win_invested)*100:.2f}%")
                st.dataframe(pd.DataFrame(win_records), use_container_width=True, hide_index=True)

        with sub_tab2:
            place_invested, place_return, place_bets, place_hits, place_records = 0, 0, 0, 0, []
            for race_id, group in df_backtest.groupby('賽事編號'):
                sorted_group = filter_and_sort_group(group, strategy_mode, min_ev, min_odds, max_odds, bet_type="place")
                if len(sorted_group) > 0:
                    pick = sorted_group.iloc[0]
                    stake = max(10, float(pick.get('suggested_stake_place', 10)) if pd.notna(pick.get('suggested_stake_place')) else 10)
                    place_bets += 1; place_invested += stake
                    is_hit = (pick['numeric_rank'] <= 3)
                    s_odds = pick.get('回測位置賠率', np.nan)
                    p_odds = float(s_odds) if pd.notna(s_odds) and float(s_odds) > 1.0 else 1.5
                    payout = stake * p_odds if is_hit else 0
                    if is_hit: place_hits += 1; place_return += payout
                    place_records.append({'賽事編號': race_id, '馬名': pick['馬名'], '實際名次': str(pick['名次']).replace('.0', ''), 'AI入位率': f"{pick['pred_place_prob']*100:.1f}%", '位置EV': round(pick['place_ev'], 2), '結果': "✅ 命中" if is_hit else "❌ 落空", '淨盈虧': f"${payout - stake:.1f}"})
            if place_bets > 0:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("投注場數", f"{place_bets} 場"); c2.metric("位置勝率", f"{place_hits/place_bets*100:.1f}%"); c3.metric("總成本", f"${place_invested:,.0f}"); c4.metric("總回收", f"${place_return:,.1f}", f"ROI: {((place_return - place_invested) / place_invested)*100:.2f}%")
                st.dataframe(pd.DataFrame(place_records), use_container_width=True, hide_index=True)

        with sub_tab3:
            qp_invested, qp_return, qp_bets, qp_hits, qp_records = 0, 0, 0, 0, []
            for race_id, group in df_backtest.groupby('賽事編號'):
                sorted_group = filter_and_sort_group(group, strategy_mode, min_ev, min_odds, max_odds, bet_type="place")
                if len(sorted_group) >= 2:
                    t1, t2 = sorted_group.iloc[0], sorted_group.iloc[1]
                    stake = max(10, min(float(t1.get('suggested_stake_place',10)), float(t2.get('suggested_stake_place',10))))
                    qp_bets += 1; qp_invested += stake
                    is_hit = (t1['numeric_rank'] <= 3) and (t2['numeric_rank'] <= 3)
                    p1_o, p2_o = t1.get('回測位置賠率', np.nan), t2.get('回測位置賠率', np.nan)
                    p1_o = float(p1_o) if pd.notna(p1_o) and p1_o > 1.0 else float(t1['位置賠率'])
                    p2_o = float(p2_o) if pd.notna(p2_o) and p2_o > 1.0 else float(t2['位置賠率'])
                    est_qp = round(p1_o * p2_o * 1.8, 1)
                    payout = stake * est_qp if is_hit else 0
                    if is_hit: qp_hits += 1; qp_return += payout
                    qp_records.append({'賽事編號': race_id, 'QP 組合': f"{t1['馬名']} / {t2['馬名']}", '實際名次': f"{str(t1['名次']).replace('.0','')} / {str(t2['名次']).replace('.0','')}", '估算QP賠率': est_qp, '結果': "✅ 命中" if is_hit else "❌ 落空", '淨盈虧': f"${payout - stake:.1f}"})
            if qp_bets > 0:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("投注場數", f"{qp_bets} 場"); c2.metric("QP勝率", f"{qp_hits/qp_bets*100:.1f}%"); c3.metric("總成本", f"${qp_invested:,.0f}"); c4.metric("總回收", f"${qp_return:,.1f}", f"ROI: {((qp_return - qp_invested) / qp_invested)*100:.2f}%")
                st.dataframe(pd.DataFrame(qp_records), use_container_width=True, hide_index=True)

        with sub_tab4:
            p3_invested, p3_return, p3_bets, p3_hits, p3_records = 0, 0, 0, 0, []
            for race_id, group in df_backtest.groupby('賽事編號'):
                sorted_group = filter_and_sort_group(group, strategy_mode, min_ev, min_odds, max_odds, bet_type="place")
                for _, pick in sorted_group.head(3).iterrows():
                    stake = max(10, float(pick.get('suggested_stake_place', 10)) if pd.notna(pick.get('suggested_stake_place')) else 10)
                    p3_bets += 1; p3_invested += stake
                    is_hit = (pick['numeric_rank'] <= 3)
                    s_o = pick.get('回測位置賠率', np.nan)
                    p_odds = float(s_o) if pd.notna(s_o) and float(s_o) > 1.0 else float(pick['位置賠率'])
                    payout = stake * p_odds if is_hit else 0
                    if is_hit: p3_hits += 1; p3_return += payout
                    p3_records.append({'賽事編號': race_id, '馬名': pick['馬名'], '實際名次': str(pick['名次']).replace('.0', ''), '派彩賠率': round(p_odds, 2), '結果': "✅ 命中" if is_hit else "❌ 落空", '淨盈虧': f"${payout - stake:.1f}"})
            if p3_bets > 0:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("總投注注數", f"{p3_bets} 注"); c2.metric("命中注數", f"{p3_hits} 注"); c3.metric("總成本", f"${p3_invested:,.0f}"); c4.metric("總回收", f"${p3_return:,.1f}", f"ROI: {((p3_return - p3_invested) / p3_invested)*100:.2f}%")
                # 準備好你要顯示的資料表
radar_df = pd.DataFrame({
    '馬號': rdf['馬號'], '馬名': rdf['馬名'], '檔位': rdf['排位檔位'], 
    '獨賠': rdf['獨贏賠率'].round(1), '位賠': rdf['位置賠率'].round(1),
    '獨贏EV': rdf['win_ev'].round(2), '位置EV': rdf['place_ev'].round(2),
    '流向': rdf['資金流向']
})

# 加上漸層顏色 (cmap='YlGn' 代表黃到綠，'Reds' 代表淺紅到深紅)
styled_radar = radar_df.style.background_gradient(cmap='YlGn', subset=['獨贏EV', '位置EV']) \
                             .background_gradient(cmap='Reds', subset=['獨賠'])

# 顯示這張帶有熱力圖的精美表格
st.dataframe(styled_radar, use_container_width=True, hide_index=True)

        with sub_tab5:
            st.info("請於賽前預測分頁下單後，再至此處查看模擬結算。")
