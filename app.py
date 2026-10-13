import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import joblib
import requests
import re
import time
from io import StringIO
from datetime import datetime, timedelta, timezone
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components

# ==========================================
# 頁面基本與美化設定 (CSS 注入)
# ==========================================
st.set_page_config(page_title="HKJC AI 智能賽馬預測系統", page_icon="🐎", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #33343d;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }
    
    .stButton > button {
        border-radius: 8px;
        font-weight: 800;
        transition: all 0.3s ease-in-out;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 4px 10px rgba(255, 75, 75, 0.4);
    }
</style>
""", unsafe_allow_html=True)

st.title("🐎 HKJC 旗艦 AI 預測系統 (雙渦輪引擎 + 專業視覺版)")
st.markdown("---")

# 初始化紀錄
if 'simulated_bets' not in st.session_state:
    st.session_state['simulated_bets'] = pd.DataFrame(columns=['下注時間', '賽事編號', '場次', '馬號', '馬名', '玩法', '買入賠率', '注碼'])
if 'frozen_races' not in st.session_state:
    st.session_state['frozen_races'] = {} 

# 模型路徑
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / 'my_hkjc_model.pkl'
PLACE_MODEL_PATH = APP_DIR / 'my_hkjc_place_model.pkl'

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
    st.error(f"⚠️ 找不到獨贏模型檔 `{MODEL_PATH}`！")
    st.stop()

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
        if '賽事編號' in valid_ranks_df.columns: valid_ranks_df = valid_ranks_df.sort_values('賽事編號')
        horse_last_rank_stats = valid_ranks_df.groupby('_clean_horse', as_index=False).last()[['_clean_horse', 'numeric_rank']].rename(columns={'numeric_rank': 'hist_horse_last_rank'})
        
        if '場地' in hist_df.columns:
            hist_df['_clean_surface'] = hist_df['場地'].astype(str).str.strip()
            horse_surface_stats = hist_df.groupby(['_clean_horse', '_clean_surface'], as_index=False)['is_win'].mean().rename(columns={'is_win': 'hist_horse_surface_win_rate'})
        else: horse_surface_stats = pd.DataFrame(columns=['_clean_horse', '_clean_surface', 'hist_horse_surface_win_rate'])
        
        if '距離' in hist_df.columns:
            hist_df['_clean_dist'] = pd.to_numeric(hist_df['距離'], errors='coerce').fillna(1200).astype(int)
            horse_dist_stats = hist_df.groupby(['_clean_horse', '_clean_dist'], as_index=False)['is_win'].mean().rename(columns={'is_win': 'hist_horse_dist_win_rate'})
        else: horse_dist_stats = pd.DataFrame(columns=['_clean_horse', '_clean_dist', 'hist_horse_dist_win_rate'])
        
        return (jockey_stats, trainer_stats, horse_stats, combo_stats, horse_last_rank_stats, horse_surface_stats, horse_dist_stats)
    except Exception: return None

def fetch_live_odds(date_str, venue):
    GRAPHQL_URL = "https://info.cld.hkjc.com/graphql/base/"
    ODDS_QUERY = """query racing($date: String,$venueCode: String, $oddsTypes: [OddsType],$raceNo: Int) { raceMeetings(date: $date, venueCode:$venueCode) { pmPools(oddsTypes: $oddsTypes, raceNo:$raceNo) { id status sellStatus oddsType lastUpdateTime leg { number races } oddsNodes { combString oddsValue } } } }"""
    GRAPHQL_HEADERS = { "Accept": "*/*", "Content-Type": "application/json", "Origin": "https://bet.hkjc.com", "User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.post(GRAPHQL_URL, headers=GRAPHQL_HEADERS, json={"operationName": "racing", "variables": {"date": date_str, "venueCode": venue, "raceNo": None, "oddsTypes": ["WIN", "PLA"]}, "query": ODDS_QUERY}, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("errors"): return None, None, f"GraphQL 伺服器拒絕：{payload['errors']}"
        meetings = (payload.get("data") or {}).get("raceMeetings") or []
        if not meetings: return None, None, "無資料（可能尚未開盤）"
        odds_by_race, timestamps = {}, []
        for pool in meetings[0].get("pmPools") or []:
            pool_type = str(pool.get("oddsType", "")).upper()
            if pool_type not in ("WIN", "PLA"): continue
            if pool.get("lastUpdateTime"): timestamps.append(pool["lastUpdateTime"])
            for race_number in (pool.get("leg") or {}).get("races") or []:
                market = odds_by_race.setdefault(int(race_number), {"WIN": {}, "PLA": {}})
                for node in pool.get("oddsNodes") or []:
                    horse_no = str(node.get("combString")).strip()
                    try: horse_no = str(int(float(horse_no)))
                    except: pass
                    odds_val = str(node.get("oddsValue")).strip()
                    if horse_no and odds_val not in ("None", "", "SCR"):
                        try: market[pool_type][horse_no] = float(odds_val)
                        except: pass
        return odds_by_race, (max(timestamps) if timestamps else None), None
    except Exception as exc: return None, None, f"連線錯誤：{exc}"

# ==========================================
# 側邊欄設定與資料讀取
# ==========================================
st.sidebar.header("📂 預測資料")
available_csvs = sorted([f.name for f in APP_DIR.glob("prediction*.csv")])
selected_csv_name = st.sidebar.selectbox("📜 選擇預測賽卡", available_csvs) if available_csvs else "prediction.csv"

if st.session_state.get('active_csv_name') != selected_csv_name:
    st.session_state['active_csv_name'] = selected_csv_name
    st.session_state.pop('df_data', None)
    st.session_state.pop('prediction_signature', None)
    st.session_state['frozen_races'] = {}
    st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version', 0) + 1

SELECTED_CSV_PATH = APP_DIR / selected_csv_name
try: df_raw = pd.read_csv(SELECTED_CSV_PATH, encoding='utf-8-sig')
except: df_raw = pd.read_csv(SELECTED_CSV_PATH, encoding='cp950')
prediction_signature = f"{SELECTED_CSV_PATH.name}:{SELECTED_CSV_PATH.stat().st_mtime_ns}"

if '獨贏賠率' not in df_raw.columns: df_raw['獨贏賠率'] = 10.0
if '位置賠率' not in df_raw.columns: df_raw['位置賠率'] = 1.0 + (pd.to_numeric(df_raw['獨贏賠率'], errors='coerce').fillna(10.0) - 1.0) / 3.2

if 'df_data' not in st.session_state or st.session_state.get('prediction_signature') != prediction_signature:
    st.session_state['df_data'] = df_raw.copy()
    st.session_state['prediction_signature'] = prediction_signature
    st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version', 0) + 1
    
    def norm_h(x):
        try: return str(int(float(str(x).strip())))
        except: return str(x).strip()
    def norm_r(x):
        m = re.search(r"(20\d{6}).*?(\d+)\s*$", str(x).strip())
        return f"{m.group(1)}-{int(m.group(2)):02d}" if m else str(x).strip()
    
    st.session_state['baseline_odds_map'] = {f"{norm_r(r['賽事編號'])}_{norm_h(r['馬號'])}": float(pd.to_numeric(r['獨贏賠率'], errors='coerce')) for _,r in df_raw.iterrows()}
    st.session_state['baseline_locked_time'] = "初始載入值"

df_editable = st.session_state['df_data'][['賽事編號', '馬號', '馬名', '排位檔位', '獨贏賠率', '位置賠率']].copy()
edited_df = st.data_editor(df_editable, disabled=['賽事編號', '馬號', '馬名', '排位檔位'], use_container_width=True, hide_index=True, key=f"odds_editor_{st.session_state['odds_editor_version']}")
st.session_state['df_data']['獨贏賠率'] = pd.to_numeric(edited_df['獨贏賠率'], errors='coerce').fillna(10.0)
st.session_state['df_data']['位置賠率'] = pd.to_numeric(edited_df['位置賠率'], errors='coerce').fillna(1.5)

backtest_file = st.sidebar.file_uploader("回測用：上傳已完成賽事 CSV", type=['csv'], key='backtest_results_upload')

st.sidebar.markdown("---")
st.sidebar.header("⚙ 投注策略與算分引擎")
strategy_mode = st.sidebar.radio("1️⃣ 推薦排序邏輯", ["⚡ 傳統純 EV 排序", "🛡️ 抗噪穩定 (過濾回飛)", "🎯 純 AI 勝率最高"], index=0)
use_dynamic_form = st.sidebar.checkbox("2️⃣ 將「上仗名次與同程」納入 AI 算分", value=False)
min_ev = st.sidebar.slider("最小期望值 (EV 門檻)", 0.0, 6.0, 0.0, 0.1)
min_odds = st.sidebar.number_input("最低獨贏賠率", min_value=1.0, max_value=50.0, value=3.0)
max_odds = st.sidebar.number_input("最高獨贏賠率", min_value=1.0, max_value=100.0, value=20.0)

st.sidebar.markdown("---")
st.sidebar.header("💰 資金管理 (凱利精算)")
total_bankroll = st.sidebar.number_input("單日預算 (HKD)", min_value=100, value=5000, step=500)
kelly_fraction = st.sidebar.slider("凱利保守係數", 0.05, 1.0, 0.15, 0.05)

# ==========================================
# 核心特徵計算與模型預測
# ==========================================
df = st.session_state['df_data'].copy()
def norm_h(x):
    try: return str(int(float(str(x).strip())))
    except: return str(x).strip()
def norm_r(x):
    m = re.search(r"(20\d{6}).*?(\d+)\s*$", str(x).strip())
    return f"{m.group(1)}-{int(m.group(2)):02d}" if m else str(x).strip()

df['_horse_uid'] = df.apply(lambda r: f"{norm_r(r['賽事編號'])}_{norm_h(r['馬號'])}", axis=1)
df['基準賠率'] = pd.to_numeric(df['_horse_uid'].map(st.session_state.get('baseline_odds_map', {})).fillna(df['獨贏賠率']), errors='coerce')
df['odds_drop_pct'] = ((df['基準賠率'] - df['獨贏賠率']) / df['基準賠率'].replace(0, np.nan)).fillna(0.0) * 100.0
df['資金流向'] = df['odds_drop_pct'].apply(lambda pct: f"🔥 大落飛 (-{pct:.0f}%)" if pct >= 20 else (f"📉 落飛 (-{pct:.0f}%)" if pct >= 10 else (f"🥶 回飛 (+{abs(pct):.0f}%)" if pct <= -20 else "➖ 穩定")))

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

X_predict = df[FEATURE_COLS_15].fillna(0)
df['pred_win_prob'] = model_win.predict_proba(X_predict)[:, 1]
df['pred_place_prob'] = model_place.predict_proba(X_predict)[:, 1] if model_place else df['pred_win_prob']

df['win_ev'] = df['pred_win_prob'] * df['獨贏賠率']
df['place_ev'] = df['pred_place_prob'] * df['位置賠率']
drop_bonus = np.clip(df['odds_drop_pct'] / 100.0, -0.35, 0.30)
df['smart_score_win'] = df['pred_win_prob'] * (1.0 + drop_bonus)
df['smart_score_place'] = df['pred_place_prob'] * (1.0 + drop_bonus)

df['suggested_stake_win'] = np.floor((total_bankroll * np.clip((df['win_ev'] - 1) / (np.maximum(df['獨贏賠率'], 1.01) - 1), 0, 1) * kelly_fraction) / 10) * 10
df['suggested_stake_place'] = np.floor((total_bankroll * np.clip((df['place_ev'] - 1) / (np.maximum(df['位置賠率'], 1.01) - 1), 0, 1) * kelly_fraction) / 10) * 10

def filter_and_sort_group(group, mode, min_ev_val, min_odds_val, max_odds_val, bet_type="win"):
    if bet_type == "win":
        cond = (group['win_ev'] >= min_ev_val) & (group['獨贏賠率'] >= min_odds_val) & (group['獨贏賠率'] <= max_odds_val)
        sort_col = 'smart_score_win' if mode.startswith("🛡️") else ('pred_win_prob' if mode.startswith("🎯") else 'win_ev')
    else:
        cond = (group['place_ev'] >= min_ev_val) & (group['位置賠率'] >= max(1.01, min_odds_val / 3.0))
        sort_col = 'smart_score_place' if mode.startswith("🛡️") else ('pred_place_prob' if mode.startswith("🎯") else 'place_ev')
    if mode.startswith("🛡️"): cond = cond & (group['odds_drop_pct'] > -25.0)
    return group[cond].sort_values(by=sort_col, ascending=False).reset_index(drop=True)

df_backtest = df.copy()
backtest_ready = False
if backtest_file is not None:
    try:
        try: result_raw = pd.read_csv(backtest_file, encoding='utf-8-sig')
        except:
            backtest_file.seek(0)
            result_raw = pd.read_csv(backtest_file, encoding='cp950')
        result_map = result_raw.copy()
        result_map['_race_key'] = result_map['賽事編號'].map(norm_r)
        result_map['_horse_key'] = result_map['馬號'].map(norm_h)
        rename_map = {'名次': '_result_rank'}
        if '獨贏賠率' in result_map.columns: rename_map['獨贏賠率'] = '_result_win_odds'
        if '位置賠率' in result_map.columns: rename_map['位置賠率'] = '_result_place_odds'
        result_map = result_map.drop_duplicates(['_race_key', '_horse_key'], keep='last').rename(columns=rename_map)
        df_backtest['_race_key'] = df_backtest['賽事編號'].map(norm_r)
        df_backtest['_horse_key'] = df_backtest['馬號'].map(norm_h)
        df_backtest = df_backtest.merge(result_map[['_race_key', '_horse_key'] + [c for c in rename_map.values() if c in result_map.columns]], on=['_race_key', '_horse_key'], how='left')
        df_backtest['名次'] = df_backtest['_result_rank']
        df_backtest['numeric_rank'] = pd.to_numeric(df_backtest['名次'], errors='coerce').fillna(99)
        if '_result_win_odds' in df_backtest.columns: df_backtest['回測獨贏賠率'] = pd.to_numeric(df_backtest['_result_win_odds'], errors='coerce')
        if '_result_place_odds' in df_backtest.columns: df_backtest['回測位置賠率'] = pd.to_numeric(df_backtest['_result_place_odds'], errors='coerce')
        df_backtest = df_backtest[df_backtest['_result_rank'].notna()].copy()
        if not df_backtest.empty:
            backtest_ready = True
            st.sidebar.success(f"✅ 成功配對回測賽果：共 {len(df_backtest)} 匹")
    except Exception: st.sidebar.error("回測讀取失敗")

tab1, tab2 = st.tabs(["🎯 預測推薦與雷達表", "📈 歷史回測"])

# ==========================================
# 分頁 1: 預測推薦與熱力雷達表
# ==========================================
with tab1:
    col_v, col_d, col_r, col_t = st.columns([1.2, 1.8, 1.2, 1.8])
    hk_timezone = timezone(timedelta(hours=8))
    try:
        race_id_prefix = str(df_raw['賽事編號'].iloc[0]).split('-')[0]
        auto_api_date = f"{race_id_prefix[:4]}-{race_id_prefix[4:6]}-{race_id_prefix[6:]}"
    except: auto_api_date = datetime.now().strftime("%Y-%m-%d")
        
    venue_code = "HV" if col_v.selectbox("場地", ["HV (跑馬地)", "ST (沙田)"], index=1 if str(df_raw.get('racecourse_code',pd.Series([''])).iloc[0])=='ST' else 0).startswith("HV") else "ST"
    api_date = col_d.text_input("API 日期", value=auto_api_date)
    
    def extract_race_no(value):
        m = re.search(r"-(\d+)\s*$", str(value).strip())
        return int(m.group(1)) if m else None
        
    races_available = sorted(int(x) for x in df_raw['賽事編號'].map(extract_race_no).dropna().unique())
    target_race = col_r.selectbox("操作場次", races_available) if races_available else col_r.number_input("場次", 1, 15, 1)
    target_time = col_t.time_input("預定開跑時間", value=datetime.strptime("18:45", "%H:%M").time())
    
    auto_col, interval_col, button_col, lock_col = st.columns([1.3, 1.3, 2.0, 2.0])
    auto_refresh = auto_col.checkbox("自動更新未凍結")
    refresh_seconds = interval_col.selectbox("間隔(秒)", [10, 30, 60], index=1, disabled=not auto_refresh)
    manual_refresh = button_col.button("🔄 立即更新", use_container_width=True)
    if lock_col.button("📌 鎖定早盤基準", use_container_width=True):
        st.session_state['baseline_odds_map'] = {f"{norm_r(r['賽事編號'])}_{norm_h(r['馬號'])}": float(r['獨贏賠率']) for _,r in df.iterrows()}
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
    if f_col2.button("🔓
