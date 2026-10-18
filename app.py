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

st.title("🐎 HKJC 旗艦 15 大特徵 AI 預測系統 (雙渦輪引擎+上仗同程+步速+手風+賽道+尋優)")
st.markdown("---")

# 初始化模擬投注紀錄與單場凍結名單
if 'simulated_bets' not in st.session_state:
    st.session_state['simulated_bets'] = pd.DataFrame(columns=[
        '下注時間', '賽事編號', '場次', '馬號', '馬名', '玩法', '買入賠率', '注碼'
    ])
if 'frozen_races' not in st.session_state:
    st.session_state['frozen_races'] = {}  # 格式: {race_no: "HH:MM:SS"}

# 設定 repo 內的模型及預測 CSV 路徑
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / 'my_hkjc_model.pkl'
PLACE_MODEL_PATH = APP_DIR / 'my_hkjc_place_model.pkl'
PREDICTION_CSV_PATH = APP_DIR / 'prediction.csv'
FEATURE_COLS_15 = [
    'market_implied_prob', '獨贏賠率', 'odds_rank', 'is_favorite',
    '排位檔位', 'weight_diff', 'weight_rank',
    'jockey_win_rate', 'trainer_win_rate', 'combo_win_rate',
    'horse_win_rate', 'horse_last_rank',
    '距離', 'horse_surface_win_rate', 'horse_dist_win_rate',
]

@st.cache_resource
def load_models():
    win_m = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    place_m = joblib.load(PLACE_MODEL_PATH) if os.path.exists(PLACE_MODEL_PATH) else None
    return win_m, place_m

model_win, model_place = load_models()

if model_win is None:
    st.error(f"⚠️ 找不到獨贏 AI 模型檔 `{MODEL_PATH}`！請確認模型是否已上傳至正確目錄。")
    st.stop()

if model_place is not None:
    st.sidebar.success("🧠 雙渦輪引擎已啟動：獨贏模型 + 位置模型")
else:
    st.sidebar.warning("⚠️ 找不到 `my_hkjc_place_model.pkl`，位置彩池將暫用獨贏模型代算。")

@st.cache_data
def load_historical_stats(history_csv_path='hkjc_all_seasons_features.csv'):
    """從歷史大表中計算騎師、練馬師、馬匹勝率(含近50場手風)，以及馬匹上仗名次、場地勝率、路程勝率"""
    csv_path = Path(__file__).resolve().parent / history_csv_path
    if not os.path.exists(csv_path):
        return None
    
    try:
        hist_df = pd.read_csv(csv_path, low_memory=False)
        hist_df['numeric_rank'] = pd.to_numeric(hist_df['名次'], errors='coerce')
        hist_df['is_win'] = (hist_df['numeric_rank'] == 1).astype(int)
        
        # 預先清洗歷史大表的名稱欄位，確保比對精準
        hist_df['_clean_jockey'] = hist_df['騎師'].astype(str).str.replace(r'\(.*?\)', '', regex=True).str.replace(' ', '')
        hist_df['_clean_trainer'] = hist_df['練馬師'].astype(str).str.replace(' ', '')
        hist_df['_clean_horse'] = hist_df['馬名'].astype(str).str.replace(r'\(.*?\)', '', regex=True).str.replace(' ', '')
        
        # 1. 基本騎師、練馬師、馬匹、騎練合作勝率
        jockey_stats = hist_df.groupby('_clean_jockey', as_index=False)['is_win'].mean().rename(columns={'is_win': 'hist_jockey_win_rate'})
        trainer_stats = hist_df.groupby('_clean_trainer', as_index=False)['is_win'].mean().rename(columns={'is_win': 'hist_trainer_win_rate'})
        horse_stats = hist_df.groupby('_clean_horse', as_index=False)['is_win'].mean().rename(columns={'is_win': 'hist_horse_win_rate'})
        combo_stats = hist_df.groupby(['_clean_jockey', '_clean_trainer'], as_index=False)['is_win'].mean().rename(columns={'is_win': 'hist_combo_win_rate'})
        
        # 2. 近期手風 (近50場勝率) 與 馬匹「上仗名次 (horse_last_rank)」
        valid_ranks_df = hist_df.dropna(subset=['numeric_rank']).copy()
        if '賽事編號' in valid_ranks_df.columns:
            valid_ranks_df = valid_ranks_df.sort_values('賽事編號')
            
        recent_j_df = valid_ranks_df.groupby('_clean_jockey').tail(50)
        recent_jockey_stats = recent_j_df.groupby('_clean_jockey', as_index=False)['is_win'].mean().rename(columns={'is_win': 'recent_jockey_win_rate'})
        jockey_stats = jockey_stats.merge(recent_jockey_stats, on='_clean_jockey', how='left')

        recent_t_df = valid_ranks_df.groupby('_clean_trainer').tail(50)
        recent_trainer_stats = recent_t_df.groupby('_clean_trainer', as_index=False)['is_win'].mean().rename(columns={'is_win': 'recent_trainer_win_rate'})
        trainer_stats = trainer_stats.merge(recent_trainer_stats, on='_clean_trainer', how='left')

        horse_last_rank_stats = (
            valid_ranks_df.groupby('_clean_horse', as_index=False)
            .last()[['_clean_horse', 'numeric_rank']]
            .rename(columns={'numeric_rank': 'hist_horse_last_rank'})
        )
        
        # 3. 馬匹「場地勝率 (horse_surface_win_rate)」
        if '場地' in hist_df.columns:
            hist_df['_clean_surface'] = hist_df['場地'].astype(str).str.strip()
            horse_surface_stats = (
                hist_df.groupby(['_clean_horse', '_clean_surface'], as_index=False)['is_win']
                .mean()
                .rename(columns={'is_win': 'hist_horse_surface_win_rate'})
            )
        else:
            horse_surface_stats = pd.DataFrame(columns=['_clean_horse', '_clean_surface', 'hist_horse_surface_win_rate'])
            
        # 4. 馬匹「路程勝率 (horse_dist_win_rate)」
        if '距離' in hist_df.columns:
            hist_df['_clean_dist'] = pd.to_numeric(hist_df['距離'], errors='coerce').fillna(1200).astype(int)
            horse_dist_stats = (
                hist_df.groupby(['_clean_horse', '_clean_dist'], as_index=False)['is_win']
                .mean()
                .rename(columns={'is_win': 'hist_horse_dist_win_rate'})
            )
        else:
            horse_dist_stats = pd.DataFrame(columns=['_clean_horse', '_clean_dist', 'hist_horse_dist_win_rate'])
            
        return (
            jockey_stats, trainer_stats, horse_stats, combo_stats,
            horse_last_rank_stats, horse_surface_stats, horse_dist_stats
        )
    except Exception as e:
        st.error(f"載入歷史數據失敗：{e}")
        return None

# 根據所選日期從 HKJC 官方排位頁抓取賽卡
def find_racecard_table(html_text):
    try:
        tables = pd.read_html(StringIO(html_text))
    except ImportError as exc:
        if any(name in str(exc).lower() for name in ('lxml', 'html5lib', 'bs4', 'beautifulsoup')):
            raise RuntimeError(
                '缺少 pandas HTML 表格解析依賴。請確認 GitHub repo 的 requirements.txt 包含 '
                'lxml、html5lib、beautifulsoup4 三行。'
            ) from exc
        raise
    except ValueError:
        return None
    for table in tables:
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = ['_'.join(map(str, col)) for col in table.columns]
        table.columns = [str(col).replace('\n', ' ').strip() for col in table.columns]
        header_text = ' '.join(table.columns)
        if '馬名' in header_text and '騎師' in header_text:
            return table
    return None

def find_column(columns, keywords, fallback=None):
    return next((col for col in columns if any(word in str(col) for word in keywords)), fallback)

def fetch_hkjc_racecard(race_date):
    date_text = race_date.strftime('%Y/%m/%d')
    race_day_id = race_date.strftime('%Y%m%d')
    season_start = race_date.year if race_date.month >= 9 else race_date.year - 1
    season = f"{season_start % 100:02d}/{(season_start + 1) % 100:02d}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    session = requests.Session()
    rows = []
    race_venue = None
    first_race = None
    for venue_code in ('ST', 'HV'):
        url = (
            'https://racing.hkjc.com/racing/information/Chinese/Racing/'
            f'RaceCard.aspx?racedate={date_text}&Racecourse={venue_code}&RaceNo=1'
        )
        response = session.get(url, headers=headers, timeout=12)
        response.encoding = 'utf-8'
        first_race_table = find_racecard_table(response.text)
        if first_race_table is not None:
            race_venue = venue_code
            first_race = (first_race_table, response.text)
            break
    if first_race is None:
        raise ValueError('HKJC 暫未提供該日期的賽卡。請確認所選日期為香港賽馬日，且排位表已公布。')

    for race_no in range(1, 16):
        if race_no == 1:
            race_table, race_html = first_race
        else:
            url = (
                'https://racing.hkjc.com/racing/information/Chinese/Racing/'
                f'RaceCard.aspx?racedate={date_text}&Racecourse={race_venue}&RaceNo={race_no}'
            )
            response = session.get(url, headers=headers, timeout=12)
            response.encoding = 'utf-8'
            race_html = response.text
            race_table = find_racecard_table(race_html)
            if race_table is None:
                break

        distance_match = re.search(r'(\d{4})\s*米', race_html)
        distance = int(distance_match.group(1)) if distance_match else 1200
        surface = '泥地' if ('泥地' in race_html or '全天候' in race_html) else '草地'
        
        track_match = re.search(r'"([A-Z0-9\+]+)"\s*賽道', race_html)
        course_track = track_match.group(1) if track_match else ('泥地' if surface == '泥地' else 'A')

        columns = list(race_table.columns)
        horse_no_col = find_column(columns, ('馬號', '編號'), columns[0] if columns else None)
        horse_name_col = find_column(columns, ('馬名',))
        weight_col = find_column(columns, ('負磅', '磅'))
        jockey_col = find_column(columns, ('騎師',))
        trainer_col = find_column(columns, ('練馬師',))
        draw_col = find_column(columns, ('檔位',))
        if not all((horse_no_col, horse_name_col, weight_col, jockey_col, trainer_col, draw_col)):
            continue

        for _, row in race_table.iterrows():
            horse_no = str(row[horse_no_col]).strip()
            if re.fullmatch(r'\d+\.0', horse_no):
                horse_no = horse_no[:-2]
            if not horse_no.isdigit():
                continue
            horse_name = re.sub(r'\(.*?\)', '', str(row[horse_name_col])).strip()
            if not horse_name or horse_name.lower() == 'nan':
                continue
            rows.append({
                '馬季': season,
                '賽事編號': f'{race_day_id}-{race_no:02d}',
                '名次': '',
                '馬號': horse_no,
                '馬名': horse_name,
                '騎師': str(row[jockey_col]).strip(),
                '練馬師': str(row[trainer_col]).strip(),
                '實際負磅': str(row[weight_col]).strip(),
                '排位檔位': str(row[draw_col]).strip(),
                '獨贏賠率': 10.0,
                '位置賠率': 1.5,
                '距離': distance,
                '場地': surface,
                '賽道': course_track,
                'racecourse_code': race_venue,
            })
        time.sleep(0.25)

    if not rows:
        raise ValueError('HKJC 沒有回傳排位表。')

    result = pd.DataFrame(rows)
    return result.drop_duplicates(['賽事編號', '馬號'], keep='last').reset_index(drop=True)

# ==========================================
# 📂 預測資料 (支援多檔案即時切換與抓取)
# ==========================================
st.sidebar.header("📂 預測資料")

available_csvs = sorted([f.name for f in APP_DIR.glob("prediction*.csv")])

if not available_csvs:
    st.sidebar.caption("⚠️ 找不到任何 prediction*.csv 檔案。")
    selected_csv_name = "prediction.csv"
else:
    selected_csv_name = st.sidebar.selectbox(
        "📜 選擇 Repo 內的預測賽卡", 
        available_csvs,
        key="repo_csv_selector"
    )

# 偵測下拉選單是否被切換
if st.session_state.get('active_csv_name') != selected_csv_name:
    st.session_state['active_csv_name'] = selected_csv_name
    st.session_state.pop('fetched_prediction_df', None)
    st.session_state.pop('df_data', None)
    st.session_state.pop('prediction_signature', None)
    st.session_state.pop('baseline_odds_map', None)
    st.session_state.pop('baseline_locked_time', None)
    st.session_state['frozen_races'] = {}
    st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version', 0) + 1

SELECTED_CSV_PATH = APP_DIR / selected_csv_name

race_date_choice = st.sidebar.date_input('選擇賽事日期 (抓新排位用)', value=datetime.now().date(), key='racecard_date_choice')
if st.sidebar.button('🏇 抓取排位並載入預測', use_container_width=True, key='fetch_racecard_button'):
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
        st.session_state.pop('odds_last_status', None)
        st.session_state.pop('odds_last_error', None)
        st.session_state.pop('baseline_odds_map', None)
        st.session_state.pop('baseline_locked_time', None)
        st.session_state['frozen_races'] = {}
        st.sidebar.success(f'已載入 {len(fetched_card)} 匹馬。')
    except Exception as exc:
        st.sidebar.error(f'抓取排位失敗：{type(exc).__name__}: {exc}')

# 下載已抓取的預測 CSV 按鈕
if st.session_state.get('fetched_prediction_df') is not None:
    csv_bytes_pred = st.session_state['fetched_prediction_df'].to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.sidebar.download_button(
        label="⬇️ 下載已抓取的預測 CSV",
        data=csv_bytes_pred,
        file_name=f"prediction_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
        help="將含有最新檔位與賽道的預測表下載到電腦，放回 GitHub 即可永久使用！"
    )

current_fetched_card = st.session_state.get('fetched_prediction_df')
if current_fetched_card is not None:
    if st.sidebar.button('🔄 放棄抓取資料，改用上方選單 CSV', use_container_width=True):
        st.session_state.pop('fetched_prediction_df', None)
        st.session_state.pop('df_data', None)
        st.session_state.pop('prediction_signature', None)
        st.session_state.pop('baseline_odds_map', None)
        st.session_state.pop('baseline_locked_time', None)
        st.session_state['frozen_races'] = {}
        st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version', 0) + 1
        st.rerun()
else:
    st.sidebar.caption(f"目前使用選單檔案：{selected_csv_name}")

backtest_file = st.sidebar.file_uploader(
    "回測用：上傳已完成賽事 CSV（需要賽事編號、馬號、名次）",
    type=['csv'],
    key='backtest_results_upload',
    help='上傳已完賽的結果檔案，系統會用上方鎖定的【買時賠率】選馬，並用此檔案的【最終賠率】計算真實派彩！',
)

# ==========================================
# ⚙️ 策略、資金與算分引擎設定
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("⚙ 投注策略與算分引擎")
strategy_mode = st.sidebar.radio(
    "1️⃣ 推薦排序核心邏輯",
    ["⚡ 傳統純 EV 排序 (高賠率優先)", "🛡️ 抗噪穩定 (含步速+手風+賽道+排除回飛)", "🎯 純 AI 勝率/入位率最高 (命中率優先)"],
    index=0,
    help="【傳統純 EV】完全依賴 EV 高低排序（重現 1004 贏 +142% 邏輯）；【抗噪穩定】結合落飛資金、預期步速、騎練近期手風與賽道檔位偏差進行綜合算分，並過濾臨場大幅回飛馬；【純 AI 機率】無視賠率高低，單純挑選模型預測機率最高的馬匹！"
)
use_dynamic_form = st.sidebar.checkbox(
    "2️⃣ 將「上仗名次與同程勝率」納入 AI 算分",
    value=False,
    help="預設不勾選（🎯 1004 爆冷模式）：AI 算分時維持訓練模型習慣的固定常數 (6.0 / 0.08)，不因馬匹上仗落敗而扣分，專抓反彈大冷門！勾選後則將真實近況納入算分（無論勾選與否，畫面表格都會顯示真實近況供肉眼參考）。"
)
min_ev = st.sidebar.slider("最小期望值 (EV 門檻)", 0.0, 6.0, 2.5, 0.1)
min_odds = st.sidebar.number_input("最低獨贏賠率", min_value=1.0, max_value=50.0, value=3.0, help="⚠️ 提示：若使用【🎯 純 AI 機率最高】模式尋找熱門穩膽，建議將此數值調低至 1.0，以免錯殺優質大熱門！(位置彩池會自動將此門檻除以3)")
max_odds = st.sidebar.number_input("最高獨贏賠率", min_value=1.0, max_value=100.0, value=25.0)

st.sidebar.markdown("---")
st.sidebar.header("💰 資金管理 (凱利注碼精算)")
total_bankroll = st.sidebar.number_input("💵 單日總本金預算 (HKD)", min_value=100, max_value=1000000, value=5000, step=500)
kelly_fraction = st.sidebar.slider("🎛️ 凱利保守係數", 0.05, 1.0, 0.15, 0.05)

# ==========================================
# 🛠️ 歷史賽果 15 大特徵資料抓取工具 (防彈升級版 + 賽道抓取)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("🛠️ 歷史賽果 15 大特徵抓取器")
st.sidebar.caption("輸入指定賽事日期，自動抓取該日賽果並產出可供模型訓練與回測對答案的 CSV。")

tool_date_choice = st.sidebar.date_input("選擇目標賽事日期", value=datetime.now().date(), key="tool_date_input")
tool_season = st.sidebar.text_input("輸入馬季 (例如 23/24 或 26/27)", value="26/27")

if st.sidebar.button("📥 抓取並產出完賽 CSV", use_container_width=True, key="run_tool_button"):
    date_str_tool = tool_date_choice.strftime('%Y/%m/%d')
    date_for_id_tool = tool_date_choice.strftime('%Y%m%d')
    
    all_races_data_tool = []
    headers_tool = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    success_count = 0
    total_horses = 0
    
    for race_no in range(1, 13):
        status_text.text(f"正在抓取第 {race_no} 場資料...")
        url = f"https://racing.hkjc.com/racing/information/Chinese/Racing/LocalResults.aspx?RaceDate={date_str_tool}&RaceNo={race_no}"
        try:
            resp = requests.get(url, headers=headers_tool, timeout=15)
            resp.encoding = 'utf-8' 
            html_text = resp.text
            
            if "沒有相關資料" in html_text or "No information" in html_text:
                progress_bar.progress(race_no / 12)
                continue

            distance = 1200
            dist_match = re.search(r'(\d{4})\s*米', html_text)
            if dist_match:
                distance = int(dist_match.group(1))
            surface_type = "草地" if "草地" in html_text else "泥地"
            
            track_match = re.search(r'"([A-Z0-9\+]+)"\s*賽道', html_text)
            course_track = track_match.group(1) if track_match else ('泥地' if surface_type == '泥地' else 'A')
            
            tables = pd.read_html(StringIO(html_text))
            target_df = None
            
            for t in tables:
                if isinstance(t.columns, pd.MultiIndex):
                    t.columns = ['_'.join(map(str, col)) for col in t.columns]
                    
                check_str = "".join([str(c) for c in t.columns])
                for row_idx in range(min(5, len(t))):
                    check_str += "".join([str(c) for c in t.iloc[row_idx].values])
                    
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
                
                rank_col = next((col for col in cols if '名次' in col), None)
                horse_no_col = next((col for col in cols if '馬號' in col or '編號' in col), None)
                horse_name_col = next((col for col in cols if '馬名' in col), None)
                jockey_col = next((col for col in cols if '騎師' in col), None)
                trainer_col = next((col for col in cols if '練馬師' in col), None)
                odds_col = next((col for col in cols if '獨贏' in col), None)
                weight_col = next((col for col in cols if '負磅' in col or '實際負磅' in col), None)
                draw_col = next((col for col in cols if '檔位' in col), None)
                
                if rank_col and horse_no_col:
                    added_in_this_race = 0
                    for idx, row in target_df.iterrows():
                        rank = str(row.get(rank_col, '')).strip()
                        horse_no = str(row.get(horse_no_col, '')).strip()
                        
                        if horse_no.endswith('.0'): horse_no = horse_no[:-2]
                        
                        if horse_no.isdigit():
                            horse_name_raw = str(row.get(horse_name_col, ''))
                            horse_name = horse_name_raw.split('(')[0].strip() if pd.notna(horse_name_raw) else ""
                            
                            odds_val = row[odds_col] if odds_col and pd.notna(row[odds_col]) else 0.0
                            actual_weight = str(row[weight_col]).strip() if weight_col and pd.notna(row[weight_col]) else '120'
                            draw_pos = str(row[draw_col]).strip() if draw_col and pd.notna(row[draw_col]) else '7'
                            
                            all_races_data_tool.append({
                                '馬季': tool_season,
                                '賽事編號': f"{date_for_id_tool}-{race_no:02d}",
                                '名次': rank,
                                '馬號': horse_no,
                                '馬名': horse_name,
                                '騎師': str(row.get(jockey_col, '')).strip(),
                                '練馬師': str(row.get(trainer_col, '')).strip(),
                                '實際負磅': actual_weight,
                                '排位檔位': draw_pos,
                                '獨贏賠率': odds_val,
                                '距離': distance,                
                                '場地': surface_type,
                                '賽道': course_track,
                                'horse_surface_win_rate': 0.08,   
                                'horse_dist_win_rate': 0.08
                            })
                            added_in_this_race += 1
                    
                    if added_in_this_race > 0:
                        success_count += 1
                        total_horses += added_in_this_race
            
            progress_bar.progress(race_no / 12)
            time.sleep(1.0)
            
        except Exception as e:
            continue
            
    progress_bar.empty()
    status_text.empty()
    
    if not all_races_data_tool:
        st.sidebar.error("❌ 該日期找不到任何賽事資料，或賽事已取消。")
    else:
        df_tool = pd.DataFrame(all_races_data_tool)
        df_tool['獨贏賠率'] = pd.to_numeric(df_tool['獨贏賠率'], errors='coerce').fillna(0.0)
        
        output_cols = ['馬季', '賽事編號', '名次', '馬號', '馬名', '騎師', '練馬師', '實際負磅', '排位檔位', '獨贏賠率', '距離', '場地', '賽道', 'horse_surface_win_rate', 'horse_dist_win_rate']
        df_tool = df_tool[[col for col in output_cols if col in df_tool.columns]]
        
        csv_filename = f"hkjc_data_15features_{date_for_id_tool}.csv"
        csv_data = df_tool.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        
        st.sidebar.success(f"🎉 成功抓取 {success_count} 場，共 {total_horses} 匹真實賽果資料！")
        st.sidebar.download_button(
            label="⬇️ 下載產出的完賽 CSV 檔",
            data=csv_data,
            file_name=csv_filename,
            mime="text/csv",
            use_container_width=True
        )
        # ==========================================
# 🔄 歷史大表 CSV 一鍵合併更新工具
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("🔄 歷史大表合併工具")
st.sidebar.caption("上傳剛抓下的完賽 CSV，系統會自動與現有歷史大表合併並去除重複紀錄。")

new_csv_files = st.sidebar.file_uploader(
    "上傳新完賽 CSV (可多選)", 
    type=['csv'], 
    accept_multiple_files=True,
    key="merge_csv_uploader"
)

if new_csv_files and st.sidebar.button("⚡ 合併並產出最新歷史大表", use_container_width=True):
    hist_path = APP_DIR / 'hkjc_all_seasons_features.csv'
    dfs_to_merge = []
    
    # 1. 讀取現有的歷史大表
    if hist_path.exists():
        try:
            base_df = pd.read_csv(hist_path, encoding='utf-8-sig', low_memory=False)
        except Exception:
            base_df = pd.read_csv(hist_path, encoding='cp950', low_memory=False)
        dfs_to_merge.append(base_df)
        old_len = len(base_df)
    else:
        old_len = 0
        st.sidebar.warning("⚠️ 找不到現有大表，將僅合併上傳的檔案。")
        
    # 2. 讀取所有上傳的新 CSV
    for f in new_csv_files:
        try:
            sub_df = pd.read_csv(f, encoding='utf-8-sig')
        except Exception:
            f.seek(0)
            sub_df = pd.read_csv(f, encoding='cp950')
        dfs_to_merge.append(sub_df)
        
    if dfs_to_merge:
        # 3. 合併、去重並按賽事編號排序
        merged_df = pd.concat(dfs_to_merge, ignore_index=True)
        
        # 統一賽事編號與馬號格式，確保去重精準
        if '賽事編號' in merged_df.columns and '馬號' in merged_df.columns:
            merged_df['_r_key'] = merged_df['賽事編號'].astype(str).str.strip()
            merged_df['_h_key'] = merged_df['馬號'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            merged_df = merged_df.drop_duplicates(subset=['_r_key', '_h_key'], keep='last')
            merged_df = merged_df.sort_values(by=['_r_key']).drop(columns=['_r_key', '_h_key']).reset_index(drop=True)
            
        new_len = len(merged_df)
        added_rows = new_len - old_len
        
        merged_bytes = merged_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.sidebar.success(f"✅ 合併完成！原 {old_len:,} 筆 ➔ 現 {new_len:,} 筆 (淨新增 {added_rows:,} 筆)")
        st.sidebar.download_button(
            label="⬇️ 下載最新 hkjc_all_seasons_features.csv",
            data=merged_bytes,
            file_name="hkjc_all_seasons_features.csv",
            mime="text/csv",
            use_container_width=True
        )
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
            return None, None, f"GraphQL 錯誤：{payload['errors']}"

        meetings = (payload.get("data") or {}).get("raceMeetings") or []
        if not meetings:
            return None, None, "找不到該日期／場地的賽事資料（可能尚未開盤或日期錯誤）。"

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
                    odds_value = str(node.get("oddsValue")).strip()
                    
                    if horse_no and odds_value not in ("None", "", "SCR"):
                        try:
                            market[pool_type][horse_no] = float(odds_value)
                        except ValueError:
                            pass

        if not odds_by_race:
            return None, None, "賽事有回應，但目前尚無可用 WIN／PLACE 賠率。"
        return odds_by_race, (max(timestamps) if timestamps else None), None

    except requests.RequestException as exc:
        return None, None, f"連線錯誤：{exc}"
    except (ValueError, TypeError, KeyError) as exc:
        return None, None, f"解析回應錯誤：{exc}"

# ==========================================
# 決定讀取哪一份賽卡資料
# ==========================================
fetched_prediction_df = st.session_state.get('fetched_prediction_df')
hk_timezone = timezone(timedelta(hours=8))

if fetched_prediction_df is not None:
    df_raw = fetched_prediction_df.copy()
    fetched_date = st.session_state.get('fetched_prediction_date', race_date_choice)
    source_label = f"HKJC {fetched_date:%Y-%m-%d} 官方排位抓取"
    prediction_signature = st.session_state.get('fetched_prediction_signature', source_label)
    st.sidebar.success(f'預測來源：{source_label}（{len(df_raw)} 匹）')
elif SELECTED_CSV_PATH.is_file():
    try:
        df_raw = pd.read_csv(SELECTED_CSV_PATH, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df_raw = pd.read_csv(SELECTED_CSV_PATH, encoding='cp950')
    source_label = f"GitHub repo：{SELECTED_CSV_PATH.name}"
    prediction_signature = f"{SELECTED_CSV_PATH.name}:{SELECTED_CSV_PATH.stat().st_mtime_ns}:{SELECTED_CSV_PATH.stat().st_size}"

    if df_raw.empty:
        st.error(f"預測 CSV 是空檔：{SELECTED_CSV_PATH.name}")
        st.stop()
    st.sidebar.success(f"已從 repo 載入：{SELECTED_CSV_PATH.name}（{len(df_raw)} 匹）")
    st.success(f"✅ 目前顯示預測賽事檔案：`{SELECTED_CSV_PATH.name}`（共 {len(df_raw)} 匹馬）")
else:
    st.error(f"找不到 repo 預測 CSV：{SELECTED_CSV_PATH.name}。")
    st.stop()

auto_odds_status = None
if fetched_prediction_df is not None and st.session_state.pop('pending_auto_odds_fetch', False):
    odds_date = st.session_state.get('fetched_prediction_date', race_date_choice).strftime('%Y-%m-%d')
    odds_venue = str(df_raw['racecourse_code'].iloc[0])
    with st.spinner('排位已載入，正在嘗試取得該日 WIN／PLACE 賠率…'):
        odds_by_race, odds_updated_at, odds_error = fetch_live_odds(odds_date, odds_venue)
    if odds_by_race:
        win_count = place_count = 0
        for idx, runner in df_raw.iterrows():
            race_no = extract_race_no(runner.get('賽事編號', ''))
            horse_no = normalize_horse_no(runner.get('馬號', ''))
            race_market = odds_by_race.get(race_no, {})
            win_price = race_market.get('WIN', {}).get(horse_no)
            place_price = race_market.get('PLA', {}).get(horse_no)
            if win_price is not None and win_price > 0:
                df_raw.at[idx, '獨贏賠率'] = win_price
                win_count += 1
            if place_price is not None and place_price > 0:
                df_raw.at[idx, '位置賠率'] = place_price
                place_count += 1
        st.session_state['df_data'] = df_raw.copy()
        st.session_state['fetched_prediction_df'] = df_raw.copy()
        
        base_map = {}
        for _, r in df_raw.iterrows():
            k = f"{normalize_race_id(r['賽事編號'])}_{normalize_horse_no(r['馬號'])}"
            base_map[k] = float(r['獨贏賠率'])
        st.session_state['baseline_odds_map'] = base_map
        st.session_state['baseline_locked_time'] = datetime.now(hk_timezone).strftime("%H:%M:%S")
        
        auto_odds_status = f'已自動更新即時賠率並鎖定初始基準：WIN {win_count} 匹／PLACE {place_count} 匹'
    else:
        auto_odds_status = f'目前未能取得已開出的官方賠率。'

if auto_odds_status:
    if auto_odds_status.startswith('已自動'):
        st.sidebar.success(auto_odds_status)
    else:
        st.sidebar.warning(auto_odds_status)

# 確保賠率、賽道與跑法欄位存在
if '獨贏賠率' not in df_raw.columns:
    df_raw['獨贏賠率'] = 10.0
if '位置賠率' not in df_raw.columns:
    win_values = pd.to_numeric(df_raw['獨贏賠率'], errors='coerce').fillna(10.0)
    df_raw['位置賠率'] = 1.0 + (win_values - 1.0) / 3.2
if '賽道' not in df_raw.columns:
    df_raw['賽道'] = 'A'
if '跑法' not in df_raw.columns:
    df_raw['跑法'] = '居中'

# 每次切換檔案時，將版本號 +1 強制刷新表格，並初始化基準賠率
if 'df_data' not in st.session_state or st.session_state.get('prediction_signature') != prediction_signature:
    st.session_state['df_data'] = df_raw.copy()
    st.session_state['prediction_signature'] = prediction_signature
    st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version', 0) + 1
    
    base_map = {}
    for _, r in df_raw.iterrows():
        k = f"{normalize_race_id(r['賽事編號'])}_{normalize_horse_no(r['馬號'])}"
        val = pd.to_numeric(r['獨贏賠率'], errors='coerce')
        base_map[k] = float(val) if pd.notna(val) and val > 0 else 10.0
    st.session_state['baseline_odds_map'] = base_map
    st.session_state['baseline_locked_time'] = "初始載入值"

# 防呆機制：確保暫存資料中也包含賽道與跑法欄位，消除 KeyError
if '賽道' not in st.session_state['df_data'].columns:
    st.session_state['df_data']['賽道'] = 'A'
if '跑法' not in st.session_state['df_data'].columns:
    st.session_state['df_data']['跑法'] = '居中'

# ==========================================
# 互動式臨場賠率與跑法輸入面板
# ==========================================
st.info("👇 以下是當前載入的賽事資料。你可以點擊表格修改「跑法」與「賠率」，或透過下方的更新面板抓取即時賠率與凍結單場盤口。")

edit_columns = ['賽事編號', '馬號', '馬名', '排位檔位', '跑法', '賽道', '獨贏賠率', '位置賠率']
for c in edit_columns:
    if c not in st.session_state['df_data'].columns:
        st.session_state['df_data'][c] = ''
df_editable = st.session_state['df_data'][edit_columns].copy()

editor_key = f"odds_editor_{selected_csv_name}_{st.session_state.get('odds_editor_version', 0)}"
edited_df = st.data_editor(
    df_editable,
    disabled=['賽事編號', '馬號', '馬名', '排位檔位', '賽道'],
    column_config={
        "跑法": st.column_config.SelectboxColumn("跑法", options=["放頭", "跟前", "居中", "後上"], required=True)
    },
    use_container_width=True,
    hide_index=True,
    key=editor_key
)

# 同步手動修改的賠率與跑法回 session_state['df_data']
st.session_state['df_data']['獨贏賠率'] = pd.to_numeric(edited_df['獨贏賠率'], errors='coerce').fillna(10.0)
st.session_state['df_data']['位置賠率'] = pd.to_numeric(edited_df['位置賠率'], errors='coerce').fillna(1.5)
st.session_state['df_data']['跑法'] = edited_df['跑法']

df = st.session_state['df_data'].copy()

# 側邊欄提供隨時下載「包含目前已凍結買時賠率與跑法」的完整賽卡快照
sample_id_for_dl = str(df['賽事編號'].iloc[0]).strip() if len(df) else ""
dl_date_match = re.search(r"(20\d{6})", sample_id_for_dl)
dl_date_tag = dl_date_match.group(1) if dl_date_match else datetime.now(hk_timezone).strftime("%Y%m%d")
snapshot_csv_bytes = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
st.sidebar.download_button(
    f'📸 下載【買時賠率】快照 (prediction_{dl_date_tag}.csv)',
    data=snapshot_csv_bytes,
    file_name=f'prediction_{dl_date_tag}.csv',
    mime='text/csv',
    use_container_width=True,
    help='將目前畫面上的賠率與設定（包含你逐場凍結的買時賠率與跑法）完整打包下載，放到 GitHub 即可隨時重現 100% 真實買時回測！'
)

# === 計算早盤 vs 臨場賠率變動 (落飛追蹤) ===
baseline_map = st.session_state.get('baseline_odds_map', {})
df['_horse_uid'] = df.apply(lambda r: f"{normalize_race_id(r['賽事編號'])}_{normalize_horse_no(r['馬號'])}", axis=1)
df['基準賠率'] = df['_horse_uid'].map(baseline_map).fillna(df['獨贏賠率'])
df['基準賠率'] = pd.to_numeric(df['基準賠率'], errors='coerce').fillna(df['獨贏賠率'])

df['odds_drop_pct'] = ((df['基準賠率'] - df['獨贏賠率']) / df['基準賠率'].replace(0, np.nan)).fillna(0.0) * 100.0

def get_odds_move_badge(pct):
    if pct >= 20.0:
        return f"🔥 大落飛 (-{pct:.0f}%)"
    elif pct >= 10.0:
        return f"📉 落飛 (-{pct:.0f}%)"
    elif pct <= -20.0:
        return f"🥶 回飛 (+{abs(pct):.0f}%)"
    else:
        return "➖ 穩定"

df['資金流向'] = df['odds_drop_pct'].map(get_odds_move_badge)

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

if '距離' not in df.columns:
    df['距離'] = 1200
df['距離'] = pd.to_numeric(df['距離'], errors='coerce').fillna(1200).astype(int)

if '場地' not in df.columns:
    df['場地'] = '草地'

# === 補全真實歷史數據 (視覺與算分雙軌設計 + 近期手風) ===
hist_stats = load_historical_stats()
if hist_stats is not None:
    (
        jockey_stats, trainer_stats, horse_stats, combo_stats,
        horse_last_rank_stats, horse_surface_stats, horse_dist_stats
    ) = hist_stats
    
    df['_clean_jockey'] = df['騎師'].astype(str).str.replace(r'\(.*?\)', '', regex=True).str.replace(' ', '')
    df['_clean_trainer'] = df['練馬師'].astype(str).str.replace(' ', '')
    df['_clean_horse'] = df['馬名'].astype(str).str.replace(r'\(.*?\)', '', regex=True).str.replace(' ', '')
    df['_clean_surface'] = df['場地'].astype(str).str.strip()
    df['_clean_dist'] = df['距離'].astype(int)
    
    df = df.merge(jockey_stats, on='_clean_jockey', how='left')
    df = df.merge(trainer_stats, on='_clean_trainer', how='left')
    df = df.merge(horse_stats, on='_clean_horse', how='left')
    df = df.merge(combo_stats, on=['_clean_jockey', '_clean_trainer'], how='left')
    df = df.merge(horse_last_rank_stats, on='_clean_horse', how='left')
    df = df.merge(horse_surface_stats, on=['_clean_horse', '_clean_surface'], how='left')
    df = df.merge(horse_dist_stats, on=['_clean_horse', '_clean_dist'], how='left')
    
    df['jockey_win_rate'] = df['hist_jockey_win_rate'].fillna(0.08)
    df['trainer_win_rate'] = df['hist_trainer_win_rate'].fillna(0.08)
    df['recent_jockey_win_rate'] = df.get('recent_jockey_win_rate', df['jockey_win_rate']).fillna(df['jockey_win_rate'])
    df['recent_trainer_win_rate'] = df.get('recent_trainer_win_rate', df['trainer_win_rate']).fillna(df['trainer_win_rate'])
    df['horse_win_rate'] = df['hist_horse_win_rate'].fillna(0.05)
    df['combo_win_rate'] = df['hist_combo_win_rate'].fillna(0.05)
    
    # 儲存真實近況與路程數據供畫面表格顯示 (不受算分開關影響)
    df['real_last_rank'] = df['hist_horse_last_rank'].fillna(6.5)
    df['real_surface_win_rate'] = df['hist_horse_surface_win_rate'].fillna(df['horse_win_rate']).fillna(0.06)
    df['real_dist_win_rate'] = df['hist_horse_dist_win_rate'].fillna(df['horse_win_rate']).fillna(0.06)
    
    # 🚀 核心切換：決定餵給 AI 模型算分的特徵值
    if use_dynamic_form:
        df['horse_last_rank'] = df['real_last_rank']
        df['horse_surface_win_rate'] = df['real_surface_win_rate']
        df['horse_dist_win_rate'] = df['real_dist_win_rate']
    else:
        # 1004 爆冷模式：維持模型訓練時習慣的固定常數，不因上仗落敗而扣掉黑馬分數！
        df['horse_last_rank'] = 6.0
        df['horse_surface_win_rate'] = 0.08
        df['horse_dist_win_rate'] = 0.08
    
    df.drop(columns=[
        '_clean_jockey', '_clean_trainer', '_clean_horse', '_clean_surface', '_clean_dist',
        'hist_jockey_win_rate', 'hist_trainer_win_rate', 'hist_horse_win_rate', 'hist_combo_win_rate',
        'hist_horse_last_rank', 'hist_horse_surface_win_rate', 'hist_horse_dist_win_rate'
    ], inplace=True, errors='ignore')
else:
    st.sidebar.error("❌ 未能讀取歷史大表，目前正在使用預設值！")
    df['jockey_win_rate'] = df.get('jockey_win_rate', 0.08)
    df['trainer_win_rate'] = df.get('trainer_win_rate', 0.08)
    df['recent_jockey_win_rate'] = df.get('recent_jockey_win_rate', 0.08)
    df['recent_trainer_win_rate'] = df.get('recent_trainer_win_rate', 0.08)
    df['combo_win_rate'] = df.get('combo_win_rate', 0.05)
    df['horse_win_rate'] = df.get('horse_win_rate', 0.05)
    df['real_last_rank'] = 6.5
    df['real_surface_win_rate'] = 0.08
    df['real_dist_win_rate'] = 0.08
    df['horse_last_rank'] = 6.0
    df['horse_surface_win_rate'] = 0.08
    df['horse_dist_win_rate'] = 0.08

# 🚀 2. 雙渦輪引擎同時預測 (Win Engine + Place Engine)
feature_cols = FEATURE_COLS_15
for col in feature_cols:
    if col not in df.columns:
        df[col] = 0.0

X_predict = df[feature_cols].fillna(0)
df['pred_win_prob'] = model_win.predict_proba(X_predict)[:, 1]
if model_place is not None:
    df['pred_place_prob'] = model_place.predict_proba(X_predict)[:, 1]
else:
    df['pred_place_prob'] = df['pred_win_prob']

df['win_ev'] = df['pred_win_prob'] * df['獨贏賠率']
df['place_ev'] = df['pred_place_prob'] * df['位置賠率']
df['ev'] = df['win_ev']  # 保持向下相容

# 🚀 3. 計算賽道移欄與檔位偏差懲罰
def calculate_track_bias(row):
    penalty = 0.0
    if row['場地'] == '草地':
        draw = int(row['排位檔位'])
        track = str(row.get('賽道', 'A')).upper()
        dist = int(row['距離'])
        rc = str(row.get('racecourse_code', 'ST'))
        if track in ['C', 'C+3', 'B+2', 'C+2']:
            if draw >= 11: penalty = -0.15
            elif draw >= 9: penalty = -0.05
        if rc == 'ST' and dist == 1000:
            if draw >= 10: penalty = +0.10
            elif draw <= 4: penalty = -0.10
    return penalty
df['track_bias_adj'] = df.apply(calculate_track_bias, axis=1)

# 🚀 4. 計算近期手風 (Hot Hand) 戰術加成
def calculate_hot_hand_adj(row):
    adj = 0.0
    j_diff = row['recent_jockey_win_rate'] - row['jockey_win_rate']
    t_diff = row['recent_trainer_win_rate'] - row['trainer_win_rate']
    if j_diff >= 0.04: adj += 0.05
    elif j_diff <= -0.04: adj -= 0.05
    if t_diff >= 0.04: adj += 0.05
    elif t_diff <= -0.04: adj -= 0.05
    return adj
df['hot_hand_adj'] = df.apply(calculate_hot_hand_adj, axis=1)

# 🚀 5. 計算預期步速 (Pace Scenario) 戰術加成
def calculate_pace_adj(df_in):
    df_in['pace_adj'] = 0.0
    df_in['race_pace'] = '🐎 正常'
    for race_id, group in df_in.groupby('賽事編號'):
        front_runners = sum(group['跑法'] == '放頭')
        if front_runners >= 3:
            pace = "fast"
            pace_label = '🔥 快步速'
        elif front_runners <= 1:
            pace = "slow"
            pace_label = '🐢 慢步速'
        else:
            pace = "normal"
            pace_label = '🐎 正常'
            
        for idx, row in group.iterrows():
            style = row['跑法']
            adj = 0.0
            if pace == "fast":
                if style == '後上': adj = 0.08
                elif style == '放頭': adj = -0.06
            elif pace == "slow":
                if style == '放頭': adj = 0.08
                elif style == '後上': adj = -0.05
            df_in.at[idx, 'pace_adj'] = adj
            df_in.at[idx, 'race_pace'] = pace_label
    return df_in
df = calculate_pace_adj(df)

# 6. 計算雙引擎「抗噪穩定綜合分數」 (結合落飛資金 + 賽道偏差 + 騎練手風 + 預期步速)
drop_bonus = np.clip(df['odds_drop_pct'] / 100.0, -0.35, 0.30)
total_tactical_adj = 1.0 + drop_bonus + df['track_bias_adj'] + df['hot_hand_adj'] + df['pace_adj']
df['smart_score_win'] = df['pred_win_prob'] * total_tactical_adj
df['smart_score_place'] = df['pred_place_prob'] * total_tactical_adj
df['smart_score'] = df['smart_score_win']

# 💰 7. 雙重凱利注碼精算 (獨贏注碼 & 位置注碼)
safe_win_odds = np.maximum(df['獨贏賠率'], 1.01)
df['kelly_pct_win'] = np.clip((df['win_ev'] - 1.0) / (safe_win_odds - 1.0), 0.0, 1.0)
df['suggested_stake_win'] = np.floor((total_bankroll * df['kelly_pct_win'] * kelly_fraction) / 10) * 10
df['suggested_stake'] = df['suggested_stake_win']

safe_place_odds = np.maximum(df['位置賠率'], 1.01)
df['kelly_pct_place'] = np.clip((df['place_ev'] - 1.0) / (safe_place_odds - 1.0), 0.0, 1.0)
df['suggested_stake_place'] = np.floor((total_bankroll * df['kelly_pct_place'] * kelly_fraction) / 10) * 10

# 🚀 核心雙軌過濾與排序邏輯 (支援 bet_type 分流)
def filter_and_sort_group(group, mode, min_ev_val, min_odds_val, max_odds_val, bet_type="win"):
    if bet_type == "win":
        cond = (group['win_ev'] >= min_ev_val) & (group['獨贏賠率'] >= min_odds_val) & (group['獨贏賠率'] <= max_odds_val)
        if mode.startswith("🛡️"):
            cond = cond & (group['odds_drop_pct'] > -25.0)
            sort_col = 'smart_score_win'
        elif mode.startswith("🎯"):
            sort_col = 'pred_win_prob'
        else:
            sort_col = 'win_ev'
    else:
        # 位置賠率較低，將獨贏最低賠率門檻除以 3 作為位置賠率下限
        min_pla_odds = max(1.01, min_odds_val / 3.0)
        cond = (group['place_ev'] >= min_ev_val) & (group['位置賠率'] >= min_pla_odds)
        if mode.startswith("🛡️"):
            cond = cond & (group['odds_drop_pct'] > -25.0)
            sort_col = 'smart_score_place'
        elif mode.startswith("🎯"):
            sort_col = 'pred_place_prob'
        else:
            sort_col = 'place_ev'
            
    return group[cond].sort_values(by=sort_col, ascending=False).reset_index(drop=True)

# 回測使用手動上傳的完賽資料
df_backtest = df.copy()
backtest_ready = False
if backtest_file is not None:
    try:
        try:
            result_raw = pd.read_csv(backtest_file, encoding='utf-8-sig')
        except UnicodeDecodeError:
            backtest_file.seek(0)
            result_raw = pd.read_csv(backtest_file, encoding='cp950')
        result_source_label = '手動上傳賽果'
        
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
                # df_backtest = df_backtest[df_backtest['_result_rank'].notna()].copy()
                st.sidebar.success(f"✅ {result_source_label}已配對：{matched} 匹")
            else:
                st.sidebar.warning("未能按賽事編號＋馬號配對到名次。")
    except Exception as exc:
        st.sidebar.error(f"讀取回測 CSV 失敗：{type(exc).__name__}: {exc}")

st.markdown("---")
tab1, tab2 = st.tabs(["🎯 各場次預測推薦與單場凍結", "📈 歷史回測 (買時選馬 vs 最終派彩)"])

# ---------------- 分頁 1: 賽前預測 ----------------
with tab1:
    st.subheader("⚡ 臨場賠率更新中心與單場盤口凍結")
    
    col_v, col_d, col_r, col_t = st.columns([1.2, 1.8, 1.2, 1.8])
    fetched_venue = str(df_raw['racecourse_code'].iloc[0]) if 'racecourse_code' in df_raw.columns else ''
    venue_default_index = 1 if fetched_venue == 'ST' else 0
    venue_input = col_v.selectbox("賽事場地", ["HV (跑馬地)", "ST (沙田)"], index=venue_default_index)
    venue_code = "HV" if venue_input.startswith("HV") else "ST"

    sample_id = str(df_raw['賽事編號'].iloc[0]).strip() if len(df_raw) else ""
    date_match = re.search(r"(20\d{6})", sample_id)
    if date_match:
        raw_date = date_match.group(1)
        auto_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
    else:
        auto_date = datetime.now().strftime("%Y-%m-%d")
    api_date = col_d.text_input("API 查詢日期 (YYYY-MM-DD)", value=auto_date, key=f"api_date_{prediction_signature}")

    race_series = df_raw['賽事編號'].map(extract_race_no)
    races_available = sorted(int(x) for x in race_series.dropna().unique())
    if races_available:
        target_race = col_r.selectbox("當前操作／查看場次", races_available)
    else:
        target_race = col_r.number_input("當前操作／查看場次", min_value=1, max_value=15, value=1, step=1)

    target_time = col_t.time_input("預定開跑時間", value=datetime.strptime("18:45", "%H:%M").time())
    
    try:
        race_date = datetime.strptime(api_date.strip(), "%Y-%m-%d").date()
    except ValueError:
        st.error("API 日期格式必須是 YYYY-MM-DD。")
        st.stop()

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
        timerElement.textContent = "已到預定開跑時間";
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

    with st.expander("🔊 HKJC 官方賽日收音機（現場評述）", expanded=False):
        st.caption("選擇官方頁面上的語言並按播放。")
        st.markdown("[在新分頁開啟 HKJC 官方賽日收音機](https://racing.hkjc.com/zh-hk/showcase/live)")
        components.iframe("https://racing.hkjc.com/zh-hk/showcase/live", height=620, scrolling=True)

    # 控制列 1：自動更新與基準鎖定
    auto_col, interval_col, button_col, lock_col = st.columns([1.3, 1.3, 2.0, 2.0])
    auto_refresh = auto_col.checkbox("自動更新未凍結場次", value=False)
    refresh_seconds = interval_col.selectbox(
        "更新間隔（秒）", [10, 30, 60, 120, 300], index=1, disabled=not auto_refresh
    )
    manual_refresh = button_col.button("🔄 立即更新未凍結場次賠率", use_container_width=True)
    lock_baseline = lock_col.button("📌 鎖定當前賠率為早盤基準", use_container_width=True, help="在開跑前 15~20 分鐘按一下鎖定早盤賠率，之後更新賠率時即可自動計算落飛幅度！")

    # 控制列 2：單場買時盤口凍結器
    f_col1, f_col2, f_col3 = st.columns([2.2, 1.8, 2.6])
    is_target_frozen = int(target_race) in st.session_state['frozen_races']
    if not is_target_frozen:
        if f_col1.button(f"🔒 凍結第 {target_race} 場【買時賠率】 (下注後點擊)", use_container_width=True, type="primary"):
            st.session_state['frozen_races'][int(target_race)] = datetime.now(hk_timezone).strftime("%H:%M:%S")
            st.rerun()
    else:
        frozen_t = st.session_state['frozen_races'][int(target_race)]
        if f_col1.button(f"🔓 解除第 {target_race} 場凍結 (已鎖於 {frozen_t})", use_container_width=True):
            st.session_state['frozen_races'].pop(int(target_race), None)
            st.rerun()

    if f_col2.button("🔓 全部場次解除凍結", use_container_width=True, disabled=len(st.session_state['frozen_races']) == 0):
        st.session_state['frozen_races'] = {}
        st.rerun()

    frozen_list_str = ", ".join([f"R{r}({t})" for r, t in sorted(st.session_state['frozen_races'].items())]) if st.session_state['frozen_races'] else "尚無凍結場次"
    f_col3.info(f"🔒 已凍結買時盤口：{frozen_list_str}")

    if lock_baseline:
        base_map = {}
        for _, r in df.iterrows():
            k = f"{normalize_race_id(r['賽事編號'])}_{normalize_horse_no(r['馬號'])}"
            base_map[k] = float(r['獨贏賠率'])
        st.session_state['baseline_odds_map'] = base_map
        st.session_state['baseline_locked_time'] = datetime.now(hk_timezone).strftime("%H:%M:%S")
        st.rerun()

    if auto_refresh:
        st_autorefresh(interval=int(refresh_seconds * 1000), key="hkjc_live_odds_autorefresh")

    if auto_refresh or manual_refresh:
        with st.spinner("正在讀取當日 WIN／PLACE 即時賠率（自動跳過已凍結場次）…"):
            live_by_race, server_updated_at, error = fetch_live_odds(api_date, venue_code)

        if live_by_race:
            df_temp = st.session_state['df_data'].copy()
            row_races = df_temp['賽事編號'].map(extract_race_no)
            row_horses = df_temp['馬號'].map(normalize_horse_no)
            updated_races_count = 0
            skipped_frozen = []

            for race_no, markets in live_by_race.items():
                r_int = int(race_no)
                if r_int in st.session_state['frozen_races']:
                    skipped_frozen.append(f"R{r_int}")
                    continue
                updated_races_count += 1
                race_mask = row_races == r_int
                for horse_no, value in markets['WIN'].items():
                    horse_mask = race_mask & (row_horses == horse_no)
                    df_temp.loc[horse_mask, '獨贏賠率'] = value
                for horse_no, value in markets['PLA'].items():
                    horse_mask = race_mask & (row_horses == horse_no)
                    df_temp.loc[horse_mask, '位置賠率'] = value

            if st.session_state.get('baseline_locked_time') == "初始載入值":
                base_map = {}
                for _, r in df_temp.iterrows():
                    k = f"{normalize_race_id(r['賽事編號'])}_{normalize_horse_no(r['馬號'])}"
                    base_map[k] = float(r['獨贏賠率'])
                st.session_state['baseline_odds_map'] = base_map
                st.session_state['baseline_locked_time'] = f"首次自動鎖定 ({datetime.now(hk_timezone).strftime('%H:%M:%S')})"

            st.session_state['df_data'] = df_temp
            st.session_state['odds_editor_version'] = st.session_state.get('odds_editor_version', 0) + 1
            skip_msg = f"（已保護跳過凍結場次：{', '.join(skipped_frozen)}）" if skipped_frozen else ""
            st.session_state['odds_last_status'] = f"✅ 成功更新 {updated_races_count} 場最新賠率 {skip_msg}（時間：{datetime.now(hk_timezone).strftime('%H:%M:%S')}）。"
            st.session_state['odds_server_time'] = server_updated_at
            st.session_state['odds_last_error'] = None
            if manual_refresh: st.rerun()
        else:
            st.session_state['odds_last_error'] = f"⚠️ 無法取得賠率：{error}"
            st.session_state['odds_last_status'] = None
            if manual_refresh: st.rerun()

    locked_time_str = st.session_state.get('baseline_locked_time', '未鎖定')
    form_mode_str = "🔥 真實近況與同程算分" if use_dynamic_form else "🎯 1004 爆冷模式 (固定常數算分)"
    engine_str = "雙渦輪 (Win + Place 獨立模型)" if model_place is not None else "單引擎 (僅獨贏模型)"
    st.caption(f"📌 早盤基準：`{locked_time_str}` ｜ 排序：`{strategy_mode}` ｜ 算分引擎：`{form_mode_str}` ｜ 模型：`{engine_str}`")

    if st.session_state.get('odds_last_error'):
        st.warning(st.session_state['odds_last_error'])
    elif st.session_state.get('odds_last_status'):
        st.success(st.session_state['odds_last_status'])

    st.markdown("---")
    st.subheader("🎯 各場次 AI 雙渦輪智慧投注推薦清單")

    recommendations = []
    for race_id, group in df.groupby('賽事編號'):
        r_no = extract_race_no(race_id)
        freeze_badge = f"🔒 已凍結 ({st.session_state['frozen_races'][r_no]})" if r_no in st.session_state['frozen_races'] else "🟢 即時跳動中"
        
        # 獨贏引擎篩選 與 位置引擎篩選
        sorted_win = filter_and_sort_group(group, strategy_mode, min_ev, min_odds, max_odds, bet_type="win")
        sorted_place = filter_and_sort_group(group, strategy_mode, min_ev, min_odds, max_odds, bet_type="place")

        win_pick = "⚠️ 無達標獨贏馬"
        hot_str = "-"
        if len(sorted_win) >= 1:
            top1 = sorted_win.iloc[0]
            last_r_str = f"上仗:{int(top1['real_last_rank'])}" if top1['real_last_rank'] != 6.5 else "上仗:新馬/無"
            j_hot = "🔥" if (top1['recent_jockey_win_rate'] - top1['jockey_win_rate']) >= 0.04 else ""
            t_hot = "🔥" if (top1['recent_trainer_win_rate'] - top1['trainer_win_rate']) >= 0.04 else ""
            hot_str = f"{top1['騎師']}{j_hot} / {top1['練馬師']}{t_hot}"
            win_pick = (
                f"馬號 {top1['馬號']} ({top1['馬名']}) "
                f"[AI勝率:{top1['pred_win_prob']*100:.1f}% | {last_r_str} | 賠率:{top1['基準賠率']:.1f}➔{top1['獨贏賠率']:.1f} | 獨EV:{top1['win_ev']:.2f}] "
                f"💰${top1['suggested_stake_win']:.0f} {top1['資金流向']}"
            )
            
        if len(sorted_place) >= 2:
            p1 = sorted_place.iloc[0]
            p2 = sorted_place.iloc[1]
            q_pick = f"{p1['馬號']} + {p2['馬號']} ({p1['馬名']} / {p2['馬名']}) [入位率:{p1['pred_place_prob']*100:.0f}% & {p2['pred_place_prob']*100:.0f}%]"
            if len(sorted_place) >= 3:
                p3 = sorted_place.iloc[2]
                qp_pick = f"{p1['馬號']} + {p2['馬號']} 或 {p1['馬號']} + {p3['馬號']}"
            else:
                qp_pick = f"{p1['馬號']} + {p2['馬號']} (僅一注 QP)"
        elif len(sorted_place) == 1:
            p1 = sorted_place.iloc[0]
            q_pick = f"單膽位置: {p1['馬號']} ({p1['馬名']}) [位EV:{p1['place_ev']:.2f}] 💰${p1['suggested_stake_place']:.0f}"
            qp_pick = "⚠️ 僅一匹位置達標 (不推薦 QP)"
        else:
            q_pick = "⚠️ 無達標位置馬"
            qp_pick = "⚠️ 無達標位置馬"

        if len(sorted_win) >= 1 or len(sorted_place) >= 1:
            ref_row = sorted_win.iloc[0] if len(sorted_win) >= 1 else sorted_place.iloc[0]
            if hot_str == "-":
                j_hot = "🔥" if (ref_row['recent_jockey_win_rate'] - ref_row['jockey_win_rate']) >= 0.04 else ""
                t_hot = "🔥" if (ref_row['recent_trainer_win_rate'] - ref_row['trainer_win_rate']) >= 0.04 else ""
                hot_str = f"{ref_row['騎師']}{j_hot} / {ref_row['練馬師']}{t_hot}"
                
            recommendations.append({
                '賽事編號': race_id,
                '賽道': ref_row.get('賽道', 'A'),
                '盤口狀態': freeze_badge,
                '🎯 獨贏首選 (Win Engine)': win_pick,
                '近況手風': hot_str,
                '🥉 位置/連贏推薦 (Place Engine)': q_pick,
                '🔗 位置Q推薦 (Place Engine)': qp_pick
            })

    rec_df = pd.DataFrame(recommendations)
    if rec_df.empty:
        st.warning("⚠️ 沒有符合當前篩選條件的馬匹。")
    else:
        st.dataframe(rec_df, use_container_width=True, hide_index=True)

    # 單場深度資金與實力雷達表 (同時顯示雙模型機率與雙 EV)
    status_tag = f"🔒 已凍結於 {st.session_state['frozen_races'][int(target_race)]}" if is_target_frozen else "🟢 盤口即時更新中"
    race_detail_df = df[df['賽事編號'].map(extract_race_no) == target_race].copy()
    current_pace_label = race_detail_df['race_pace'].iloc[0] if not race_detail_df.empty else "未知"
    
    st.markdown(f"#### 🔍 第 {target_race} 場 — 臨場落飛與雙模型實力深度雷達 ({status_tag} ｜ 預期步速: {current_pace_label})")
    if not race_detail_df.empty:
        if strategy_mode.startswith("🛡️"):
            sort_col_detail = 'smart_score_win'
        elif strategy_mode.startswith("🎯"):
            sort_col_detail = 'pred_win_prob'
        else:
            sort_col_detail = 'win_ev'
            
        race_detail_df = race_detail_df.sort_values(by=sort_col_detail, ascending=False).reset_index(drop=True)
        display_detail = pd.DataFrame({
            '馬號': race_detail_df['馬號'],
            '馬名': race_detail_df['馬名'],
            '騎師(近50場)': race_detail_df['騎師'] + " (" + (race_detail_df['recent_jockey_win_rate'] * 100).round(1).astype(str) + "%)",
            '練馬師': race_detail_df['練馬師'],
            '檔位': race_detail_df['排位檔位'],
            '跑法': race_detail_df['跑法'],
            '步速加成': race_detail_df['pace_adj'].map(lambda x: f"{x*100:+.0f}%" if x != 0 else "-"),
            '檔位調整': race_detail_df.apply(lambda r: "泥地無移欄" if r.get('場地') == '泥地' else (f"{r['track_bias_adj']*100:+.0f}%" if r['track_bias_adj'] != 0 else "內檔安全"), axis=1),
            '上仗名次': race_detail_df['real_last_rank'].map(lambda x: "新馬/無" if x == 6.5 else f"第 {int(x)} 名"),
            '同程勝率': (race_detail_df['real_dist_win_rate'] * 100).round(1).astype(str) + '%',
            'AI 勝率': (race_detail_df['pred_win_prob'] * 100).round(1).astype(str) + '%',
            'AI 入位率': (race_detail_df['pred_place_prob'] * 100).round(1).astype(str) + '%',
            '基準獨贏': race_detail_df['基準賠率'].round(1),
            '最新獨贏': race_detail_df['獨贏賠率'].round(1),
            '位置賠率': race_detail_df['位置賠率'].round(1),
            '資金流向': race_detail_df['資金流向'],
            '獨贏 EV': race_detail_df['win_ev'].round(2),
            '位置 EV': race_detail_df['place_ev'].round(2),
            '抗噪分': (race_detail_df['smart_score_win'] * 100).round(1),
            '獨贏注碼': race_detail_df['suggested_stake_win'].apply(lambda x: f"${x:.0f}"),
            '位置注碼': race_detail_df['suggested_stake_place'].apply(lambda x: f"${x:.0f}")
        })
        st.dataframe(display_detail, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🛒 虛擬模擬投注站 (Paper Trading)")
    st.caption("開跑前 2~3 分鐘看準上方推薦後，在此鎖定當下買入賠率（下注同時會自動🔒凍結該場盤口）。")

    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns([1, 2, 1.5, 1.5, 1.5])
    
    sim_race = col_s1.selectbox("場次", races_available if races_available else [1], key="sim_race")
    
    horses_in_race = df[df['賽事編號'].map(extract_race_no) == sim_race].copy()
    horses_in_race['display_name'] = horses_in_race['馬號'].astype(str) + " (" + horses_in_race['馬名'] + ")"
    sim_horse = col_s2.selectbox("選擇馬匹", horses_in_race['display_name'].tolist() if not horses_in_race.empty else ["無資料"], key="sim_horse")
    
    sim_type = col_s3.selectbox("玩法", ["獨贏 (WIN)", "位置 (PLACE)"], key="sim_type")
    sim_amount = col_s4.number_input("注碼", min_value=10, value=100, step=10, key="sim_amount")
    
    if col_s5.button("➕ 確定下注並凍結該場", use_container_width=True, type="primary") and not horses_in_race.empty:
        horse_no_raw = sim_horse.split(" ")[0]
        horse_name_raw = sim_horse.split("(")[1].replace(")", "")
        
        target_row = horses_in_race[horses_in_race['馬號'].astype(str) == horse_no_raw]
        if "WIN" in sim_type:
            current_odds = target_row['獨贏賠率'].values[0]
            bet_type = "WIN"
        else:
            current_odds = target_row['位置賠率'].values[0]
            bet_type = "PLA"
            
        now_str = datetime.now(hk_timezone).strftime("%H:%M:%S")
        new_bet = pd.DataFrame([{
            '下注時間': now_str,
            '賽事編號': target_row['賽事編號'].values[0],
            '場次': sim_race,
            '馬號': horse_no_raw,
            '馬名': horse_name_raw,
            '玩法': bet_type,
            '買入賠率': current_odds,
            '注碼': sim_amount
        }])
        
        st.session_state['simulated_bets'] = pd.concat([st.session_state['simulated_bets'], new_bet], ignore_index=True)
        st.session_state['frozen_races'][int(sim_race)] = now_str
        st.success(f"✅ 成功記錄注單並自動凍結第 {sim_race} 場買時盤口：{horse_no_raw} 號 ({bet_type}) @ 賠率 {current_odds}")

    if not st.session_state['simulated_bets'].empty:
        st.markdown("##### 🧾 目前模擬注單紀錄")
        st.dataframe(st.session_state['simulated_bets'], use_container_width=True, hide_index=True)
        
        dl_col, clear_col = st.columns([2, 2])
        csv_sim = st.session_state['simulated_bets'].to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        dl_col.download_button(
            label="⬇ 下載模擬注單 (CSV)",
            data=csv_sim,
            file_name=f"simulated_bets_{api_date}.csv",
            mime="text/csv",
            use_container_width=True
        )
        if clear_col.button("🗑 清除所有模擬紀錄", use_container_width=True):
            st.session_state['simulated_bets'] = pd.DataFrame(columns=[
                '下注時間', '賽事編號', '場次', '馬號', '馬名', '玩法', '買入賠率', '注碼'
            ])
            st.rerun()

# ---------------- 分頁 2: 賽後回測 ----------------
with tab2:
    st.subheader("📊 雙引擎多彩種策略回測總覽 (真實凱利注碼 vs 官方最終派彩)")
    st.caption(f"目前回測套用之排序邏輯：`{strategy_mode}` ｜ 算分引擎：`{form_mode_str}` ｜ 模型：`{engine_str}`")
    if df_backtest.empty or not backtest_ready:
        st.warning("請在左側上傳已完賽的結果 CSV。需有『賽事編號』、『馬號』及『名次』。")
    else:
        # 🤖 核心升級：AI 參數自動尋優 (Grid Search)
        st.markdown("---")
        if st.button("🪄 一鍵 AI 參數尋優 (自動找出獨贏最高 ROI 參數)", type="primary", use_container_width=True):
            with st.spinner("🤖 AI 正在背景進行網格搜索 (Grid Search)，運算上百種參數組合..."):
                best_roi = -999
                best_params = None
                
                for test_ev in np.arange(1.0, 4.1, 0.5):
                    for test_odds in np.arange(2.0, 10.1, 1.0):
                        temp_invested = 0
                        temp_return = 0
                        bets_count = 0
                        
                        for race_id, group in df_backtest.groupby('賽事編號'):
                            sg = filter_and_sort_group(group, strategy_mode, test_ev, test_odds, max_odds, bet_type="win")
                            if len(sg) > 0:
                                pick = sg.iloc[0]
                                stake = max(10, float(pick.get('suggested_stake_win', 10)) if pd.notna(pick.get('suggested_stake_win')) else 10)
                                temp_invested += stake
                                bets_count += 1
                                
                                if pick['numeric_rank'] == 1:
                                    settlement_win_odds = pick.get('回測獨贏賠率', np.nan)
                                    if pd.isna(settlement_win_odds) or settlement_win_odds <= 1:
                                        settlement_win_odds = pick['獨贏賠率']
                                    temp_return += stake * float(settlement_win_odds)
                                    
                        if temp_invested > 0 and bets_count >= 2:
                            roi = ((temp_return - temp_invested) / temp_invested) * 100
                            if roi > best_roi:
                                best_roi = roi
                                best_params = {
                                    'ev': test_ev, 'odds': test_odds, 
                                    'bets': bets_count, 'roi': roi, 
                                    'profit': temp_return - temp_invested
                                }
                                
                if best_params and best_params['roi'] > 0:
                    st.success(f"🎉 **尋優完成！今日最佳破局密碼：**\n\n建議將左側滑桿設定為 ➡️ **EV 門檻 `{best_params['ev']:.1f}`** + **最低賠率 `{best_params['odds']:.1f}` 倍**。\n在此參數下，獨贏策略可出手 {best_params['bets']} 場，預期真實 **ROI 高達 {best_params['roi']:.2f}%** (淨利 ${best_params['profit']:.0f})！")
                else:
                    st.warning("⚠️ 在目前的策略與賽果下，找不到能產生穩定正利潤的參數組合。建議切換左側的「推薦排序核心邏輯」再跑一次尋優。")
        st.markdown("---")

        sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
            "🥇 獨贏 (Win Engine)", "🥈 單匹位置 (Place Engine)", "🔗 位置Q (Place Engine)", "🥉 3匹位置包抄 (Place Engine)", "🛒 模擬結算 (Paper Trade)"
        ])

        with sub_tab1:
            st.markdown("#### 🥇 獨贏 (Win) 策略回測 — 由 `my_hkjc_model.pkl` 驅動")
            win_invested, win_return, win_bets, win_hits = 0, 0, 0, 0
            win_records = []
            for race_id, group in df_backtest.groupby('賽事編號'):
                sorted_group = filter_and_sort_group(group, strategy_mode, min_ev, min_odds, max_odds, bet_type="win")
                if len(sorted_group) > 0:
                    pick = sorted_group.iloc[0]
                    raw_stake = pick.get('suggested_stake_win', 10)
                    stake = max(10, float(raw_stake) if pd.notna(raw_stake) else 10)
                    
                    win_bets += 1
                    win_invested += stake
                    is_hit = (pick['numeric_rank'] == 1)
                    settlement_win_odds = pick.get('回測獨贏賠率', np.nan)
                    if pd.isna(settlement_win_odds) or settlement_win_odds <= 1:
                        settlement_win_odds = pick['獨贏賠率']
                        
                    payout = stake * float(settlement_win_odds) if is_hit else 0
                    if is_hit:
                        win_hits += 1
                        win_return += payout
                        result_str = "✅ 贏"
                    else:
                        result_str = "❌ 輸"
                        
                    win_records.append({
                        '賽事編號': race_id,
                        '賽道': pick.get('賽道', 'A'),
                        '投注馬號': f"{pick['馬號']} ({pick['馬名']})",
                        '上仗名次': "新馬/無" if pick['real_last_rank'] == 6.5 else f"第 {int(pick['real_last_rank'])} 名",
                        '實際名次': str(pick['名次']).replace('.0', ''),
                        '買時賠率 (選馬用)': round(pick['獨贏賠率'], 1),
                        '最終賠率 (派彩用)': round(float(settlement_win_odds), 1),
                        'AI勝率': f"{pick['pred_win_prob']*100:.1f}%",
                        '買時 獨贏EV': round(pick['win_ev'], 2),
                        '凱利注碼': f"${stake:.0f}",
                        '結果': result_str,
                        '派彩': f"${payout:.1f}",
                        '淨盈虧': f"${payout - stake:.1f}"
                    })
            if win_bets > 0:
                roi = ((win_return - win_invested) / win_invested) * 100
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("投注場數", f"{win_bets} 場")
                col2.metric("命中場數", f"{win_hits} 場", f"勝率: {win_hits/win_bets*100:.1f}%")
                col3.metric("總投入成本", f"${win_invested:,.0f}")
                col4.metric("總回收派彩", f"${win_return:,.1f}", f"真實 ROI: {roi:.2f}%")
                st.markdown("##### 📝 獨贏明細 (對照買時賠率與最終派彩)")
                st.dataframe(pd.DataFrame(win_records), use_container_width=True, hide_index=True)
            else:
                st.info("💡 目前設定下沒有符合獨贏出手的場次。")

        with sub_tab2:
            st.markdown("#### 🥈 位置 (Place) 策略回測 — 由 `my_hkjc_place_model.pkl` 驅動")
            place_invested, place_return, place_bets, place_hits = 0, 0, 0, 0
            place_records = []
            for race_id, group in df_backtest.groupby('賽事編號'):
                sorted_group = filter_and_sort_group(group, strategy_mode, min_ev, min_odds, max_odds, bet_type="place")
                if len(sorted_group) > 0:
                    pick = sorted_group.iloc[0]
                    raw_stake = pick.get('suggested_stake_place', 10)
                    stake = max(10, float(raw_stake) if pd.notna(raw_stake) else 10)
                    
                    place_bets += 1
                    place_invested += stake
                    is_hit = (pick['numeric_rank'] <= 3)
                    settlement_place_odds = pick.get('回測位置賠率', np.nan)
                    if pd.isna(settlement_place_odds) or settlement_place_odds <= 1:
                        settlement_place_odds = pick['位置賠率']
                    p_odds = float(settlement_place_odds) if pd.notna(settlement_place_odds) and float(settlement_place_odds) > 1.0 else 1.5
                    
                    payout = stake * p_odds if is_hit else 0
                    if is_hit:
                        place_hits += 1
                        place_return += payout
                        result_str = "✅ 命中位置"
                    else:
                        result_str = "❌ 未入前三"
                        
                    place_records.append({
                        '賽事編號': race_id,
                        '賽道': pick.get('賽道', 'A'),
                        '投注馬號': f"{pick['馬號']} ({pick['馬名']})",
                        '上仗名次': "新馬/無" if pick['real_last_rank'] == 6.5 else f"第 {int(pick['real_last_rank'])} 名",
                        '實際名次': str(pick['名次']).replace('.0', ''),
                        'AI入位率': f"{pick['pred_place_prob']*100:.1f}%",
                        '買時 位置EV': round(pick['place_ev'], 2),
                        '派彩賠率': round(p_odds, 2),
                        '凱利注碼': f"${stake:.0f}",
                        '結果': result_str,
                        '派彩': f"${payout:.1f}",
                        '淨盈虧': f"${payout - stake:.1f}"
                    })
            if place_bets > 0:
                roi = ((place_return - place_invested) / place_invested) * 100
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("投注場數", f"{place_bets} 場")
                col2.metric("命中位置場數", f"{place_hits} 場", f"位置勝率: {place_hits/place_bets*100:.1f}%")
                col3.metric("總投入成本", f"${place_invested:,.0f}")
                col4.metric("總回收派彩", f"${place_return:,.1f}", f"真實 ROI: {roi:.2f}%")
                st.markdown("##### 📝 位置明細")
                st.dataframe(pd.DataFrame(place_records), use_container_width=True, hide_index=True)
            else:
                st.info("💡 目前設定下沒有符合位置出手的場次。")

        with sub_tab3:
            st.markdown("#### 🔗 位置Q (QP) 策略回測 — 由 `my_hkjc_place_model.pkl` 驅動")
            qp_invested, qp_return, qp_bets, qp_hits = 0, 0, 0, 0
            qp_records = []
            for race_id, group in df_backtest.groupby('賽事編號'):
                sorted_group = filter_and_sort_group(group, strategy_mode, min_ev, min_odds, max_odds, bet_type="place")
                if len(sorted_group) >= 2:
                    top1 = sorted_group.iloc[0]
                    top2 = sorted_group.iloc[1]
                    stake = max(10, min(float(top1.get('suggested_stake_place',10) if pd.notna(top1.get('suggested_stake_place')) else 10), float(top2.get('suggested_stake_place',10) if pd.notna(top2.get('suggested_stake_place')) else 10)))
                    
                    qp_bets += 1
                    qp_invested += stake
                    is_hit = (top1['numeric_rank'] <= 3) and (top2['numeric_rank'] <= 3)
                    
                    p1_settlement = top1.get('回測位置賠率', np.nan)
                    p2_settlement = top2.get('回測位置賠率', np.nan)
                    p1_odds = float(p1_settlement if pd.notna(p1_settlement) and p1_settlement > 1.0 else top1['位置賠率'])
                    p2_odds = float(p2_settlement if pd.notna(p2_settlement) and p2_settlement > 1.0 else top2['位置賠率'])
                    estimated_qp_odds = round(p1_odds * p2_odds * 1.8, 1) 
                    
                    payout = stake * estimated_qp_odds if is_hit else 0
                    if is_hit:
                        qp_hits += 1
                        qp_return += payout
                        result_str = "✅ 命中位置Q"
                    else:
                        result_str = "❌ 落空"
                        
                    qp_records.append({
                        '賽事編號': race_id,
                        'QP 組合': f"{top1['馬號']} + {top2['馬號']} ({top1['馬名']} / {top2['馬名']})",
                        'AI入位率': f"{top1['pred_place_prob']*100:.0f}% / {top2['pred_place_prob']*100:.0f}%",
                        '實際名次': f"首選:第{str(top1['名次']).replace('.0','')}名 | 次選:第{str(top2['名次']).replace('.0','')}名",
                        '估算QP賠率': estimated_qp_odds,
                        '凱利注碼': f"${stake:.0f}",
                        '結果': result_str,
                        '派彩': f"${payout:.1f}",
                        '淨盈虧': f"${payout - stake:.1f}"
                    })
            if qp_bets > 0:
                roi = ((qp_return - qp_invested) / qp_invested) * 100
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("投注場數", f"{qp_bets} 場")
                col2.metric("命中 QP 場數", f"{qp_hits} 場", f"QP 命中率: {qp_hits/qp_bets*100:.1f}%")
                col3.metric("總投入成本", f"${qp_invested:,.0f}")
                col4.metric("總回收派彩", f"${qp_return:,.1f}", f"真實 ROI: {roi:.2f}%")
                st.markdown("##### 📝 位置Q (QP) 明細")
                st.dataframe(pd.DataFrame(qp_records), use_container_width=True, hide_index=True)
            else:
                st.info("💡 目前設定下沒有符合位置 Q 出手的場次。")

        with sub_tab4:
            st.markdown("#### 🥉 3匹位置包抄 (3x Place) 策略回測 — 由 `my_hkjc_place_model.pkl` 驅動")
            st.caption("策略邏輯：每場買入位置模型推薦最高分的 3 匹馬位置。")
            p3_invested, p3_return, p3_bets, p3_hits = 0, 0, 0, 0
            p3_records = []
            
            for race_id, group in df_backtest.groupby('賽事編號'):
                sorted_group = filter_and_sort_group(group, strategy_mode, min_ev, min_odds, max_odds, bet_type="place")
                picks = sorted_group.head(3)
                
                for _, pick in picks.iterrows():
                    raw_stake = pick.get('suggested_stake_place', 10)
                    stake = max(10, float(raw_stake) if pd.notna(raw_stake) else 10)
                    
                    p3_bets += 1
                    p3_invested += stake
                    is_hit = (pick['numeric_rank'] <= 3)
                    
                    settlement_place_odds = pick.get('回測位置賠率', np.nan)
                    if pd.isna(settlement_place_odds) or settlement_place_odds <= 1:
                        settlement_place_odds = pick['位置賠率']
                    p_odds = float(settlement_place_odds) if pd.notna(settlement_place_odds) and float(settlement_place_odds) > 1.0 else 1.5
                    
                    payout = stake * p_odds if is_hit else 0
                    if is_hit:
                        p3_hits += 1
                        p3_return += payout
                        result_str = "✅ 命中位置"
                    else:
                        result_str = "❌ 未入前三"
                        
                    p3_records.append({
                        '賽事編號': race_id,
                        '投注馬號': f"{pick['馬號']} ({pick['馬名']})",
                        '上仗名次': "新馬/無" if pick['real_last_rank'] == 6.5 else f"第 {int(pick['real_last_rank'])} 名",
                        '實際名次': str(pick['名次']).replace('.0', ''),
                        'AI入位率': f"{pick['pred_place_prob']*100:.1f}%",
                        '派彩賠率': round(p_odds, 2),
                        '買時 位置EV': round(pick['place_ev'], 2),
                        '凱利注碼': f"${stake:.0f}",
                        '結果': result_str,
                        '派彩': f"${payout:.1f}",
                        '淨盈虧': f"${payout - stake:.1f}"
                    })
                    
            if p3_bets > 0:
                roi = ((p3_return - p3_invested) / p3_invested) * 100
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("總投注注數", f"{p3_bets} 注")
                col2.metric("命中注數", f"{p3_hits} 注", f"命中率: {p3_hits/p3_bets*100:.1f}%")
                col3.metric("總投入成本", f"${p3_invested:,.0f}")
                col4.metric("總回收派彩", f"${p3_return:,.1f}", f"真實 ROI: {roi:.2f}%")
                st.markdown("##### 📝 3匹位置包抄明細")
                st.dataframe(pd.DataFrame(p3_records), use_container_width=True, hide_index=True)
            else:
                st.info("💡 目前設定下沒有符合出手的場次。")

        with sub_tab5:
            st.markdown("#### 🛒 模擬投注真實結算 (Paper Trading PnL)")
            st.caption("系統會將你的虛擬注單與官方賽果比對，並優先使用官方最終派彩賠率（若無則用買入賠率）計算真實盈虧。")
            
            if 'simulated_bets' not in st.session_state or st.session_state['simulated_bets'].empty:
                st.info("💡 目前沒有任何模擬投注紀錄。請先在「賽前預測」分頁進行模擬下注。")
            else:
                sim_df = st.session_state['simulated_bets'].copy()
                sim_results = []
                sim_invested, sim_return, sim_hits = 0, 0, 0
                sim_bets = len(sim_df)

                for idx, bet in sim_df.iterrows():
                    race_id = bet['賽事編號']
                    horse_no = str(bet['馬號'])
                    bet_type = bet['玩法']
                    locked_odds = float(bet['買入賠率'])
                    stake = float(bet['注碼'])

                    sim_invested += stake

                    match = df_backtest[(df_backtest['賽事編號'] == race_id) & (df_backtest['馬號'].astype(str) == horse_no)]

                    if match.empty or pd.isna(match['numeric_rank'].values[0]) or match['numeric_rank'].values[0] == 99:
                        sim_results.append({
                            '下注時間': bet['下注時間'],
                            '場次': bet['場次'],
                            '馬匹': f"{horse_no} ({bet['馬名']})",
                            '玩法': bet_type,
                            '注碼': f"${stake:.0f}",
                            '買入時賠率': locked_odds,
                            '實際名次': '-',
                            '結果': '⏳ 待開彩',
                            '派彩': '$0.0',
                            '淨盈虧': '$0.0'
                        })
                    else:
                        actual_rank = match['numeric_rank'].values[0]
                        rank_str = str(match['名次'].values[0]).replace('.0', '')

                        is_hit = False
                        if bet_type == "WIN" and actual_rank == 1:
                            is_hit = True
                            sp_odds = match.get('回測獨贏賠率', pd.Series([np.nan])).values[0]
                            final_odds = float(sp_odds) if pd.notna(sp_odds) and float(sp_odds) > 1.0 else locked_odds
                        elif bet_type == "PLA" and actual_rank <= 3:
                            is_hit = True
                            sp_odds = match.get('回測位置賠率', pd.Series([np.nan])).values[0]
                            final_odds = float(sp_odds) if pd.notna(sp_odds) and float(sp_odds) > 1.0 else locked_odds
                        else:
                            final_odds = locked_odds

                        if is_hit:
                            sim_hits += 1
                            payout = stake * final_odds
                            sim_return += payout
                            result_text = "✅ 贏"
                        else:
                            payout = 0
                            result_text = "❌ 輸"

                        sim_results.append({
                            '下注時間': bet['下注時間'],
                            '場次': bet['場次'],
                            '馬匹': f"{horse_no} ({bet['馬名']})",
                            '玩法': bet_type,
                            '注碼': f"${stake:.0f}",
                            '買入時賠率': locked_odds,
                            '最終派彩賠率': round(final_odds, 1),
                            '實際名次': f"第 {rank_str} 名",
                            '結果': result_text,
                            '派彩': f"${payout:.1f}",
                            '淨盈虧': f"${payout - stake:.1f}"
                        })

                if sim_invested > 0:
                    roi = ((sim_return - sim_invested) / sim_invested) * 100
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("模擬總注數", f"{sim_bets} 注")
                    col2.metric("命中注數", f"{sim_hits} 注", f"命中率: {sim_hits/sim_bets*100:.1f}%")
                    col3.metric("總投入資金", f"${sim_invested:,.1f}")
                    col4.metric("總回收派彩", f"${sim_return:,.1f}", f"真實 ROI: {roi:.2f}%")

                    st.markdown("##### 🧾 模擬投注詳細結算單")
                    st.dataframe(pd.DataFrame(sim_results), use_container_width=True, hide_index=True)
