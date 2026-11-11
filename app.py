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
import json
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

st.title("🐎 HKJC 15大特徵 AI 賽馬預測系統（WIN／PLACE／QP排序引擎）")
st.markdown("---")

# 初始化單場凍結名單
st.session_state.pop('simulated_bets', None)
if 'frozen_races' not in st.session_state:
    st.session_state['frozen_races'] = {}  # 格式: {race_no: "HH:MM:SS"}

ENGINE_VERSION = 'asof-ranker-v2-20261111-backtest'
if st.session_state.get('prediction_engine_version') != ENGINE_VERSION:
    st.session_state.pop('df_data', None)
    st.session_state.pop('prediction_signature', None)
    st.session_state.pop('baseline_odds_map', None)
    st.session_state.pop('baseline_locked_time', None)
    st.session_state['frozen_races'] = {}
    saved_card = st.session_state.get('fetched_prediction_df')
    if isinstance(saved_card, pd.DataFrame):
        for odds_col in ('獨贏賠率', '位置賠率'):
            saved_card[odds_col] = np.nan
        st.session_state['fetched_prediction_df'] = saved_card
        st.session_state['pending_auto_odds_fetch'] = True
    st.session_state['prediction_engine_version'] = ENGINE_VERSION

# 設定 repo 內的模型及預測 CSV 路徑
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / 'my_hkjc_model.pkl'  # 現有WIN分類器，保留作備援
PLACE_MODEL_PATH = APP_DIR / 'my_hkjc_place_model.pkl'  # 現有Dynamic PLACE模型
RANKER_ROOT = APP_DIR / 'win_ranker_redesign_20261110'
WIN_RANKER_FIXED_PATH = RANKER_ROOT / 'fixed_form' / 'my_hkjc_win_ranker_candidate.pkl'
WIN_RANKER_DYNAMIC_PATH = RANKER_ROOT / 'dynamic_form' / 'my_hkjc_win_ranker_candidate.pkl'
WIN_META_FIXED_PATH = RANKER_ROOT / 'fixed_form' / 'candidate_metadata.json'
WIN_META_DYNAMIC_PATH = RANKER_ROOT / 'dynamic_form' / 'candidate_metadata.json'
PLACE_FIXED_PATH = RANKER_ROOT / 'fixed_form' / 'my_hkjc_place_pair_candidate.pkl'
PLACE_DYNAMIC_PATH = RANKER_ROOT / 'dynamic_form' / 'my_hkjc_place_pair_candidate.pkl'
QP_PAIR_FIXED_PATH = RANKER_ROOT / 'fixed_form' / 'my_hkjc_qp_pair_ranker_candidate.pkl'
QP_PAIR_DYNAMIC_PATH = RANKER_ROOT / 'dynamic_form' / 'my_hkjc_qp_pair_ranker_candidate.pkl'
PREDICTION_CSV_PATH = APP_DIR / 'prediction.csv'
FEATURE_COLS_15 = [
    'market_implied_prob', '獨贏賠率', 'odds_rank', 'is_favorite',
    '排位檔位', 'weight_diff', 'weight_rank',
    'jockey_win_rate', 'trainer_win_rate', 'combo_win_rate',
    'horse_win_rate', 'horse_last_rank',
    '距離', 'horse_surface_win_rate', 'horse_dist_win_rate',
]
PAIR_STATS = ('sum', 'absdiff', 'min', 'max')
PAIR_FEATURES = [f'pair_{stat}_{col}' for col in FEATURE_COLS_15 for stat in PAIR_STATS]

def _model_file_signature(model_path):
    """Return a cache key that changes when a model is added or replaced."""
    try:
        stat = Path(model_path).stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        return "missing"


@st.cache_resource
def load_models(win_model_path, place_model_path, win_signature, place_signature):
    # Signatures are intentional cache-key inputs: a model uploaded after the
    # first Streamlit run must not remain stuck as the previously cached None.
    win_m = joblib.load(win_model_path) if win_signature != "missing" else None
    place_m = joblib.load(place_model_path) if place_signature != "missing" else None
    return win_m, place_m

model_win, model_place = load_models(
    str(MODEL_PATH),
    str(PLACE_MODEL_PATH),
    _model_file_signature(MODEL_PATH),
    _model_file_signature(PLACE_MODEL_PATH),
)

@st.cache_resource
def load_optional_model(model_path, signature):
    return joblib.load(model_path) if signature != 'missing' else None

@st.cache_data
def load_model_metadata(metadata_path, signature):
    if signature == 'missing':
        return {}
    try:
        return json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    except Exception:
        return {}

model_win_ranker_fixed = load_optional_model(str(WIN_RANKER_FIXED_PATH), _model_file_signature(WIN_RANKER_FIXED_PATH))
model_win_ranker_dynamic = load_optional_model(str(WIN_RANKER_DYNAMIC_PATH), _model_file_signature(WIN_RANKER_DYNAMIC_PATH))
model_place_fixed = load_optional_model(str(PLACE_FIXED_PATH), _model_file_signature(PLACE_FIXED_PATH))
model_place_dynamic = load_optional_model(str(PLACE_DYNAMIC_PATH), _model_file_signature(PLACE_DYNAMIC_PATH))
model_qp_fixed = load_optional_model(str(QP_PAIR_FIXED_PATH), _model_file_signature(QP_PAIR_FIXED_PATH))
model_qp_dynamic = load_optional_model(str(QP_PAIR_DYNAMIC_PATH), _model_file_signature(QP_PAIR_DYNAMIC_PATH))
meta_win_fixed = load_model_metadata(str(WIN_META_FIXED_PATH), _model_file_signature(WIN_META_FIXED_PATH))
meta_win_dynamic = load_model_metadata(str(WIN_META_DYNAMIC_PATH), _model_file_signature(WIN_META_DYNAMIC_PATH))

if model_win is None:
    st.error(f"⚠️ 找不到獨贏 AI 模型檔 `{MODEL_PATH}`！請確認模型是否已上傳至正確目錄。")
    st.stop()

if model_place is not None:
    st.sidebar.success("🧠 雙渦輪引擎已啟動：獨贏模型 + 位置模型")
else:
    st.sidebar.warning("⚠️ 找不到 PLACE 模型，位置彩池將暫用獨贏模型代算。")
    st.sidebar.caption(f"預期 PLACE 模型路徑：`{PLACE_MODEL_PATH}`")

@st.cache_data
def load_historical_stats(history_csv_path='hkjc_all_seasons_features.csv', as_of_date=None, file_signature=''):
    """只用目標賽日前已完成的正式出賽，計算15-feature歷史統計。"""
    csv_path = Path(__file__).resolve().parent / history_csv_path
    if not csv_path.exists():
        return None
    try:
        try:
            hist_df = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)
        except UnicodeDecodeError:
            hist_df = pd.read_csv(csv_path, encoding='cp950', low_memory=False)
        if not {'賽事編號','名次','馬名','騎師','練馬師'}.issubset(hist_df.columns):
            raise ValueError('History缺少賽事編號、名次、馬名、騎師或練馬師欄位')
        hist_df['_race_date'] = pd.to_datetime(
            hist_df['賽事編號'].astype(str).str.extract(r'(20\d{6})', expand=False),
            format='%Y%m%d', errors='coerce')
        if as_of_date:
            cutoff = pd.Timestamp(as_of_date).normalize()
            hist_df = hist_df.loc[hist_df['_race_date'].notna() & (hist_df['_race_date'] < cutoff)].copy()
        else:
            raise ValueError('無法從預測賽事編號解析賽日；為避免未來資料洩漏，停止載入History特徵')
        rank_text = hist_df['名次'].astype('string').fillna('').str.strip()
        raw_win_odds = pd.to_numeric(hist_df.get('獨贏賠率', pd.Series(np.nan,index=hist_df.index)), errors='coerce')
        ambiguous_blank = rank_text.eq('') & raw_win_odds.gt(1.0)
        if ambiguous_blank.any():
            raise ValueError(f'有 {int(ambiguous_blank.sum())} 列名次空白但WIN賠率有效；請先核實是否出賽，避免污染歷史勝率')
        nonstarter = rank_text.str.contains(r'^(?:WR|WV|WX|TNP|WD|SCR|SCRATCHED|WITHDRAWN)|退出|退賽|未有出賽', case=False, regex=True, na=False)
        nonstarter = nonstarter | (rank_text.eq('') & ~raw_win_odds.gt(1.0))
        void = rank_text.str.contains(r'VOID|ABANDONED|取消賽事', case=False, regex=True, na=False)
        hist_df['numeric_rank'] = pd.to_numeric(rank_text.str.extract(r'^\s*(\d+)', expand=False), errors='coerce')
        hist_df = hist_df.loc[~nonstarter & ~void].copy()
        hist_df = hist_df.drop_duplicates(['賽事編號','馬號'],keep='last').copy()
        if hist_df.empty:
            raise ValueError(f'History在{pd.Timestamp(as_of_date).date()}前沒有可用完賽資料')
        hist_df['is_win'] = hist_df['numeric_rank'].eq(1).fillna(False).astype('int8')
        hist_df['_clean_jockey'] = hist_df['騎師'].astype(str).str.replace(r'\(.*?\)', '', regex=True).str.replace(' ', '')
        hist_df['_clean_trainer'] = hist_df['練馬師'].astype(str).str.replace(' ', '')
        hist_df['_clean_horse'] = hist_df['馬名'].astype(str).str.replace(r'\(.*?\)', '', regex=True).str.replace(' ', '')
        jockey_stats = hist_df.groupby('_clean_jockey', as_index=False)['is_win'].mean().rename(columns={'is_win':'hist_jockey_win_rate'})
        trainer_stats = hist_df.groupby('_clean_trainer', as_index=False)['is_win'].mean().rename(columns={'is_win':'hist_trainer_win_rate'})
        horse_stats = hist_df.groupby('_clean_horse', as_index=False)['is_win'].mean().rename(columns={'is_win':'hist_horse_win_rate'})
        combo_stats = hist_df.groupby(['_clean_jockey','_clean_trainer'], as_index=False)['is_win'].mean().rename(columns={'is_win':'hist_combo_win_rate'})
        ordered = hist_df.sort_values(['_race_date','賽事編號'],kind='mergesort')
        recent_j = ordered.groupby('_clean_jockey',sort=False).tail(50).groupby('_clean_jockey',as_index=False)['is_win'].mean().rename(columns={'is_win':'recent_jockey_win_rate'})
        recent_t = ordered.groupby('_clean_trainer',sort=False).tail(50).groupby('_clean_trainer',as_index=False)['is_win'].mean().rename(columns={'is_win':'recent_trainer_win_rate'})
        jockey_stats = jockey_stats.merge(recent_j,on='_clean_jockey',how='left')
        trainer_stats = trainer_stats.merge(recent_t,on='_clean_trainer',how='left')
        numeric_ordered = ordered.loc[ordered['numeric_rank'].notna()].copy()
        horse_last_rank_stats = numeric_ordered.groupby('_clean_horse',as_index=False).tail(1)[['_clean_horse','numeric_rank']].rename(columns={'numeric_rank':'hist_horse_last_rank'})
        if '場地' in hist_df.columns:
            hist_df['_clean_surface'] = hist_df['場地'].astype(str).str.strip()
            horse_surface_stats = hist_df.groupby(['_clean_horse','_clean_surface'],as_index=False)['is_win'].mean().rename(columns={'is_win':'hist_horse_surface_win_rate'})
        else:
            horse_surface_stats = pd.DataFrame(columns=['_clean_horse','_clean_surface','hist_horse_surface_win_rate'])
        if '距離' in hist_df.columns:
            hist_df['_clean_dist'] = pd.to_numeric(hist_df['距離'],errors='coerce').fillna(1200).astype(int)
            horse_dist_stats = hist_df.groupby(['_clean_horse','_clean_dist'],as_index=False)['is_win'].mean().rename(columns={'is_win':'hist_horse_dist_win_rate'})
        else:
            horse_dist_stats = pd.DataFrame(columns=['_clean_horse','_clean_dist','hist_horse_dist_win_rate'])
        return jockey_stats,trainer_stats,horse_stats,combo_stats,horse_last_rank_stats,horse_surface_stats,horse_dist_stats
    except Exception as e:
        st.sidebar.error(f'載入賽日前History失敗：{e}')
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
                '獨贏賠率': np.nan,
                '位置賠率': np.nan,
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
    help='WIN只按CSV中官方最終獨贏賠率作回收；PLACE需提供每$10官方位置派彩，QP需提供逐組官方派彩，缺項時只報命中率。',
)

# ==========================================
# ⚙️ 策略、資金與算分引擎設定
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("⚙ 投注策略與算分引擎")
strategy_mode = st.sidebar.radio(
    "1️⃣ 推薦排序核心邏輯",
    ["⚡ 傳統純 EV 排序 (高賠率優先)", "🛡️ 抗噪穩定 (含步速+手風+賽道+排除回飛)", "🎯 純 AI 勝率/入位率最高 (命中率優先)"],
    index=2,
    help="預設【純AI機率】用於你指定的每場WIN最高分馬命中與PLACE/QP組合命中目標；不代表ROI較高。可切換【傳統純EV】或【抗噪穩定】作並行比較。"
)
win_labels = {}
if model_win_ranker_fixed is not None: win_labels['新WIN排序器（Fixed）'] = 'ranker_fixed'
if model_win_ranker_dynamic is not None: win_labels['新WIN排序器（Dynamic）'] = 'ranker_dynamic'
if model_win is not None: win_labels['現行WIN分類器（Fixed）'] = 'legacy_fixed'
win_default = next((i for i,k in enumerate(win_labels) if k == 'ranker_fixed'), 0)
win_model_choice = st.sidebar.selectbox('2️⃣ WIN引擎（每場只選一匹）', list(win_labels) or ['無可用WIN模型'], index=win_default,
    help='新WIN排序器直接學每場頭馬排序，機率以驗證期溫度校準後按場合計100%；現行分類器保留作比較。')
win_mode = win_labels.get(win_model_choice, 'legacy_fixed')
win_dynamic_form = win_mode == 'ranker_dynamic'

place_labels = {}
if model_place_dynamic is not None: place_labels['現有PLACE模型（Dynamic）'] = 'legacy_place_dynamic'
if model_place_fixed is not None: place_labels['新PLACE模型（Fixed）'] = 'place_fixed'
if model_place_dynamic is not None: place_labels['新PLACE模型（Dynamic）'] = 'place_dynamic'
if model_place is not None: place_labels['GitHub PLACE模型（Dynamic）'] = 'github_place_dynamic'
place_default = next((i for i,k in enumerate(place_labels) if k == 'place_fixed'), 0)
place_model_choice = st.sidebar.selectbox('3️⃣ PLACE引擎（入位排序）', list(place_labels) or ['無可用PLACE模型'], index=place_default,
    help='PLACE模型獨立選Form；此選擇會影響位置候選及以PLACE馬匹組合構成的QP。')
place_mode = place_labels.get(place_model_choice, 'github_place_dynamic')
place_dynamic_form = place_mode.endswith('dynamic')

qp_labels = {'以單馬PLACE排序組成QP': 'place_top2'}
if model_qp_fixed is not None: qp_labels['直接QP雙馬排序器（Fixed）'] = 'qp_fixed'
if model_qp_dynamic is not None: qp_labels['直接QP雙馬排序器（Dynamic）'] = 'qp_dynamic'
qp_choice = st.sidebar.selectbox('4️⃣ 位置Q/QP組合排序', list(qp_labels), index=0,
    help='直接QP模型以歷史同場馬匹配對是否同時入位為目標；若模型不可用，保留PLACE前兩名組合。')
qp_mode = qp_labels[qp_choice]
qp_dynamic_form = qp_mode == 'qp_dynamic'
min_win_ev = st.sidebar.slider("🥇 獨贏最小期望值 (Win EV)", 0.0, 6.0, 0.0, 0.1)
min_place_ev = st.sidebar.slider("🥉 位置最小期望值 (Place EV)", 0.0, 6.0, 0.0, 0.1)
min_odds = st.sidebar.number_input("最低獨贏賠率", min_value=1.0, max_value=50.0, value=1.0, help="預設1.0代表純AI命中率評估不以最低賠率篩馬；提高門檻會改變候選母體。位置彩池會自動將此門檻除以3。")
max_odds = st.sidebar.number_input("最高獨贏賠率", min_value=1.0, max_value=100.0, value=100.0)

st.sidebar.markdown("---")
st.sidebar.header("💰 資金管理 (凱利注碼精算)")
total_bankroll = st.sidebar.number_input("💵 單日總本金預算 (HKD)", min_value=100, max_value=1000000, value=500, step=500)
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

def is_scratched(value):
    text = str(value).strip().upper()
    return text in {"SCR", "SCRATCH", "SCRATCHED", "退出", "退賽", "已退出", "取消"} or "退出" in text or "SCRATCH" in text

def extract_race_no(value):
    match = re.search(r"-(\d+)\s*$", str(value).strip())
    return int(match.group(1)) if match else None

def normalize_race_id(value):
    text = str(value).strip()
    match = re.search(r"(20\d{6}).*?(\d+)\s*$", text)
    if match:
        return f"{match.group(1)}-{int(match.group(2)):02d}"
    return text

def fetch_live_odds(date_str, venue, race_no=None):
    body = {
        "operationName": "racing",
        "variables": {
            "date": date_str,
            "venueCode": venue,
            "raceNo": int(race_no) if race_no is not None else None,
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
                    race_number, {"WIN": {}, "PLA": {}, "SCRATCHED": set()}
                )
                for node in pool.get("oddsNodes") or []:
                    horse_no = normalize_horse_no(node.get("combString"))
                    odds_value = str(node.get("oddsValue")).strip()
                    if horse_no and odds_value.upper() == "SCR":
                        market["SCRATCHED"].add(horse_no)
                    elif horse_no and odds_value not in ("None", ""):
                        try:
                            market[pool_type][horse_no] = float(odds_value)
                        except ValueError:
                            pass

        if not odds_by_race:
            return None, None, "賽事有回應，但目前尚無可用 WIN／PLACE 賠率。"
        if race_no is not None:
            requested_race = int(race_no)
            odds_by_race = {n: market for n, market in odds_by_race.items() if int(n) == requested_race}
            if not odds_by_race:
                return None, max(timestamps) if timestamps else None, f"HKJC回應內沒有第 {requested_race} 場報價。"
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
    st.sidebar.info('Repo尚無預測卡；可直接使用上方「抓取排位並載入預測」取得新賽日官方排位。')
    empty_card_columns = [
        '賽事編號','名次','馬號','馬名','騎師','練馬師','實際負磅',
        '排位檔位','獨贏賠率','位置賠率','距離','場地','賽道'
    ]
    df_raw = pd.DataFrame({column: pd.Series(dtype=object) for column in empty_card_columns})
    source_label = '等待抓取HKJC官方排位'
    prediction_signature = 'awaiting-official-racecard'

# 缺少狀態欄時預設仍在陣；只有明確官方SCR或檔案明確標記才視為退出。
if '參賽狀態' not in df_raw.columns:
    df_raw['參賽狀態'] = '參賽'

# Historical prediction CSVs are archived model inputs for backtesting. Keep
# their saved prices; for today's/future repo cards require a fresh HKJC quote.
race_date_values = pd.Series(dtype='datetime64[ns]')
if len(df_raw) and '賽事編號' in df_raw.columns:
    race_date_values = pd.to_datetime(
        df_raw['賽事編號'].astype(str).str.extract(r'(20\d{6})', expand=False),
        format='%Y%m%d', errors='coerce'
    )
archived_prediction_card = bool(
    len(df_raw) > 0 and race_date_values.notna().all()
    and race_date_values.nunique() == 1
    and race_date_values.iloc[0].date() < datetime.now(hk_timezone).date()
)
saved_odds_snapshot = bool(
    fetched_prediction_df is None
    and SELECTED_CSV_PATH.is_file()
    and 'bet_odds_updated' in SELECTED_CSV_PATH.stem.lower()
)
archived_prediction_card = archived_prediction_card or saved_odds_snapshot
if fetched_prediction_df is None and not archived_prediction_card:
    df_raw['獨贏賠率'] = np.nan
    df_raw['位置賠率'] = np.nan
elif saved_odds_snapshot:
    st.sidebar.warning('已載入保存的落註賠率回測快照：保留檔案內WIN／PLACE價格，不會清空或當作即時更新報價。')
elif archived_prediction_card:
    st.sidebar.warning('已載入過去賽日預測卡：保留檔案內賠率供歷史命中率回測；不視為目前即時官方報價。')

auto_odds_status = None
if fetched_prediction_df is not None and not archived_prediction_card and st.session_state.pop('pending_auto_odds_fetch', False):
    odds_date = st.session_state.get('fetched_prediction_date', race_date_choice).strftime('%Y-%m-%d')
    odds_venue = str(df_raw['racecourse_code'].iloc[0])
    initial_race = min(df_raw['賽事編號'].map(extract_race_no).dropna().astype(int)) if df_raw['賽事編號'].map(extract_race_no).notna().any() else 1
    with st.spinner(f'排位已載入，正在嘗試取得第 {initial_race} 場 WIN／PLACE 賠率…'):
        odds_by_race, odds_updated_at, odds_error = fetch_live_odds(odds_date, odds_venue, initial_race)
    if odds_by_race:
        if '參賽狀態' not in df_raw.columns:
            df_raw['參賽狀態'] = '參賽'
        win_count = place_count = 0
        for idx, runner in df_raw.iterrows():
            race_no = extract_race_no(runner.get('賽事編號', ''))
            horse_no = normalize_horse_no(runner.get('馬號', ''))
            race_market = odds_by_race.get(race_no, {})
            if horse_no in race_market.get('SCRATCHED', set()):
                df_raw.at[idx, '參賽狀態'] = '退出'
                df_raw.at[idx, '獨贏賠率'] = np.nan
                df_raw.at[idx, '位置賠率'] = np.nan
                continue
            win_price = race_market.get('WIN', {}).get(horse_no)
            place_price = race_market.get('PLA', {}).get(horse_no)
            if win_price is not None and win_price > 1.0:
                df_raw.at[idx, '獨贏賠率'] = win_price
                win_count += 1
            if place_price is not None and place_price > 1.0:
                df_raw.at[idx, '位置賠率'] = place_price
                place_count += 1
        st.session_state['df_data'] = df_raw.copy()
        st.session_state['fetched_prediction_df'] = df_raw.copy()
        
        base_map = {}
        for _, r in df_raw.iterrows():
            k = f"{normalize_race_id(r['賽事編號'])}_{normalize_horse_no(r['馬號'])}"
            base_value = pd.to_numeric(r['獨贏賠率'], errors='coerce')
            if pd.notna(base_value) and base_value > 1.0:
                base_map[k] = float(base_value)
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
    df_raw['獨贏賠率'] = np.nan
else:
    df_raw['獨贏賠率'] = pd.to_numeric(df_raw['獨贏賠率'], errors='coerce')
if '位置賠率' not in df_raw.columns:
    df_raw['位置賠率'] = np.nan
else:
    df_raw['位置賠率'] = pd.to_numeric(df_raw['位置賠率'], errors='coerce')
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
        if pd.notna(val) and val > 1.0:
            base_map[k] = float(val)
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
st.session_state['df_data']['獨贏賠率'] = pd.to_numeric(edited_df['獨贏賠率'], errors='coerce')
st.session_state['df_data']['位置賠率'] = pd.to_numeric(edited_df['位置賠率'], errors='coerce')
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
race_date_text = df['賽事編號'].astype(str).str.extract(r'(20\d{6})', expand=False)
parsed_race_dates = pd.to_datetime(race_date_text, format='%Y%m%d', errors='coerce')
as_of_date = parsed_race_dates.min()
history_path = APP_DIR / 'hkjc_all_seasons_features.csv'
history_signature = _model_file_signature(history_path)
hist_stats = load_historical_stats('hkjc_all_seasons_features.csv', as_of_date.strftime('%Y-%m-%d') if pd.notna(as_of_date) else None, history_signature)
history_features_ok = hist_stats is not None
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
    df['real_last_rank'] = pd.to_numeric(df['hist_horse_last_rank'], errors='coerce').astype(float).fillna(6.5)
    df['real_surface_win_rate'] = df['hist_horse_surface_win_rate'].fillna(df['horse_win_rate']).fillna(0.06)
    df['real_dist_win_rate'] = df['hist_horse_dist_win_rate'].fillna(df['horse_win_rate']).fillna(0.06)
    
    # 🚀 核心切換：決定餵給 AI 模型算分的特徵值
    if win_dynamic_form:
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
    st.sidebar.error("❌ 未能讀取History；模型分數已停止，並未以預設勝率代替。")
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

# 🚀 WIN/PLACE分別依模型訓練Form計分；缺少任何馬匹官方WIN報價的賽事不計模型分數
feature_cols = FEATURE_COLS_15
for col in feature_cols:
    if col not in df.columns:
        df[col] = np.nan

def make_form_matrix(frame, dynamic):
    x = frame[FEATURE_COLS_15].copy()
    if dynamic:
        x['horse_last_rank'] = pd.to_numeric(frame['real_last_rank'], errors='coerce').fillna(6.5)
        x['horse_surface_win_rate'] = pd.to_numeric(frame['real_surface_win_rate'], errors='coerce').fillna(0.06)
        x['horse_dist_win_rate'] = pd.to_numeric(frame['real_dist_win_rate'], errors='coerce').fillna(0.06)
    else:
        x['horse_last_rank'] = 6.0
        x['horse_surface_win_rate'] = 0.08
        x['horse_dist_win_rate'] = 0.08
    return x.apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0.0)

X_win = make_form_matrix(df, win_dynamic_form)
X_place = make_form_matrix(df, place_dynamic_form)
X_qp = make_form_matrix(df, qp_dynamic_form)
price_numeric = pd.to_numeric(df['獨贏賠率'], errors='coerce')
if archived_prediction_card:
    # In archived cards, zero/missing prices commonly belong to non-starters.
    # Score only quoted runners; the result file later identifies scratches.
    valid_idx = df.index[price_numeric.gt(1.0)]
else:
    scratched_mask = df['參賽狀態'].map(is_scratched)
    complete_races = df.assign(_eligible_runner_price=price_numeric.gt(1.0) | scratched_mask).groupby('賽事編號')['_eligible_runner_price'].all()
    valid_race_ids = set(complete_races[complete_races].index)
    valid_idx = df.index[df['賽事編號'].isin(valid_race_ids) & ~scratched_mask & price_numeric.gt(1.0)]
if not history_features_ok:
    valid_idx = df.index[:0]
    st.sidebar.error('已停止預測：History未能讀取或計算，不能用預設勝率代替訓練特徵。')
df['pred_win_prob'] = np.nan
df['pred_win_score'] = np.nan
df['pred_place_prob'] = np.nan
df['qp_pair_score'] = np.nan
qp_pair_recommendations = {}

if len(valid_idx):
    if win_mode.startswith('ranker_'):
        chosen_ranker = model_win_ranker_dynamic if win_dynamic_form else model_win_ranker_fixed
        chosen_meta = meta_win_dynamic if win_dynamic_form else meta_win_fixed
        if chosen_ranker is not None:
            scores = chosen_ranker.predict(X_win.loc[valid_idx])
            df.loc[valid_idx, 'pred_win_score'] = scores
            temperature = max(float(chosen_meta.get('ranker_softmax_temperature',1.0)),1e-6)
            for race_id, inds in df.loc[valid_idx].groupby('賽事編號',sort=False).groups.items():
                vals = df.loc[inds,'pred_win_score'].to_numpy(dtype=float)
                logits = (vals - np.max(vals)) / temperature
                probs = np.exp(np.clip(logits,-700,0)); probs = probs / probs.sum()
                df.loc[inds,'pred_win_prob'] = probs
        elif model_win is not None:
            st.sidebar.warning('選定WIN排序器檔案未載入，暫用現行WIN分類器；請檢查模型包。')
            df.loc[valid_idx,'pred_win_prob'] = model_win.predict_proba(X_win.loc[valid_idx])[:,1]
    elif model_win is not None:
        df.loc[valid_idx,'pred_win_prob'] = model_win.predict_proba(X_win.loc[valid_idx])[:,1]

    place_model_choice_obj = {
        'legacy_place_dynamic': model_place,
        'place_fixed': model_place_fixed,
        'place_dynamic': model_place_dynamic,
        'github_place_dynamic': model_place,
    }.get(place_mode)
    if place_model_choice_obj is None:
        place_model_choice_obj = model_win
        st.sidebar.warning('所選PLACE模型未載入；位置概率將暫用WIN模型，請勿當作正式PLACE估值。')
    if place_model_choice_obj is not None:
        Xp = X_place if place_dynamic_form else X_place
        df.loc[valid_idx,'pred_place_prob'] = place_model_choice_obj.predict_proba(Xp.loc[valid_idx])[:,1]

    qp_ranker_obj = model_qp_dynamic if qp_dynamic_form else model_qp_fixed
    if qp_mode in ('qp_fixed','qp_dynamic') and qp_ranker_obj is not None:
        from itertools import combinations
        for race_id, inds in df.loc[valid_idx].groupby('賽事編號',sort=False).groups.items():
            inds=list(inds); vals=X_qp.loc[inds,FEATURE_COLS_15].to_numpy(dtype=float)
            pairs=[]; pair_rows=[]
            for i,j in combinations(range(len(inds)),2):
                row={}
                for k,col in enumerate(FEATURE_COLS_15):
                    a,b=float(vals[i,k]),float(vals[j,k])
                    row[f'pair_sum_{col}']=a+b; row[f'pair_absdiff_{col}']=abs(a-b)
                    row[f'pair_min_{col}']=min(a,b); row[f'pair_max_{col}']=max(a,b)
                pairs.append(row); pair_rows.append((inds[i],inds[j]))
            if pairs:
                pair_scores=qp_ranker_obj.predict(pd.DataFrame(pairs)[PAIR_FEATURES])
                order=np.argsort(-np.asarray(pair_scores),kind='stable')
                recs=[]
                for rank,pos in enumerate(order[:3],start=1):
                    ia,ib=pair_rows[int(pos)]
                    recs.append((rank,ia,ib,float(pair_scores[pos])))
                    df.loc[[ia,ib],'qp_pair_score']=float(pair_scores[pos])
                qp_pair_recommendations[str(race_id)]=recs

# No estimated PLACE price: EV/stake stays blank unless the official HKJC quote exists.
df['win_ev'] = df['pred_win_prob'] * pd.to_numeric(df['獨贏賠率'],errors='coerce')
df['place_ev'] = df['pred_place_prob'] * pd.to_numeric(df['位置賠率'],errors='coerce')
df['ev'] = df['win_ev']

# 6. 計算雙引擎「抗噪穩定綜合分數」 (僅結合臨場落飛資金)
drop_bonus = np.clip(df['odds_drop_pct'] / 100.0, -0.35, 0.30)
total_tactical_adj = 1.0 + drop_bonus  # 👈 拆掉後面的三個變數
df['smart_score_win'] = df['pred_win_prob'] * total_tactical_adj
df['smart_score_place'] = df['pred_place_prob'] * total_tactical_adj
df['smart_score'] = df['smart_score_win']

# 💰 7. 雙重凱利注碼精算 (獨贏注碼 & 位置注碼)
safe_win_odds = np.maximum(df['獨贏賠率'], 1.01)
df['kelly_pct_win'] = np.clip((df['win_ev'] - 1.0) / (safe_win_odds - 1.0), 0.0, 1.0).fillna(0.0)
df['suggested_stake_win'] = np.floor((total_bankroll * df['kelly_pct_win'] * kelly_fraction) / 10) * 10
df['suggested_stake'] = df['suggested_stake_win']

safe_place_odds = np.maximum(df['位置賠率'], 1.01)
df['kelly_pct_place'] = np.clip((df['place_ev'] - 1.0) / (safe_place_odds - 1.0), 0.0, 1.0).fillna(0.0)
df['suggested_stake_place'] = np.floor((total_bankroll * df['kelly_pct_place'] * kelly_fraction) / 10) * 10

# 🚀 核心雙軌過濾與排序邏輯 (支援 bet_type 分流)
def filter_and_sort_group(group, mode, min_ev_val, min_odds_val, max_odds_val, bet_type="win"):
    if bet_type == "win":
        if mode.startswith("🎯"):
            cond = group['pred_win_prob'].notna()
            sort_col = 'pred_win_prob'
        else:
            cond = (group['win_ev'] >= min_ev_val) & (group['獨贏賠率'] >= min_odds_val) & (group['獨贏賠率'] <= max_odds_val)
        if mode.startswith("🛡️"):
            cond = cond & (group['odds_drop_pct'] > -25.0)
            sort_col = 'smart_score_win'
        elif not mode.startswith("🎯"):
            sort_col = 'win_ev'
    else:
        if mode.startswith("🎯"):
            cond = group['pred_place_prob'].notna()
            sort_col = 'pred_place_prob'
        else:
            # 位置賠率較低，將獨贏最低賠率門檻除以 3 作為位置賠率下限
            min_pla_odds = max(1.01, min_odds_val / 3.0)
            cond = (group['place_ev'] >= min_ev_val) & (group['位置賠率'] >= min_pla_odds)
        if mode.startswith("🛡️"):
            cond = cond & (group['odds_drop_pct'] > -25.0)
            sort_col = 'smart_score_place'
        elif not mode.startswith("🎯"):
            sort_col = 'place_ev'
            
    return group[cond].sort_values(by=sort_col, ascending=False).reset_index(drop=True)

def place_slots_for_race(group):
    rank = pd.to_numeric(group.get('numeric_rank', pd.Series(index=group.index,dtype=float)), errors='coerce')
    n = int(rank.between(1, 90).sum())
    return 2 if 4 <= n <= 6 else (3 if n >= 7 else 0)

# 回測使用手動上傳的完賽資料
df_backtest = df.copy()
backtest_ready = False
backtest_cutoff_excluded_races = 0
backtest_incomplete_excluded_races = 0
backtest_final_odds_match_rate = np.nan
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
            win_dividend_aliases = ('獨贏派彩','WIN派彩','官方獨贏派彩','Win Dividend','win_dividend')
            win_dividend_col = next((c for c in win_dividend_aliases if c in result_map.columns), None)
            if win_dividend_col:
                rename_map[win_dividend_col] = '_result_win_dividend'
            place_dividend_aliases = ('位置派彩','PLACE派彩','官方位置派彩','Place Dividend','place_dividend')
            place_dividend_col = next((c for c in place_dividend_aliases if c in result_map.columns), None)
            if place_dividend_col:
                rename_map[place_dividend_col] = '_result_place_dividend'
            result_map = result_map.drop_duplicates(['_race_key', '_horse_key'], keep='last')
            result_map = result_map[['_race_key', '_horse_key'] + [c for c in rename_map if c in result_map.columns]]
            result_map = result_map.rename(columns=rename_map)

            df_backtest['_race_key'] = df_backtest['賽事編號'].map(normalize_race_id)
            df_backtest['_horse_key'] = df_backtest['馬號'].map(normalize_horse_no)
            df_backtest = df_backtest.merge(
                result_map, on=['_race_key', '_horse_key'], how='left', validate='many_to_one', indicator='_result_match'
            )
            if archived_prediction_card and '_result_win_odds' in df_backtest.columns:
                saved_win = pd.to_numeric(df_backtest['獨贏賠率'], errors='coerce')
                final_win = pd.to_numeric(df_backtest['_result_win_odds'], errors='coerce')
                comparable_odds = saved_win.gt(1.0) & final_win.gt(1.0)
                if int(comparable_odds.sum()) >= 20:
                    backtest_final_odds_match_rate = float(
                        np.isclose(saved_win[comparable_odds], final_win[comparable_odds], rtol=0.0, atol=0.01).mean()
                    )
                    if backtest_final_odds_match_rate >= 0.95:
                        st.sidebar.error(
                            f'⚠️ 存檔WIN價有 {backtest_final_odds_match_rate:.0%} 與完賽CSV官方終盤價相同；本模型含賠率特徵，以下命中率屬事後診斷，不能視為賽前實戰準確率。'
                        )
            df_backtest['名次'] = df_backtest['_result_rank']
            df_backtest['numeric_rank'] = pd.to_numeric(df_backtest['名次'], errors='coerce').fillna(99)
            if '_result_win_odds' in df_backtest.columns:
                df_backtest['回測獨贏賠率'] = pd.to_numeric(df_backtest['_result_win_odds'], errors='coerce')
            if '_result_win_dividend' in df_backtest.columns:
                df_backtest['回測獨贏派彩'] = pd.to_numeric(df_backtest['_result_win_dividend'], errors='coerce')
            if '_result_place_dividend' in df_backtest.columns:
                df_backtest['回測位置派彩'] = pd.to_numeric(df_backtest['_result_place_dividend'], errors='coerce')
            has_official_result = df_backtest['_result_match'].eq('both') & df_backtest['_result_rank'].notna()
            matched = int(has_official_result.sum())
            backtest_ready = matched > 0
            if backtest_ready:
                df_backtest = df_backtest[df_backtest['_result_rank'].notna()].copy()
                result_rank_text = df_backtest['_result_rank'].astype(str).str.strip()
                nonstarter = result_rank_text.str.match(r'(?i)^(WV|SCR|WD|WITHDRAWN|退出|取消|未出賽)$', na=False)
                active_result_rows = df_backtest.loc[~nonstarter].copy()
                score_cols = ['pred_win_prob', 'pred_place_prob']
                available_score_cols = [c for c in score_cols if c in active_result_rows.columns]
                if available_score_cols:
                    incomplete_mask = active_result_rows[available_score_cols].isna().any(axis=1)
                    incomplete_races = set(active_result_rows.loc[incomplete_mask, '賽事編號'].astype(str))
                    backtest_incomplete_excluded_races = len(incomplete_races)
                    if incomplete_races:
                        df_backtest = df_backtest[~df_backtest['賽事編號'].astype(str).isin(incomplete_races)].copy()
                        st.sidebar.warning(
                            f'已排除 {backtest_incomplete_excluded_races} 場不完整回測：完賽資料顯示有正式參賽馬，但該馬沒有有效模型分數。'
                        )
                selected_meta = []
                if win_mode == 'ranker_fixed': selected_meta.append(meta_win_fixed)
                if win_mode == 'ranker_dynamic': selected_meta.append(meta_win_dynamic)
                if place_mode == 'place_fixed': selected_meta.append(meta_win_fixed)
                if place_mode == 'place_dynamic': selected_meta.append(meta_win_dynamic)
                if qp_mode == 'qp_fixed': selected_meta.append(meta_win_fixed)
                if qp_mode == 'qp_dynamic': selected_meta.append(meta_win_dynamic)
                train_end_dates = [pd.to_datetime(m.get('final_fit_end_date'),errors='coerce') for m in selected_meta]
                train_end_dates = [d.normalize() for d in train_end_dates if pd.notna(d)]
                if train_end_dates:
                    model_cutoff = max(train_end_dates)
                    result_dates = pd.to_datetime(
                        df_backtest['賽事編號'].astype(str).str.extract(r'(20\d{6})',expand=False),
                        format='%Y%m%d',errors='coerce'
                    ).dt.normalize()
                    contaminated = result_dates.le(model_cutoff) | result_dates.isna()
                    backtest_cutoff_excluded_races = int(df_backtest.loc[contaminated,'賽事編號'].nunique())
                    if contaminated.any():
                        df_backtest = df_backtest.loc[~contaminated].copy()
                        st.sidebar.warning(
                            f'已排除 {backtest_cutoff_excluded_races} 場不適合做外測的賽事：所選候選模型最終訓練資料截至 '
                            f'{model_cutoff:%Y-%m-%d}，同日或更早賽事不能用此最終模型回測。'
                        )
                    backtest_ready = not df_backtest.empty
                st.sidebar.success(f"✅ {result_source_label}已配對：{matched} 匹；完整可回測賽事 {df_backtest['賽事編號'].nunique() if backtest_ready else 0} 場")
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
        target_race = col_r.selectbox("當前操作／查看場次", races_available, key="target_race_select")
    else:
        target_race = col_r.number_input("當前操作／查看場次", min_value=1, max_value=15, value=1, step=1)

    selected_race_rows = df[df['賽事編號'].map(extract_race_no) == int(target_race)].copy()
    selected_race_win = pd.to_numeric(selected_race_rows.get('獨贏賠率', pd.Series(index=selected_race_rows.index, dtype=float)), errors='coerce')
    selected_race_active = ~selected_race_rows.get('參賽狀態', pd.Series('參賽', index=selected_race_rows.index)).map(is_scratched)
    selected_race_valid_win = selected_race_win.gt(1.0) & selected_race_active
    selected_race_scored = (
        selected_race_rows.get('pred_win_prob', pd.Series(index=selected_race_rows.index, dtype=float)).notna()
        & selected_race_rows.get('pred_place_prob', pd.Series(index=selected_race_rows.index, dtype=float)).notna()
    )
    if not selected_race_rows.empty and not bool(selected_race_scored.any()):
        missing_horses = selected_race_rows.loc[selected_race_active & ~selected_race_win.gt(1.0), '馬號'].astype(str).tolist()
        if selected_race_active.any() and not (selected_race_valid_win | ~selected_race_active).all():
            missing_text = '、'.join(missing_horses) if missing_horses else '未知'
            st.warning(
                f"第 {target_race} 場 AI 勝率／入位率暫無有效分數：目前只有 {int(selected_race_valid_win.sum())}/{int(selected_race_active.sum())} 匹仍參賽馬有有效官方WIN賠率；缺價馬號：{missing_text}。"
                "此模型需要該場完整WIN盤特徵。請選中本場後按「立即更新目前場次賠率」；若HKJC尚未開盤，稍後再試。系統不會用估算值補價。"
            )
        elif not selected_race_active.any():
            st.warning(f'第 {target_race} 場的馬匹均被標記為退出，請確認所選賽卡／HKJC狀態。')
        elif not history_features_ok:
            st.error('本場WIN盤已有有效報價，但History載入失敗，模型分數已停止；請查看左側History錯誤訊息。')
        else:
            st.error('本場WIN盤看似完整，但模型仍未產生分數。請檢查模型檔是否成功載入，並查看左側模型載入提示。')
    elif not selected_race_rows.empty:
        st.caption(
            f"第 {target_race} 場模型分數狀態：WIN有效報價 {int(selected_race_valid_win.sum())}/{int(selected_race_active.sum())} 匹有效參賽馬；"
            f"已有AI分數 {int((selected_race_scored & selected_race_active).sum())}/{int(selected_race_active.sum())} 匹有效參賽馬。"
        )

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
    auto_refresh = auto_col.checkbox("自動更新目前場次", value=False, disabled=archived_prediction_card)
    refresh_seconds = interval_col.selectbox(
        "更新間隔（秒）", [10, 30, 60, 120, 300], index=1, disabled=not auto_refresh
    )
    manual_refresh = button_col.button("🔄 立即更新目前場次賠率", use_container_width=True, disabled=archived_prediction_card)
    lock_baseline = lock_col.button("📌 鎖定當前賠率為早盤基準", use_container_width=True, disabled=archived_prediction_card, help="在開跑前 15~20 分鐘按一下鎖定早盤賠率，之後更新賠率時即可自動計算落飛幅度！")
    if archived_prediction_card:
        st.info("🗂️ 歷史回測模式：目前顯示該預測 CSV 存檔的賽前價格，不會向 HKJC 刷新或把存檔價標成即時盤口。")

    # 控制列 2：單場買時盤口凍結器
    f_col1, f_col2, f_col3 = st.columns([2.2, 1.8, 2.6])
    is_target_frozen = int(target_race) in st.session_state['frozen_races']
    if not is_target_frozen:
        if f_col1.button(f"🔒 凍結第 {target_race} 場【買時賠率】 (下注後點擊)", use_container_width=True, type="primary", disabled=archived_prediction_card):
            st.session_state['frozen_races'][int(target_race)] = datetime.now(hk_timezone).strftime("%H:%M:%S")
            st.rerun()
    else:
        frozen_t = st.session_state['frozen_races'][int(target_race)]
        if f_col1.button(f"🔓 解除第 {target_race} 場凍結 (已鎖於 {frozen_t})", use_container_width=True, disabled=archived_prediction_card):
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
            base_value = pd.to_numeric(r['獨贏賠率'], errors='coerce')
            if pd.notna(base_value) and base_value > 1.0:
                base_map[k] = float(base_value)
        st.session_state['baseline_odds_map'] = base_map
        st.session_state['baseline_locked_time'] = datetime.now(hk_timezone).strftime("%H:%M:%S")
        st.rerun()

    if auto_refresh and not archived_prediction_card:
        st_autorefresh(interval=int(refresh_seconds * 1000), key="hkjc_live_odds_autorefresh")

    if (auto_refresh or manual_refresh) and is_target_frozen:
        st.info(f"第 {target_race} 場已凍結；本輪不會送出賠率請求。")
    elif auto_refresh or manual_refresh:
        with st.spinner(f"正在讀取第 {target_race} 場 WIN／PLACE 即時賠率…"):
            live_by_race, server_updated_at, error = fetch_live_odds(api_date, venue_code, int(target_race))

        if live_by_race:
            df_temp = st.session_state['df_data'].copy()
            row_races = df_temp['賽事編號'].map(extract_race_no)
            row_horses = df_temp['馬號'].map(normalize_horse_no)
            updated_races_count = 0
            skipped_frozen = []

            for race_no, markets in live_by_race.items():
                r_int = int(race_no)
                if r_int != int(target_race):
                    continue
                if r_int in st.session_state['frozen_races']:
                    skipped_frozen.append(f"R{r_int}")
                    continue
                updated_races_count += 1
                race_mask = row_races == r_int
                scratched_horses = markets.get('SCRATCHED', set())
                if scratched_horses:
                    if '參賽狀態' not in df_temp.columns:
                        df_temp['參賽狀態'] = '參賽'
                    scratch_mask = race_mask & row_horses.isin(scratched_horses)
                    df_temp.loc[scratch_mask, '參賽狀態'] = '退出'
                    df_temp.loc[scratch_mask, ['獨贏賠率', '位置賠率']] = np.nan
                for horse_no, value in markets['WIN'].items():
                    if horse_no in scratched_horses:
                        continue
                    horse_mask = race_mask & (row_horses == horse_no)
                    df_temp.loc[horse_mask, '獨贏賠率'] = value
                for horse_no, value in markets.get('PLA', {}).items():
                    if horse_no in scratched_horses:
                        continue
                    horse_mask = race_mask & (row_horses == horse_no)
                    df_temp.loc[horse_mask, '位置賠率'] = value

            base_map = st.session_state.get('baseline_odds_map', {}).copy()
            new_baseline_count = 0
            for _, r in df_temp.loc[row_races == int(target_race)].iterrows():
                k = f"{normalize_race_id(r['賽事編號'])}_{normalize_horse_no(r['馬號'])}"
                base_value = pd.to_numeric(r['獨贏賠率'], errors='coerce')
                if k not in base_map and pd.notna(base_value) and base_value > 1.0:
                    base_map[k] = float(base_value)
                    new_baseline_count += 1
            if new_baseline_count:
                st.session_state['baseline_odds_map'] = base_map
                if st.session_state.get('baseline_locked_time') in (None, '未鎖定', '初始載入值'):
                    st.session_state['baseline_locked_time'] = f"首筆官方實報基準 ({datetime.now(hk_timezone).strftime('%H:%M:%S')})"

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
    form_mode_str = f"WIN:{'Dynamic' if win_dynamic_form else 'Fixed'} / PLACE:{'Dynamic' if place_dynamic_form else 'Fixed'} / QP:{qp_choice}"
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
        sorted_win = filter_and_sort_group(group, strategy_mode, min_win_ev, min_odds, max_odds, bet_type="win")
        sorted_place = filter_and_sort_group(group, strategy_mode, min_place_ev, min_odds, max_odds, bet_type="place")
        qp_candidates = group.loc[group['pred_place_prob'].notna()].sort_values(['pred_place_prob','馬號'],ascending=[False,True],kind='mergesort')

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
            
        if len(sorted_place) >= 1:
            p1 = sorted_place.iloc[0]
            q_pick = f"單膽位置: {p1['馬號']} ({p1['馬名']}) [位EV:{p1['place_ev']:.2f}] 💰${p1['suggested_stake_place']:.0f}"
        elif not qp_candidates.empty:
            p1 = qp_candidates.iloc[0]
            q_pick = f"單膽位置候選: {p1['馬號']} ({p1['馬名']}) [AI入位率:{p1['pred_place_prob']*100:.0f}%；未有官方位置價，不計EV]"
        else:
            q_pick = "⚠️ 位置模型未有有效分數"

        pair_rows = qp_pair_recommendations.get(str(race_id), [])
        if qp_mode in ('qp_fixed','qp_dynamic') and pair_rows:
            pair_text=[]
            for rank, ia, ib, score in pair_rows[:3]:
                a,b=df.loc[ia],df.loc[ib]
                pair_text.append(f"{int(a['馬號'])}+{int(b['馬號'])}")
            qp_pick = '直接QP排序：' + '、'.join(pair_text)
        elif len(qp_candidates) >= 2:
            p1,p2=qp_candidates.iloc[0],qp_candidates.iloc[1]
            q_pick = f"{p1['馬號']} + {p2['馬號']} ({p1['馬名']} / {p2['馬名']}) [AI入位率:{p1['pred_place_prob']*100:.0f}% & {p2['pred_place_prob']*100:.0f}%]"
            if len(qp_candidates) >= 3:
                p3=qp_candidates.iloc[2]
                qp_pick = f"{p1['馬號']} + {p2['馬號']} 或 {p1['馬號']} + {p3['馬號']}"
            else:
                qp_pick = f"{p1['馬號']} + {p2['馬號']} (一注QP；按PLACE排序)"
        else:
            qp_pick = "⚠️ 至少需要兩匹有有效PLACE模型分數才可組QP"

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
    st.markdown(f"#### 🔍 第 {target_race} 場 — 臨場落飛與雙模型實力深度雷達 ({status_tag})")
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

# ---------------- 分頁 2: 賽後回測 ----------------
with tab2:
    st.subheader("📊 雙引擎多彩種策略回測總覽（命中率與官方派彩）")
    st.caption(f"目前回測套用之排序邏輯：`{strategy_mode}` ｜ 算分引擎：`{form_mode_str}` ｜ 模型：`{engine_str}`；測試採賽日前History")
    if archived_prediction_card:
        if pd.notna(backtest_final_odds_match_rate) and backtest_final_odds_match_rate >= 0.95:
            st.error(f"事後資訊警告：本次存檔WIN價與官方終盤價有 {backtest_final_odds_match_rate:.0%} 一致；由於模型輸入包括WIN賠率、市場勝率與賠率排名，這次命中率不能當作賽前預測表現。")
        else:
            st.warning("歷史回測使用所選 prediction CSV 內存檔的WIN／PLACE價格；如檔案沒有報價時間戳，不能確認價格在開跑前何時取得，也不能重播原App某一分鐘的即時畫面。精確重播需用開跑前保存的完整賽卡／模型分數快照。")
    if df_backtest.empty or not backtest_ready:
        if backtest_cutoff_excluded_races:
            st.warning("已排除的賽事均不晚於候選模型訓練截止日，不能作為該最終模型的無洩漏回測。請使用截止日之後的賽事結果。")
        else:
            st.warning("請在左側上傳已完賽的結果 CSV。需有『賽事編號』、『馬號』及『名次』。")
    else:
        st.info("為避免用同一批賽果挑參數造成過度擬合，本頁不再提供單日ROI尋優。請以較早賽期定參數，再用未參與調整的後續賽期驗證。")

        sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
            "🥇 獨贏 (Win Engine)", "🥈 單匹位置 (Place Engine)", "🔗 位置Q (Place Engine)", "🥉 3匹位置包抄 (Place Engine)"
        ])

        with sub_tab1:
            st.markdown("#### 🥇 獨贏（WIN）策略回測 — 使用目前選定的WIN引擎")
            win_invested, win_return, win_bets, win_hits = 0, 0, 0, 0
            win_missing_dividend = False
            win_records = []
            for race_id, group in df_backtest.groupby('賽事編號'):
                sorted_group = filter_and_sort_group(group, strategy_mode, min_win_ev, min_odds, max_odds, bet_type="win")
                if len(sorted_group) > 0:
                    pick = sorted_group.iloc[0]
                    raw_stake = pick.get('suggested_stake_win', 10)
                    stake = max(10, float(raw_stake) if pd.notna(raw_stake) else 10)
                    
                    win_bets += 1
                    win_invested += stake
                    is_hit = (pick['numeric_rank'] == 1)
                    settlement_win_dividend = pick.get('回測獨贏派彩', np.nan)
                    win_dividend = float(settlement_win_dividend) if pd.notna(settlement_win_dividend) and float(settlement_win_dividend) > 0 else np.nan
                    payout = stake * win_dividend / 10.0 if is_hit and pd.notna(win_dividend) else np.nan if is_hit else 0.0
                    if is_hit:
                        win_hits += 1
                        if pd.notna(payout):
                            win_return += payout
                        else:
                            win_missing_dividend = True
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
                        '官方獨贏派彩（每$10）': round(win_dividend, 2) if pd.notna(win_dividend) else '未提供',
                        'AI勝率': f"{pick['pred_win_prob']*100:.1f}%",
                        '買時 獨贏EV': round(pick['win_ev'], 2),
                        '凱利注碼': f"${stake:.0f}",
                        '結果': result_str,
                        '派彩': f"${payout:.1f}" if pd.notna(payout) else '不計（缺官方派彩）',
                        '淨盈虧': f"${payout - stake:.1f}" if pd.notna(payout) else '不計（缺官方派彩）'
                    })
            if win_bets > 0:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("投注場數", f"{win_bets} 場")
                col2.metric("命中場數", f"{win_hits} 場", f"勝率: {win_hits/win_bets*100:.1f}%")
                col3.metric("總投入成本", f"${win_invested:,.0f}")
                if not win_missing_dividend:
                    roi = ((win_return - win_invested) / win_invested) * 100
                    col4.metric("官方派彩ROI", f"{roi:.2f}%", f"官方回收 ${win_return:,.1f}")
                else:
                    col4.metric("ROI", "未計算", "命中注缺官方派彩")
                st.markdown("##### 📝 獨贏明細（派彩需官方每$10派彩欄）")
                st.dataframe(pd.DataFrame(win_records), use_container_width=True, hide_index=True)
            else:
                st.info("💡 目前設定下沒有符合獨贏出手的場次。")

        with sub_tab2:
            st.markdown("#### 🥈 位置（PLACE）策略回測 — 使用目前選定的PLACE引擎")
            place_invested, place_return, place_bets, place_hits = 0, 0, 0, 0
            place_records = []
            for race_id, group in df_backtest.groupby('賽事編號'):
                sorted_group = filter_and_sort_group(group, strategy_mode, min_place_ev, min_odds, max_odds, bet_type="place")
                if len(sorted_group) > 0:
                    pick = sorted_group.iloc[0]
                    raw_stake = pick.get('suggested_stake_place', 10)
                    stake = max(10, float(raw_stake) if pd.notna(raw_stake) else 10)
                    
                    place_bets += 1
                    place_invested += stake
                    slots = place_slots_for_race(group)
                    is_hit = (1 <= pick['numeric_rank'] <= slots) if slots else False
                    settlement_place_dividend = pick.get('回測位置派彩', np.nan)
                    p_dividend = float(settlement_place_dividend) if pd.notna(settlement_place_dividend) and float(settlement_place_dividend) > 0 else np.nan
                    payout = stake * p_dividend / 10.0 if is_hit and pd.notna(p_dividend) else np.nan
                    if is_hit:
                        place_hits += 1
                        result_str = "✅ 命中位置"
                    else:
                        result_str = "❌ 未入位"
                        
                    place_records.append({
                        '賽事編號': race_id,
                        '賽道': pick.get('賽道', 'A'),
                        '投注馬號': f"{pick['馬號']} ({pick['馬名']})",
                        '上仗名次': "新馬/無" if pick['real_last_rank'] == 6.5 else f"第 {int(pick['real_last_rank'])} 名",
                        '實際名次': str(pick['名次']).replace('.0', ''),
                        'AI入位率': f"{pick['pred_place_prob']*100:.1f}%",
                        '買時 位置EV': round(pick['place_ev'], 2),
                        '官方位置派彩（每$10）': round(p_dividend, 2) if pd.notna(p_dividend) else '未提供',
                        '凱利注碼': f"${stake:.0f}",
                        '結果': result_str,
                        '派彩': f"${payout:.1f}" if pd.notna(payout) else '不計（缺官方派彩）',
                        '淨盈虧': f"${payout - stake:.1f}" if pd.notna(payout) else '不計（缺官方派彩）'
                    })
            if place_bets > 0:
                col1, col2, col3 = st.columns(3)
                col1.metric("投注場數", f"{place_bets} 場")
                col2.metric("命中位置場數", f"{place_hits} 場", f"位置率: {place_hits/place_bets*100:.1f}%")
                col3.metric("ROI", "未計算", "需要官方位置派彩")
                st.caption("位置報價不等於官方派彩；沒有官方派彩欄時只報命中率。")
                st.markdown("##### 📝 位置明細")
                st.dataframe(pd.DataFrame(place_records), use_container_width=True, hide_index=True)
            else:
                st.info("💡 目前設定下沒有符合位置出手的場次。")

        with sub_tab3:
            st.markdown("#### 🔗 位置Q (QP) 策略回測 — 由 `my_hkjc_place_model.pkl` 驅動")
            qp_invested, qp_return, qp_bets, qp_hits = 0, 0, 0, 0
            qp_records = []
            for race_id, group in df_backtest.groupby('賽事編號'):
                pair_rows = qp_pair_recommendations.get(str(race_id), [])
                chosen_pair = None
                if qp_mode in ('qp_fixed','qp_dynamic') and pair_rows:
                    _, ia, ib, _ = pair_rows[0]
                    pair_group = group.loc[group.index.isin([ia, ib])]
                    if len(pair_group) == 2:
                        chosen_pair = (pair_group.loc[ia], pair_group.loc[ib])
                if chosen_pair is not None:
                    top1, top2 = chosen_pair
                else:
                    sorted_group = filter_and_sort_group(group, strategy_mode, min_place_ev, min_odds, max_odds, bet_type="place")
                    chosen_pair = (sorted_group.iloc[0], sorted_group.iloc[1]) if len(sorted_group) >= 2 else None
                if chosen_pair is not None:
                    top1, top2 = chosen_pair
                    stake = max(10, min(float(top1.get('suggested_stake_place',10) if pd.notna(top1.get('suggested_stake_place')) else 10), float(top2.get('suggested_stake_place',10) if pd.notna(top2.get('suggested_stake_place')) else 10)))
                    
                    qp_bets += 1
                    qp_invested += stake
                    slots = place_slots_for_race(group)
                    is_hit = slots > 0 and (1 <= top1['numeric_rank'] <= slots) and (1 <= top2['numeric_rank'] <= slots)
                    
                    p1_settlement = top1.get('回測位置賠率', np.nan)
                    p2_settlement = top2.get('回測位置賠率', np.nan)
                    # QP dividend depends on the official combination pool; it cannot be estimated from two PLACE prices.
                    official_qp_dividend = np.nan
                    payout = np.nan
                    if is_hit:
                        qp_hits += 1
                        result_str = "✅ 命中位置Q"
                    else:
                        result_str = "❌ 落空"
                        
                    qp_records.append({
                        '賽事編號': race_id,
                        'QP 組合': f"{top1['馬號']} + {top2['馬號']} ({top1['馬名']} / {top2['馬名']})",
                        'AI入位率': f"{top1['pred_place_prob']*100:.0f}% / {top2['pred_place_prob']*100:.0f}%",
                        '實際名次': f"首選:第{str(top1['名次']).replace('.0','')}名 | 次選:第{str(top2['名次']).replace('.0','')}名",
                        '官方QP派彩': '未提供',
                        '凱利注碼': f"${stake:.0f}",
                        '結果': result_str,
                        '派彩': '不計（缺官方QP派彩）',
                        '淨盈虧': '不計（缺官方QP派彩）'
                    })
            if qp_bets > 0:
                col1, col2, col3 = st.columns(3)
                col1.metric("投注場數", f"{qp_bets} 場")
                col2.metric("命中 QP 場數", f"{qp_hits} 場", f"QP命中率: {qp_hits/qp_bets*100:.1f}%")
                col3.metric("ROI", "未計算", "需要官方QP派彩")
                st.caption("QP命中按兩匹均入位計；缺逐組官方QP派彩時不估算回報。")
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
                sorted_group = filter_and_sort_group(group, strategy_mode, min_place_ev, min_odds, max_odds, bet_type="place")
                picks = sorted_group.head(3)
                
                for _, pick in picks.iterrows():
                    raw_stake = pick.get('suggested_stake_place', 10)
                    stake = max(10, float(raw_stake) if pd.notna(raw_stake) else 10)
                    
                    p3_bets += 1
                    p3_invested += stake
                    slots = place_slots_for_race(group)
                    is_hit = (1 <= pick['numeric_rank'] <= slots) if slots else False
                    
                    settlement_place_dividend = pick.get('回測位置派彩', np.nan)
                    p_dividend = float(settlement_place_dividend) if pd.notna(settlement_place_dividend) and float(settlement_place_dividend) > 0 else np.nan
                    payout = stake * p_dividend / 10.0 if is_hit and pd.notna(p_dividend) else np.nan
                    if is_hit:
                        p3_hits += 1
                        result_str = "✅ 命中位置"
                    else:
                        result_str = "❌ 未入前三"
                        
                    p3_records.append({
                        '賽事編號': race_id,
                        '投注馬號': f"{pick['馬號']} ({pick['馬名']})",
                        '上仗名次': "新馬/無" if pick['real_last_rank'] == 6.5 else f"第 {int(pick['real_last_rank'])} 名",
                        '實際名次': str(pick['名次']).replace('.0', ''),
                        'AI入位率': f"{pick['pred_place_prob']*100:.1f}%",
                        '官方位置派彩（每$10）': round(p_dividend, 2) if pd.notna(p_dividend) else '未提供',
                        '買時 位置EV': round(pick['place_ev'], 2),
                        '凱利注碼': f"${stake:.0f}",
                        '結果': result_str,
                        '派彩': f"${payout:.1f}" if pd.notna(payout) else '不計（缺官方派彩）',
                        '淨盈虧': f"${payout - stake:.1f}" if pd.notna(payout) else '不計（缺官方派彩）'
                    })
                    
            if p3_bets > 0:
                col1, col2, col3 = st.columns(3)
                col1.metric("總投注注數", f"{p3_bets} 注")
                col2.metric("命中注數", f"{p3_hits} 注", f"位置率: {p3_hits/p3_bets*100:.1f}%")
                col3.metric("ROI", "未計算", "需要官方位置派彩")
                st.caption("此處只報入位命中率；位置報價不等於官方派彩。")
                st.markdown("##### 📝 3匹位置包抄明細")
                st.dataframe(pd.DataFrame(p3_records), use_container_width=True, hide_index=True)
            else:
                st.info("💡 目前設定下沒有符合出手的場次。")
