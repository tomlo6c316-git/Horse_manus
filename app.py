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

st.title("🐎 HKJC 旗艦 15 大特徵 AI 預測系統 (支援即時賠率)")
st.markdown("---")

# 初始化模擬投注紀錄 (Paper Trading)
if 'simulated_bets' not in st.session_state:
    st.session_state['simulated_bets'] = pd.DataFrame(columns=[
        '下注時間', '賽事編號', '場次', '馬號', '馬名', '玩法', '買入賠率', '注碼'
    ])

# 設定 repo 內的模型及預測 CSV 路徑
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / 'my_hkjc_model.pkl'
PREDICTION_CSV_PATH = APP_DIR / 'prediction.csv'
FEATURE_COLS_15 = [
    'market_implied_prob', '獨贏賠率', 'odds_rank', 'is_favorite',
    '排位檔位', 'weight_diff', 'weight_rank',
    'jockey_win_rate', 'trainer_win_rate', 'combo_win_rate',
    'horse_win_rate', 'horse_last_rank',
    '距離', 'horse_surface_win_rate', 'horse_dist_win_rate',
]

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

if model is None:
    st.error(f"⚠️ 找不到 AI 模型檔 `{MODEL_PATH}`！請確認模型是否已上傳至正確目錄。")
    st.stop()

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
        columns = list(race_table.columns)
        horse_no_col = find_column(columns, ('馬號', '編號'), columns[0] if columns else None)
        horse_name_col = find_column(columns, ('馬名',))
        weight_col = find_column(columns, ('負磅', '磅'))
        jockey_col = find_column(columns, ('騎師',))
        trainer_col = find_column(columns, ('練馬師',))
        draw_col = find_column(columns, ('檔位', '排位', '檔'))
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
                'racecourse_code': race_venue,
            })
        time.sleep(0.25)

    if not rows:
        raise ValueError('HKJC 沒有回傳排位表。')

    result = pd.DataFrame(rows)
    return result.drop_duplicates(['賽事編號', '馬號'], keep='last').reset_index(drop=True)


# 預測賽卡可一鍵從 HKJC 抓取，或沿用 GitHub repo 的 prediction.csv
st.sidebar.header("📂 預測資料")
race_date_choice = st.sidebar.date_input('選擇賽事日期', value=datetime.now().date(), key='racecard_date_choice')
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
        st.sidebar.success(f'已載入 {len(fetched_card)} 匹馬。')
    except Exception as exc:
        st.sidebar.error(f'抓取排位失敗：{type(exc).__name__}: {exc}')

current_fetched_card = st.session_state.get('fetched_prediction_df')
if current_fetched_card is not None:
    csv_bytes = current_fetched_card.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.sidebar.download_button(
        '⬇️ 下載 prediction.csv', data=csv_bytes, file_name='prediction.csv',
        mime='text/csv', use_container_width=True, key='download_prediction_csv'
    )
else:
    st.sidebar.caption(f"未抓取新賽卡時，預測資料由 repo 的 {PREDICTION_CSV_PATH.name} 載入。")

backtest_file = st.sidebar.file_uploader(
    "回測用：上傳已完成賽事 CSV（需要賽事編號、馬號、名次）",
    type=['csv'],
    key='backtest_results_upload',
    help='上傳已完賽的結果檔案，用來進行各策略與模擬結算的覆盤。',
)

st.sidebar.markdown("---")
st.sidebar.header("⚙ 投注策略參數設定")
min_ev = st.sidebar.slider("最小期望值 (EV 門檻)", 0.0, 1.5, 0.0, 0.05)
min_odds = st.sidebar.number_input("最低獨贏賠率", min_value=1.0, max_value=50.0, value=3.0)
max_odds = st.sidebar.number_input("最高獨贏賠率", min_value=1.0, max_value=100.0, value=20.0)

# ==========================================
# 🛠️ 歷史賽果 15 大特徵資料抓取工具 (新增整合)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("🛠️ 歷史賽果 15 大特徵抓取器")
st.sidebar.caption("輸入指定賽事日期，自動抓取該日賽果並產出可供模型訓練的 CSV。")

tool_date_choice = st.sidebar.date_input("選擇目標賽事日期", value=datetime.now().date(), key="tool_date_input")
tool_season = st.sidebar.text_input("輸入馬季 (例如 23/24 或 25/26)", value="25/26")

if st.sidebar.button("📥 抓取並產出訓練 CSV", use_container_width=True, key="run_tool_button"):
    date_str_tool = tool_date_choice.strftime('%Y/%m/%d')
    date_for_id_tool = tool_date_choice.strftime('%Y%m%d')
    
    all_races_data_tool = []
    headers_tool = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    success_count = 0
    for race_no in range(1, 13):
        status_text.text(f"正在抓取第 {race_no} 場資料...")
        url = f"https://racing.hkjc.com/racing/information/Chinese/Racing/LocalResults.aspx?RaceDate={date_str_tool}&RaceNo={race_no}"
        try:
            resp = requests.get(url, headers=headers_tool, timeout=15)
            resp.encoding = 'utf-8' 
            html_text = resp.text
            
            if "沒有相關資料" in html_text or "No information" in html_text:
                break

            distance = 1200
            dist_match = re.search(r'(\d{4})\s*米', html_text)
            if dist_match:
                distance = int(dist_match.group(1))
            surface_type = "草地" if "草地" in html_text else "泥地"
            
            tables = pd.read_html(StringIO(html_text))
            target_df = None
            for t in tables:
                check_str = "".join([str(c) for c in t.columns])
                if len(t) > 0:
                    check_str += "".join([str(c) for c in t.iloc[0].values])
                if '名次' in check_str and '馬號' in check_str and '獨贏' in check_str:
                    if not any('名次' in str(c) for c in t.columns):
                        t.columns = t.iloc[0]
                        t = t.drop(0)
                    target_df = t
                    break
                    
            if target_df is not None and not target_df.empty:
                cols = target_df.columns.astype(str)
                odds_col = next((col for col in cols if '獨贏' in col), None)
                weight_col = next((col for col in cols if '負磅' in col), None)
                draw_col = next((col for col in cols if '檔位' in col or '排位' in col), None)
                
                for idx, row in target_df.iterrows():
                    rank = str(row.get('名次', '')).strip()
                    horse_no = str(row.get('馬號', '')).strip()
                    
                    if horse_no.isdigit():
                        horse_name_raw = str(row.get('馬名', ''))
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
                            '騎師': str(row.get('騎師', '')),
                            '練馬師': str(row.get('練馬師', '')),
                            '實際負磅': actual_weight,
                            '排位檔位': draw_pos,
                            '獨贏賠率': odds_val,
                            '距離': distance,                
                            '場地': surface_type,            
                            'horse_surface_win_rate': 0.08,   
                            'horse_dist_win_rate': 0.08
                        })
                success_count += 1
            progress_bar.progress(race_no / 12)
            time.sleep(1.0)
        except Exception as e:
            continue
            
    progress_bar.empty()
    status_text.empty()
    
    if not all_races_data_tool:
        st.sidebar.error("❌ 該日期找不到任何賽事資料，請確認日期是否正確或當日是否有賽事。")
    else:
        df_tool = pd.DataFrame(all_races_data_tool)
        df_tool['獨贏賠率'] = pd.to_numeric(df_tool['獨贏賠率'], errors='coerce').fillna(0.0)
        
        output_cols = ['馬季', '賽事編號', '名次', '馬號', '馬名', '騎師', '練馬師', '實際負磅', '排位檔位', '獨贏賠率', '距離', '場地', 'horse_surface_win_rate', 'horse_dist_win_rate']
        df_tool = df_tool[[col for col in output_cols if col in df_tool.columns]]
        
        csv_filename = f"hkjc_data_15features_{date_for_id_tool}.csv"
        csv_data = df_tool.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        
        st.sidebar.success(f"🎉 成功抓取 {success_count} 場，共 {len(df_tool)} 筆資料！")
        st.sidebar.download_button(
            label="⬇️ 下載產出的訓練 CSV 檔",
            data=csv_data,
            file_name=csv_filename,
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
                    odds_value = str(node.get("oddsValue")).strip() # 轉成字串並去除空白
                    
                    # 排除 None, 空白, 以及 "SCR"
                    if horse_no and odds_value not in ("None", "", "SCR"):
                        try:
                            market[pool_type][horse_no] = float(odds_value)
                        except ValueError:
                            # 如果馬會未來回傳其他非數字字串 (例如 "REF" 退款)，直接略過避免崩潰
                            pass

        if not odds_by_race:
            return None, None, "賽事有回應，但目前尚無可用 WIN／PLACE 賠率。"
        return odds_by_race, (max(timestamps) if timestamps else None), None

    except requests.RequestException as exc:
        return None, None, f"連線錯誤：{exc}"
    except (ValueError, TypeError, KeyError) as exc:
        return None, None, f"解析回應錯誤：{exc}"

fetched_prediction_df = st.session_state.get('fetched_prediction_df')
if fetched_prediction_df is not None:
    df_raw = fetched_prediction_df.copy()
    fetched_date = st.session_state.get('fetched_prediction_date', race_date_choice)
    source_label = f"HKJC {fetched_date:%Y-%m-%d} 官方排位抓取"
    prediction_signature = st.session_state.get('fetched_prediction_signature', source_label)
    st.sidebar.success(f'預測來源：{source_label}（{len(df_raw)} 匹）')
elif PREDICTION_CSV_PATH.is_file():
    try:
        df_raw = pd.read_csv(PREDICTION_CSV_PATH, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df_raw = pd.read_csv(PREDICTION_CSV_PATH, encoding='cp950')
    source_label = f"GitHub repo：{PREDICTION_CSV_PATH.name}"
    prediction_signature = f"{PREDICTION_CSV_PATH.name}:{PREDICTION_CSV_PATH.stat().st_mtime_ns}:{PREDICTION_CSV_PATH.stat().st_size}"

    if df_raw.empty:
        st.error(f"預測 CSV 是空檔：{PREDICTION_CSV_PATH.name}")
        st.stop()
    st.sidebar.success(f"已從 repo 載入預測卡：{PREDICTION_CSV_PATH.name}（{len(df_raw)} 匹）")
    st.success("✅ 已從 GitHub repo 載入預測賽事資料！")
else:
    st.error(f"找不到 repo 預測 CSV：{PREDICTION_CSV_PATH.name}。")
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
        auto_odds_status = f'已自動更新即時賠率：WIN {win_count} 匹／PLACE {place_count} 匹'
    else:
        auto_odds_status = f'目前未能取得已開出的官方賠率。'

if auto_odds_status:
    if auto_odds_status.startswith('已自動'):
        st.sidebar.success(auto_odds_status)
    else:
        st.sidebar.warning(auto_odds_status)

# 確保賠率欄位存在
if '獨贏賠率' not in df_raw.columns:
    df_raw['獨贏賠率'] = 10.0
if '位置賠率' not in df_raw.columns:
    win_values = pd.to_numeric(df_raw['獨贏賠率'], errors='coerce').fillna(10.0)
    df_raw['位置賠率'] = 1.0 + (win_values - 1.0) / 3.2

if 'df_data' not in st.session_state or st.session_state.get('prediction_signature') != prediction_signature:
    st.session_state['df_data'] = df_raw.copy()
    st.session_state['prediction_signature'] = prediction_signature
    st.session_state['odds_editor_version'] = 0

# ==========================================
# 互動式臨場賠率輸入面板
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
feature_cols = FEATURE_COLS_15

for col in feature_cols:
    if col not in df.columns:
        df[col] = 0.0

X_predict = df[feature_cols].fillna(0)
df['pred_win_prob'] = model.predict_proba(X_predict)[:, 1]
df['ev'] = df['pred_win_prob'] * df['獨贏賠率']

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
                df_backtest = df_backtest.loc[has_official_result].copy()
            df_backtest.drop(columns=['_race_key', '_horse_key', '_result_rank', '_result_win_odds', '_result_place_odds', '_result_match'], errors='ignore', inplace=True)
            if backtest_ready:
                st.sidebar.success(f"{result_source_label}已配對：{matched} 匹")
            else:
                st.sidebar.warning("未能按賽事編號＋馬號配對到名次。")
    except Exception as exc:
        st.sidebar.error(f"讀取回測 CSV 失敗：{type(exc).__name__}: {exc}")

st.markdown("---")
tab1, tab2 = st.tabs(["🎯 各場次預測推薦", "📈 歷史回測 (獨贏/位置/位置Q)"])

# ---------------- 分頁 1: 賽前預測 ----------------
with tab1:
    st.subheader("⚡ 臨場賠率更新中心與實戰計時")
    
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
    api_date = col_d.text_input("API 查詢日期 (YYYY-MM-DD)", value=auto_date)

    race_series = df_raw['賽事編號'].map(extract_race_no)
    races_available = sorted(int(x) for x in race_series.dropna().unique())
    if races_available:
        target_race = col_r.selectbox("查看場次", races_available)
    else:
        target_race = col_r.number_input("查看場次", min_value=1, max_value=15, value=1, step=1)

    target_time = col_t.time_input("預定開跑時間", value=datetime.strptime("14:30", "%H:%M").time())
    
    try:
        race_date = datetime.strptime(api_date.strip(), "%Y-%m-%d").date()
    except ValueError:
        st.error("API 日期格式必須是 YYYY-MM-DD。")
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

    auto_col, interval_col, button_col = st.columns([1.5, 1.5, 2.5])
    auto_refresh = auto_col.checkbox("自動更新全部場次", value=False)
    refresh_seconds = interval_col.selectbox(
        "更新間隔（秒）", [10, 30, 60, 120, 300], index=1, disabled=not auto_refresh
    )
    manual_refresh = button_col.button("🔄 立即更新全部場次賠率", use_container_width=True)

    if auto_refresh:
        st_autorefresh(interval=int(refresh_seconds * 1000), key="hkjc_live_odds_autorefresh")

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
            st.session_state['odds_last_status'] = f"成功取得 {len(live_by_race)} 場賠率。"
            st.session_state['odds_server_time'] = server_updated_at
            st.session_state['odds_last_error'] = None
        else:
            st.session_state['odds_last_error'] = error

    if st.session_state.get('odds_last_error'):
        st.warning(st.session_state['odds_last_error'])
    elif st.session_state.get('odds_last_status'):
        st.success(st.session_state['odds_last_status'])

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
        st.warning("⚠️ 沒有符合當前篩選條件的馬匹。")
    else:
        st.dataframe(rec_df, use_container_width=True)

        st.markdown("---")
        st.subheader("🛒 虛擬模擬投注站 (Paper Trading)")
        st.caption("在這裡模擬實戰買入，系統會鎖定當下的即時賠率。")

        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns([1, 2, 1.5, 1.5, 1.5])
        
        sim_race = col_s1.selectbox("場次", races_available, key="sim_race")
        
        horses_in_race = df[df['賽事編號'].map(extract_race_no) == sim_race].copy()
        horses_in_race['display_name'] = horses_in_race['馬號'].astype(str) + " (" + horses_in_race['馬名'] + ")"
        sim_horse = col_s2.selectbox("選擇馬匹", horses_in_race['display_name'].tolist(), key="sim_horse")
        
        sim_type = col_s3.selectbox("玩法", ["獨贏 (WIN)", "位置 (PLACE)"], key="sim_type")
        sim_amount = col_s4.number_input("注碼", min_value=10, value=100, step=10, key="sim_amount")
        
        if col_s5.button("➕ 確定下注", use_container_width=True, type="primary"):
            horse_no_raw = sim_horse.split(" ")[0]
            horse_name_raw = sim_horse.split("(")[1].replace(")", "")
            
            target_row = horses_in_race[horses_in_race['馬號'].astype(str) == horse_no_raw]
            if "WIN" in sim_type:
                current_odds = target_row['獨贏賠率'].values[0]
                bet_type = "WIN"
            else:
                current_odds = target_row['位置賠率'].values[0]
                bet_type = "PLA"
                
            new_bet = pd.DataFrame([{
                '下注時間': datetime.now(hk_timezone).strftime("%H:%M:%S"),
                '賽事編號': target_row['賽事編號'].values[0],
                '場次': sim_race,
                '馬號': horse_no_raw,
                '馬名': horse_name_raw,
                '玩法': bet_type,
                '買入賠率': current_odds,
                '注碼': sim_amount
            }])
            
            st.session_state['simulated_bets'] = pd.concat([st.session_state['simulated_bets'], new_bet], ignore_index=True)
            st.success(f"✅ 成功記錄：第 {sim_race} 場 {horse_no_raw} 號 ({bet_type}) | 注碼 ${sim_amount} @ 賠率 {current_odds}")

        if not st.session_state['simulated_bets'].empty:
            st.markdown("##### 🧾 目前模擬注單紀錄")
            st.dataframe(st.session_state['simulated_bets'], use_container_width=True, hide_index=True)
            
            dl_col, clear_col = st.columns([2, 2])
            csv_sim = st.session_state['simulated_bets'].to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            dl_col.download_button(
                label="⬇️ 下載模擬注單 (CSV)",
                data=csv_sim,
                file_name=f"simulated_bets_{api_date}.csv",
                mime="text/csv",
                use_container_width=True
            )
            if clear_col.button("🗑️ 清除所有模擬紀錄", use_container_width=True):
                st.session_state['simulated_bets'] = pd.DataFrame(columns=[
                    '下注時間', '賽事編號', '場次', '馬號', '馬名', '玩法', '買入賠率', '注碼'
                ])
                st.rerun()

# ---------------- 分頁 2: 賽後回測 ----------------
with tab2:
    st.subheader("📊 多彩種策略回測總覽")
    if not backtest_ready:
        st.warning("請在左側上傳已完賽的結果 CSV。需有『賽事編號』、『馬號』及『名次』。")
    else:
        sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
            "🥇 獨贏 (Win)", "🥈 單匹位置 (Place)", "🔗 位置Q (QP)", "🥉 3匹位置包抄", "🛒 模擬結算 (Paper Trade)"
        ])
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
                st.info("💡 目前設定下沒有符合位置 Q 出手的場次。")

        with sub_tab4:
            st.markdown("#### 🥉 3匹位置包抄 (3x Place) 策略回測")
            st.caption("策略邏輯：每場買入 AI 推薦最高分的 3 匹馬位置（每匹各一注）。")
            p3_invested, p3_return, p3_bets, p3_hits = 0, 0, 0, 0
            p3_records = []
            
            for race_id, group in df_backtest.groupby('賽事編號'):
                filtered_group = group[(group['ev'] >= min_ev) & (group['獨贏賠率'] >= min_odds) & (group['獨贏賠率'] <= max_odds)]
                sorted_group = filtered_group.sort_values(by='ev', ascending=False).reset_index(drop=True)
                
                picks = sorted_group.head(3)
                
                for _, pick in picks.iterrows():
                    p3_bets += 1
                    p3_invested += BET_AMOUNT
                    is_hit = (pick['numeric_rank'] <= 3)
                    
                    settlement_place_odds = pick.get('回測位置賠率', np.nan)
                    if pd.isna(settlement_place_odds) or settlement_place_odds <= 1:
                        settlement_place_odds = pick['位置賠率']
                    p_odds = float(settlement_place_odds) if pd.notna(settlement_place_odds) and float(settlement_place_odds) > 1.0 else 1.5
                    
                    if is_hit:
                        p3_hits += 1
                        payout = BET_AMOUNT * p_odds
                        p3_return += payout
                        result_str = "✅ 命中位置"
                    else:
                        payout = 0
                        result_str = "❌ 未入前三"
                        
                    p3_records.append({
                        '賽事編號': race_id,
                        '投注馬號': f"{pick['馬號']} ({pick['馬名']})",
                        '實際名次': str(pick['名次']).replace('.0', ''),
                        '位置賠率': round(p_odds, 2),
                        'EV': round(pick['ev'], 2),
                        '結果': result_str,
                        '派彩': f"${payout:.1f}",
                        '淨盈虧': f"${payout - BET_AMOUNT:.1f}"
                    })
                    
            if p3_bets > 0:
                roi = ((p3_return - p3_invested) / p3_invested) * 100
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("總投注注數", f"{p3_bets} 注")
                col2.metric("命中注數", f"{p3_hits} 注", f"命中率: {p3_hits/p3_bets*100:.1f}%")
                col3.metric("總成本", f"${p3_invested}")
                col4.metric("總回收", f"${p3_return:.1f}", f"ROI: {roi:.2f}%")
                st.markdown("##### 📝 3匹位置包抄明細")
                st.dataframe(pd.DataFrame(p3_records), use_container_width=True)
            else:
                st.info("💡 目前設定下沒有符合出手的場次。")

        with sub_tab5:
            st.markdown("#### 🛒 模擬投注真實結算 (Paper Trading PnL)")
            st.caption("系統會將你的虛擬注單與官方賽果比對，並使用你【鎖定當下】的賠率計算真實盈虧。")
            
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
                            '買入賠率': locked_odds,
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
                        elif bet_type == "PLA" and actual_rank <= 3:
                            is_hit = True

                        if is_hit:
                            sim_hits += 1
                            payout = stake * locked_odds
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
                            '買入賠率': locked_odds,
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
                    st.dataframe(pd.DataFrame(sim_results), use_container_width=True)
