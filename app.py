import streamlit as st
import json
import jieba
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from udicOpenData.stopwords import rmsw
from rank_bm25 import BM25Okapi
import os
from datetime import datetime
import pandas as pd

# --- UI 標題 ---
st.title("IR 作業範例：社群文字風向球")

st.sidebar.header("Params setting")

# BM25 參數
top_n = st.sidebar.slider("顯示前幾個關鍵詞", 50, 500, 200)
k1 = st.sidebar.slider("BM25 k1 (TF 重要度)", 0.1, 3.0, 1.5, 0.1)
b = st.sidebar.slider("BM25 b (長度調整)", 0.0, 1.0, 0.3, 0.1)

# --- 資料讀取 ---
DEFAULT_PATH = "./data/高虹安.json"
st.sidebar.write("使用資料：", DEFAULT_PATH)

if not os.path.exists(DEFAULT_PATH):
    st.error(f"找不到檔案：{DEFAULT_PATH}")
    st.stop()

with open(DEFAULT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [
    item["_source"]["content"]
    for item in data["hits"]
    if item.get("_source", {}).get("content", "").strip()
]

# --- 自訂詞典區 ---
st.header("自訂詞典與過濾字")

# 初始化 session state
if "custom_dict" not in st.session_state:
    st.session_state.custom_dict = pd.DataFrame([

        # 政黨
        {"word": "民眾黨", "weight": 10},
        {"word": "民進黨", "weight": 10},
        {"word": "國民黨", "weight": 10},
        {"word": "時代力量", "weight": 10},

        # 人名
        {"word": "高虹安", "weight": 20},
        {"word": "柯文哲", "weight": 10},
        {"word": "林智堅", "weight": 10},
        {"word": "沈慧虹", "weight": 10},
        {"word": "王郁文", "weight": 10},
        {"word": "黃惠玟", "weight": 10},
        {"word": "蔡麗清", "weight": 10},
        {"word": "邱臣遠", "weight": 10},
        {"word": "張治祥", "weight": 10},
        {"word": "林耕仁", "weight": 10},
        {"word": "柯建銘", "weight": 10},
        {"word": "李忠庭", "weight": 10},
        {"word": "陳奐宇", "weight": 10},
        {"word": "林佳龍", "weight": 10},
        {"word": "賴清德", "weight": 10},
        {"word": "蔡英文", "weight": 10},

        # 地名/機構
        {"word": "新竹市", "weight": 8},
        {"word": "資策會", "weight": 8},
        {"word": "交大", "weight": 8},
        {"word": "清大", "weight": 8},
        {"word": "竹科", "weight": 8},
        {"word": "台積電", "weight": 8},
        {"word": "聯發科", "weight": 8},

        # 議題/其他
        {"word": "投票", "weight": 10},
        {"word": "助理費", "weight": 10},
        {"word": "論文門", "weight": 10},
        {"word": "塔綠班", "weight": 10},
        {"word": "公投", "weight": 10},
        {"word": "選舉", "weight": 10},
        {"word": "罷免", "weight": 10},
        {"word": "立委", "weight": 10},
        {"word": "市長", "weight": 10},
        {"word": "議員", "weight": 10},
        {"word": "助理", "weight": 10},
       
    ])

if "ignore_dict" not in st.session_state:
    st.session_state.ignore_dict = pd.DataFrame(
        [{"word": w} for w in ["的", "了", "呢", "嗎", "得", "地", "著", "我", "他", "她", "你", "我們", "你們", "這個", "那個", "是", "在", "有", "沒有", "也", "不", "不是", "是不是", "已經", "很", "都", "與", "和", "及", "就","而", "但", "或", "要", "為", "說", "Sent", "from", "my", "on", "news", "gur"]]
    )

# 編輯表格（可新增、刪除、修改）
st.subheader("自訂關鍵詞與權重")
st.info("可直接在下表編輯或刪除行。")
custom_words_df = st.data_editor(
    st.session_state.custom_dict,
    num_rows="dynamic",
    use_container_width=True,
    key="custom_editor"
)

st.subheader("忽略詞清單")
ignore_words_df = st.data_editor(
    st.session_state.ignore_dict,
    num_rows="dynamic",
    use_container_width=True,
    key="ignore_editor"
)

# --- 更新 session 狀態 ---
st.session_state.custom_dict = custom_words_df
st.session_state.ignore_dict = ignore_words_df

# --- 處理流程 ---
if st.button("生成"):
    with st.spinner("waitting..."):
        # 加入自訂詞
        for _, row in st.session_state.custom_dict.iterrows():
            jieba.add_word(row["word"], freq=row["weight"])

        ignore_list = st.session_state.ignore_dict["word"].tolist()

       # 斷詞
        words_list = []
        for text in texts:
            raw = rmsw(text)
            clean_text = ''.join(raw) if not isinstance(raw, str) else raw
            tokens = [w for w in jieba.cut(clean_text) if w not in ignore_list and w.strip()]
            words_list.append(tokens)


        # BM25
        bm25 = BM25Okapi(words_list, k1=k1, b=b)
        all_words = [w for words in words_list for w in words]
        unique_words = set(all_words)
        word_scores = {w: bm25.get_scores([w]).mean() for w in unique_words}

        freq = dict(sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_n])

        # --- 文字雲 ---
        cloud = WordCloud(
            font_path="./src/Handwriting.ttf",
            background_color="white",
            width=1000,
            height=600
        ).generate_from_frequencies(freq)

        # 儲存輸出
        os.makedirs("./outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"./outputs/wordcloud_{timestamp}.png"
        cloud.to_file(output_path)

        # 顯示結果
        st.image(cloud.to_array(), caption=f"sucess：{output_path}")

        # 關鍵詞表格
        st.subheader("Ranked Keywords")
        freq_df = pd.DataFrame(list(freq.items()), columns=["關鍵詞", "BM25 分數"])
        st.dataframe(freq_df, use_container_width=True)

        st.success("result saved in outputs folder")
