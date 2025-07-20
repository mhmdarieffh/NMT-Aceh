import torch
import torch.nn as nn
import math
import json
import re
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

from nmt_model_loader import (
    load_model_and_vocab,
    smart_translate
)

# Load kedua arah
model_indo2aceh, src_i2a, tgt_i2a, inv_tgt_i2a = load_model_and_vocab(
    "nmtindo.pt",   
    "src_vocabindo2aceh.json",
    "tgt_vocabindo2aceh.json"
)

model_aceh2indo, src_a2i, tgt_a2i, inv_tgt_a2i = load_model_and_vocab(
    "nmtaceh.pt",
    "src_vocabaceh2indo.json",
    "tgt_vocabaceh2indo.json"
)
# Judul dan Tab
st.set_page_config(layout="wide")
st.title("Machine Translation & Analisis Teks")

tabs = st.tabs(["🌐 Penerjemahan", "📊 Analisis Teks"])

# ============================ #
#          Tab 1: MT           #
# ============================ #
with tabs[0]:
    st.markdown("<h3 style='text-align: center;'>Machine Translation</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Aceh ↔ Indonesia</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        from_lang = st.selectbox("Dari", ["Bahasa Aceh", "Bahasa Indonesia"])

    to_lang = "Bahasa Aceh" if from_lang == "Bahasa Indonesia" else "Bahasa Indonesia"

    with col2:
        st.markdown(f"<div style='margin-top: 35px; font-weight: bold;'>Ke: {to_lang}</div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        text_input = st.text_area("Ketik teks", placeholder="Tulis teks di sini...", height=200)

    with col4:
        if "translation" not in st.session_state:
            st.session_state["translation"] = ""
        st.text_area("Hasil terjemahan", value=st.session_state["translation"], height=200, disabled=True)

    if st.button("Terjemahkan"):
        if not text_input.strip():
            st.warning("Teks tidak boleh kosong.")
        else:
            if from_lang == "Bahasa Indonesia":
                result = smart_translate(text_input, model_indo2aceh, src_i2a, tgt_i2a, inv_tgt_i2a)
            elif from_lang == "Bahasa Aceh":
                result = smart_translate(text_input, model_aceh2indo, src_a2i, tgt_a2i, inv_tgt_a2i)
            else:
                result = "Pilihan bahasa tidak valid."
            st.session_state["translation"] = result
            st.experimental_rerun()

# ============================ #
#      Tab 2: Analisis Teks    #
# ============================ #
with tabs[1]:
    st.subheader("📊 Analisis Word Cloud & Histogram Frekuensi")

    # Load data dari kode
    df = pd.read_csv("databiasa.csv")
    text_aceh = " ".join(df["Aceh"].astype(str).tolist())
    text_indo = " ".join(df["Indonesia"].astype(str).tolist())

    def get_word_counts(text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        return Counter(tokens)

    word_counts_aceh = get_word_counts(text_aceh)
    word_counts_indo = get_word_counts(text_indo)

    lang_option = st.radio("Pilih Bahasa", ["Bahasa Aceh", "Bahasa Indonesia"], horizontal=True)

    # Pilih data berdasarkan radio button
    if lang_option == "Bahasa Aceh":
        word_counts = word_counts_aceh
        judul_wc = "Word Cloud - Bahasa Aceh"
        judul_hist = "Histogram Frekuensi Kata - Bahasa Aceh"
    else:
        word_counts = word_counts_indo
        judul_wc = "Word Cloud - Bahasa Indonesia"
        judul_hist = "Histogram Frekuensi Kata - Bahasa Indonesia"

    # Word Cloud
    wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis')
    wc.generate_from_frequencies(word_counts)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis('off')
    ax_wc.set_title(judul_wc, fontsize=16)
    st.pyplot(fig_wc)

    # Histogram
    top_n = st.slider("Jumlah kata yang ditampilkan di histogram", min_value=10, max_value=50, value=30)
    most_common = word_counts.most_common(top_n)
    words, counts = zip(*most_common)
    fig_hist, ax_hist = plt.subplots(figsize=(12, 6))
    bars = ax_hist.bar(words, counts, color='skyblue')
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax_hist.text(bar.get_x() + bar.get_width()/2, height + 0.5, str(count),
                     ha='center', va='bottom', fontsize=9)
    ax_hist.set_xlabel("Kata")
    ax_hist.set_ylabel("Frekuensi")
    ax_hist.set_title(judul_hist)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_hist)
