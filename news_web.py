import streamlit as st
from news import summarize_stock_news
from pathlib import Path

OUTPUT = Path("news.txt")

st.set_page_config(page_title="Stock News", page_icon="📰", layout="centered")
st.title("Stock News — Yahoo Finance (summary)")
st.write("Enter a ticker symbol (e.g. AAPL) and click Fetch. The app will save a short summary to `news.txt`.")

symbol = st.text_input("Ticker symbol", value="AAPL")
if st.button("Fetch"):
    if not symbol.strip():
        st.warning("Please enter a ticker symbol.")
    else:
        with st.spinner(f"Fetching news for {symbol}..."):
            summary = summarize_stock_news(symbol)
            try:
                OUTPUT.write_text(summary, encoding="utf-8")
            except Exception as e:
                st.error(f"Failed to save file: {e}")
        st.success("Saved summary to news.txt")
        st.code(summary)
        st.download_button("Download news.txt", data=summary.encode("utf-8"), file_name="news.txt")

st.write("---")
if OUTPUT.exists():
    if st.checkbox("Show existing news.txt contents"):
        st.code(OUTPUT.read_text(encoding="utf-8"))
