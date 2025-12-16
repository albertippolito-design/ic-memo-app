import streamlit as st
from pathlib import Path
from datetime import datetime

STORAGE = Path("user.txt")

st.set_page_config(page_title="Text Saver", page_icon="✉️", layout="centered")

# Styling
st.markdown(
    """
    <style>
    .stApp { font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; }
    .title { font-size:28px; font-weight:600; }
    .muted { color: #6c757d; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='title'>Text Saver</div>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Type something below and Save — entries are appended to `user.txt` with a timestamp.</div>", unsafe_allow_html=True)
st.write("---")

with st.form("input_form", clear_on_submit=False):
    text = st.text_area("Your text", placeholder="Write something meaningful (or not)...", height=160)
    cols = st.columns([1, 1, 1])
    submit = cols[0].form_submit_button("Save")
    download_now = cols[1].form_submit_button("Save & Download")
    clear_file = cols[2].form_submit_button("Clear File")

if submit or download_now:
    if not text or not text.strip():
        st.warning("Please enter some text before saving.")
    else:
        try:
            STORAGE.parent.mkdir(parents=True, exist_ok=True)
            entry = f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}] {text.strip()}\n"
            with open(STORAGE, "a", encoding="utf-8") as f:
                f.write(entry)
            st.success("Saved to user.txt")
            if download_now:
                st.download_button("Download just-saved entry", data=entry.encode("utf-8"), file_name="entry.txt")
        except Exception as e:
            st.error(f"Failed to save: {e}")

if clear_file:
    try:
        if STORAGE.exists():
            STORAGE.unlink()
        STORAGE.write_text("", encoding="utf-8")
        st.info("Cleared user.txt")
    except Exception as e:
        st.error(f"Failed to clear file: {e}")

st.write("### Current contents")
if STORAGE.exists() and STORAGE.stat().st_size > 0:
    try:
        text_data = STORAGE.read_text(encoding="utf-8")
        lines = [l for l in text_data.splitlines() if l.strip()]
        last = lines[-20:]
        st.code("\n".join(last))
        st.download_button("Download user.txt", data=text_data.encode("utf-8"), file_name="user.txt")
    except Exception as e:
        st.error(f"Unable to read file: {e}")
else:
    st.write("_(no entries yet)_")

st.write("---")
st.markdown("<small class='muted'>Entries include an ISO timestamp. File stored next to this script as `user.txt`.</small>", unsafe_allow_html=True)
