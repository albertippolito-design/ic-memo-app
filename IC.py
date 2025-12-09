import os
import streamlit as st
from docx import Document
from PyPDF2 import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv

# ---------------- FORCE LOAD .env FROM PROJECT FOLDER ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=DOTENV_PATH, override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error(
        f"GEMINI_API_KEY not found.\n\n"
        f"Checked path:\n{DOTENV_PATH}\n\n"
        f"Make sure a file named .env exists in that folder and contains:\n\n"
        f"GEMINI_API_KEY=your_key_here"
    )
    st.stop()

# Configure Gemini once
genai.configure(api_key=GEMINI_API_KEY)

# ---------------- CONFIG ----------------

IC_SECTIONS = [
    "Executive Summary",
    "Transaction Overview",
    "Asset Description",
    "Market & Competition",
    "Business Plan",
    "Financials",
    "Risks & Mitigants",
    "Recommendation"
]

# ---------------- FILE PARSING ----------------

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

def extract_text(file, filetype):
    if filetype == "docx":
        return extract_text_from_docx(file)
    elif filetype == "pdf":
        return extract_text_from_pdf(file)
    elif filetype == "txt":
        return extract_text_from_txt(file)
    return ""

def detect_filetype(filename: str):
    ext = filename.split(".")[-1].lower()
    if ext in ["docx", "pdf", "txt"]:
        return ext
    return None

# ---------------- GEMINI MEMO GENERATOR ----------------

def generate_ic_memo(guide_text: str, deal_text: str, sections):
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Safety truncation
    guide_snippet = guide_text[:8000]
    deal_snippet = deal_text[:12000]

    sections_block = "\n".join(f"- {s}" for s in sections)

    prompt = f"""
You are a professional business documentation assistant for real estate and private equity.

You are given:
1) GUIDE MEMO (style reference only)
2) DEAL MATERIALS (source of facts)

Required structure:
{sections_block}

Rules:
- Use the GUIDE MEMO only for tone, style, and structure.
- Use the DEAL MATERIALS only for facts.
- Do NOT invent numbers or specific facts. If something is missing, write "No information available."
- Output a full investment-committee-style memo in plain text.
- Use clear section titles exactly matching the required structure.

--- GUIDE MEMO (STYLE REFERENCE) ---
{guide_snippet}

--- DEAL MATERIALS (FACTUAL INPUT) ---
{deal_snippet}
"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip() if hasattr(response, "text") else str(response)
        return text
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {e}")

# ---------------- TXT WRITER ----------------

def memo_to_txt(memo_text: str, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(memo_text)

# ---------------- STREAMLIT UI ----------------

st.title("Investment Committee Memo Generator (Gemini → TXT)")
st.write("Upload a guide IC memo and deal materials. The app generates a structured IC memo as a .txt file.")

if "ic_count" not in st.session_state:
    st.session_state.ic_count = 1

# Step 1: Guide memo
guide_file = st.file_uploader(
    "Upload Guide IC Memo (.docx, .pdf, .txt)",
    type=["docx", "pdf", "txt"],
    key="guide_uploader"
)

guide_text = ""
if guide_file is not None:
    filetype = detect_filetype(guide_file.name)
    guide_text = extract_text(guide_file, filetype)
    st.success("Guide memo loaded.")
    st.text_area("Guide Preview", guide_text, height=200)

# Step 2: Deal materials
deal_files = st.file_uploader(
    "Upload Deal Materials (.docx, .pdf, .txt) – multiple allowed",
    type=["docx", "pdf", "txt"],
    accept_multiple_files=True,
    key="deal_uploader"
)

deal_text = ""
if deal_files:
    for f in deal_files:
        filetype = detect_filetype(f.name)
        deal_text += extract_text(f, filetype) + "\n"
    st.success(f"{len(deal_files)} deal document(s) loaded.")
    st.text_area("Deal Preview", deal_text, height=200)

# Step 3: Generate
if st.button("Generate IC Memo", key="generate_button"):
    if not guide_text or not deal_text:
        st.error("Upload both a guide memo and deal materials before generating.")
    else:
        try:
            with st.spinner("Generating memo with Gemini..."):
                memo_text = generate_ic_memo(guide_text, deal_text, IC_SECTIONS)

            os.makedirs("output", exist_ok=True)
            filename = f"IC{st.session_state.ic_count}.txt"
            filepath = os.path.join("output", filename)

            memo_to_txt(memo_text, filepath)
            st.session_state.ic_count += 1

            with open(filepath, "rb") as f:
                st.download_button("Download Memo", f.read(), file_name=filename)

            st.success(f"Memo generated: {filename}")

        except Exception as e:
            st.error(str(e))
