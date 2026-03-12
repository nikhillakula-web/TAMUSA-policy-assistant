import base64
import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        /* Background image */
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Professional deep navy/slate overlay — clean and formal */
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: linear-gradient(
                160deg,
                rgba(10, 25, 60, 0.82) 0%,
                rgba(5, 15, 40, 0.88) 100%
            );
            z-index: 0;
            pointer-events: none;
        }}

        /* Content above overlay */
        .stApp > * {{
            position: relative;
            z-index: 1;
        }}

        /* Transparent backgrounds */
        .main, .block-container {{
            background-color: transparent !important;
        }}

        h1, h2, h3, h4, h5, h6, p, div, label, span, li {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


st.set_page_config(
    page_title="TAMUSA University Policy Assistant",
    page_icon="🎓",
    layout="wide",
)

set_background("assets/college.jpg")


st.markdown("""
<style>
/* ── Typography ── */
.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-top: 40px;
    text-shadow: 0 2px 20px rgba(0,0,0,0.6);
    color: white !important;
    letter-spacing: -0.5px;
}
.subtitle {
    text-align: center;
    color: rgba(255,255,255,0.55) !important;
    margin-bottom: 40px;
    font-size: 16px;
    letter-spacing: 0.3px;
}

/* ── Answer container (navy glass card) ── */
.answer-header {
    background: linear-gradient(135deg, rgba(30, 64, 130, 0.75), rgba(15, 35, 80, 0.80));
    backdrop-filter: blur(14px);
    border: 1px solid rgba(100, 160, 255, 0.30);
    border-left: 4px solid #4a90e2;
    padding: 22px 28px 10px 28px;
    border-radius: 12px 12px 0 0;
    margin-top: 24px;
}
.answer-body {
    background: rgba(15, 35, 80, 0.65);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(100, 160, 255, 0.20);
    border-top: none;
    padding: 20px 28px 24px 28px;
    border-radius: 0 0 12px 12px;
    margin-bottom: 24px;
    line-height: 1.75;
    font-size: 15px;
}

/* ── Source docs box ── */
.docs-box {
    background-color: rgba(10, 20, 50, 0.60);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(100, 160, 255, 0.18);
    border-left: 3px solid rgba(100, 160, 255, 0.5);
    padding: 16px 20px;
    border-radius: 8px;
    margin-bottom: 10px;
    font-size: 13.5px;
    line-height: 1.65;
    color: rgba(255,255,255,0.78) !important;
}

/* ── Input field ── */
.stTextInput > div > div > input {
    background-color: rgba(10, 25, 60, 0.75) !important;
    color: white !important;
    border: 1px solid rgba(100, 160, 255, 0.45) !important;
    border-radius: 8px !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(100, 160, 255, 0.85) !important;
    box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.25) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background-color: rgba(10, 25, 60, 0.60) !important;
    border: 1px solid rgba(100, 160, 255, 0.25) !important;
    border-radius: 8px !important;
    color: rgba(255,255,255,0.80) !important;
    font-size: 14px !important;
}
.streamlit-expanderContent {
    background-color: rgba(5, 15, 40, 0.55) !important;
    border: 1px solid rgba(100, 160, 255, 0.15) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ── Spinner ── */
.stSpinner > div {
    color: rgba(255,255,255,0.7) !important;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


st.markdown("<div class='title'>🎓 TAMUSA University Policy Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ask questions based on official university policies</div>", unsafe_allow_html=True)


@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        "policy_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()


query = st.text_input("Enter your question", placeholder="e.g. What is the grade appeal process?")

if query:
    with st.spinner("Searching official policies..."):

        # 1. Retrieve docs
        docs = vectorstore.similarity_search(query, k=6)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 2. Build prompt
        prompt = f"""
You are an AI University Policy Assistant.

Answer the user question using ONLY the provided policy context below.

If the question is broad, summarize the relevant rules from the context.

If partial information exists, explain what is available.

Only say "I could not find this information in official university policies."
if the context is completely unrelated to the question.

---------------------------------
POLICY CONTEXT:
{context}
---------------------------------

USER QUESTION:
{query}

Provide a clear and structured answer.
"""

        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        answer = response.choices[0].message.content

  
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(30,64,130,0.75), rgba(15,35,80,0.85));
            backdrop-filter: blur(14px);
            border: 1px solid rgba(100,160,255,0.30);
            border-left: 4px solid #4a90e2;
            border-radius: 12px;
            padding: 24px 30px;
            margin-top: 24px;
            margin-bottom: 24px;
        ">
            <div style="font-size:12px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase;
                        color:rgba(120,175,255,0.90) !important; margin-bottom:16px;">
                🏛️ &nbsp;Policy Guidance
            </div>
            <div style="font-size:15px; line-height:1.85; color:rgba(255,255,255,0.92) !important;">
                {answer.replace(chr(10), '<br>')}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

   
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📄 View source excerpts from official policy documents"):
        for i, d in enumerate(docs, 1):
            st.markdown(
                f"<div class='docs-box'><strong>Source {i}</strong><br><br>{d.page_content[:500]}</div>",
                unsafe_allow_html=True
            )
