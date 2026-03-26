import base64
import streamlit as st
import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

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

        .stApp > * {{
            position: relative;
            z-index: 1;
        }}

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
.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-top: 40px;
    text-shadow: 0 2px 20px rgba(0,0,0,0.6);
}
.subtitle {
    text-align: center;
    color: rgba(255,255,255,0.55);
    margin-bottom: 40px;
}

.docs-box {
    background-color: rgba(10, 20, 50, 0.60);
    border-left: 3px solid rgba(100, 160, 255, 0.5);
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 10px;
    font-size: 13px;
}

.stTextInput input {
    background-color: rgba(10, 25, 60, 0.75) !important;
    color: white !important;
}
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

# ==========================
# INPUT
# ==========================
query = st.text_input("Enter your question", placeholder="e.g. What is the grade appeal process?")

if query:
    with st.spinner("Searching official policies..."):

        docs = vectorstore.similarity_search(query, k=6)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are an AI University Policy Assistant.

Answer ONLY using the given context.
If not found, say:
"I could not find this information in official university policies."

CONTEXT:
{context}

QUESTION:
{query}
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        answer = response.choices[0].message.content

        # ✅ SHOW ANSWER FIRST
        st.markdown("### 🏛️ Policy Guidance")
        st.write(answer)

        # ✅ SOURCES BELOW
        with st.expander("📄 View source excerpts"):
            for i, d in enumerate(docs, 1):
                st.markdown(
                    f"<div class='docs-box'><b>Source {i}</b><br>{d.page_content[:500]}</div>",
                    unsafe_allow_html=True
                )