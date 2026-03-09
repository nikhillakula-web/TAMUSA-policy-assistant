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
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .main {{
            background-color: rgba(0, 0, 0, 1);
            padding: 20px;
            border-radius: 10px;
        }}

        h1, h2, h3, h4, h5, h6, p, div {{
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
.main {
    background-color: #0f172a;
    color: white;
}
.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-top: 40px;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 40px;
}
.answer-box {
    background-color: #1e293b;
    padding: 25px;
    border-radius: 12px;
    margin-top: 20px;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'> TAMUSA University Policy Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ask questions based on official university policies</div>", unsafe_allow_html=True)

# ==========================
# LOAD VECTOR STORE
# ==========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "policy_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# ==========================
# QUESTION INPUT
# ==========================
query = st.text_input("Enter your question")

if query:
    with st.spinner("Searching official policies..."):
        docs = vectorstore.similarity_search(query, k=6)

        st.write("Information retrieved from official university policy documents:")
        for d in docs:
            st.write(d.page_content[:500])

        context = "\n\n".join([doc.page_content for doc in docs])

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

        st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
        st.markdown("###  Answer")
        st.write(answer)
        st.markdown("</div>", unsafe_allow_html=True)
