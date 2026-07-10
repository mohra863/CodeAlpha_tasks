import streamlit as st
import json
import base64
from sentence_transformers import SentenceTransformer, util

st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="👀",
    layout="centered"
)

st.title("AI & Machine Learning FAQ Chatbot")

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = get_base64("pic.png")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_data():
    with open("faqs.json", "r", encoding="utf-8") as f:
        faqs = json.load(f)
    questions = [item["question"] for item in faqs]
    answers = [item["answer"] for item in faqs]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    question_embeddings = model.encode(questions, convert_to_tensor=True)
    return questions, answers, model, question_embeddings

questions, answers, model, question_embeddings = load_data()

def get_best_answer(user_question, threshold=0.4):
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, question_embeddings)[0]
    best_index = similarities.argmax().item()
    best_score = similarities[best_index].item()
    if best_score < threshold:
        return "Sorry, I don't have an answer to that question."
    return answers[best_index]

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask your question about AI or Machine Learning...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    answer = get_best_answer(user_input)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)