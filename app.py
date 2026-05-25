import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
from tinydb import TinyDB, Query
import uuid
from datetime import datetime
from streamlit_pdf_viewer import pdf_viewer
import os
import io
import html

st.markdown("""
<style>

/* ===== GLOBAL ===== */

.stApp {

    background:
    radial-gradient(circle at top left, #312e81 0%, transparent 30%),
    radial-gradient(circle at bottom right, #7c3aed 0%, transparent 30%),
    linear-gradient(135deg, #0f172a, #020617);

    color: white;
}

/* ===== MAIN ===== */

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
}

/* ===== TITLE ===== */

h1 {

    font-size: 46px !important;

    font-weight: 800 !important;

    letter-spacing: -2px;

    background: linear-gradient(
        90deg,
        #c084fc,
        #f472b6
    );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

/* ===== SUBTEXT ===== */

.stCaption {

    color: #cbd5e1 !important;

    font-size: 15px;
}

/* ===== SIDEBAR ===== */

section[data-testid="stSidebar"] {

    background: rgba(15, 23, 42, 0.72);

    backdrop-filter: blur(20px);

    border-right: 1px solid rgba(255,255,255,0.05);
}

/* ===== CHAT ===== */

.user-message {

    background: linear-gradient(
        135deg,
        #7c3aed,
        #ec4899
    );

    padding: 18px;

    border-radius: 20px;

    margin-top: 18px;

    margin-bottom: 10px;

    box-shadow:
        0 10px 30px rgba(124,58,237,0.35);
}

.bot-message {

    background: rgba(30,41,59,0.5);

    backdrop-filter: blur(18px);

    border: 1px solid rgba(255,255,255,0.06);

    padding: 18px;

    border-radius: 20px;

    margin-bottom: 24px;

    box-shadow:
        0 8px 30px rgba(0,0,0,0.25);
}

/* ===== INPUT ===== */

.stTextInput input {

    background: rgba(30,41,59,0.55) !important;

    border: 1px solid rgba(192,132,252,0.28) !important;

    border-radius: 16px !important;

    padding: 15px !important;

    color: white !important;

    font-size: 16px !important;

    box-shadow:
        0 0 0 rgba(0,0,0,0);
}

/* ===== BUTTON ===== */

.stButton button {

    width: 100%;

    border: none;

    border-radius: 16px;

    background: linear-gradient(
        135deg,
        #7c3aed,
        #ec4899
    );

    color: white;

    font-weight: 700;

    padding: 12px;

    transition: 0.3s ease;
}

.stButton button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 12px 30px rgba(236,72,153,0.35);
}

/* ===== FILE UPLOADER ===== */

[data-testid="stFileUploader"] {

    background: rgba(30,41,59,0.45);

    border: 1px dashed rgba(192,132,252,0.3);

    border-radius: 20px;

    padding: 25px;
}

/* ===== EXPANDERS ===== */

details {

    background: rgba(30,41,59,0.45);

    border-radius: 16px;

    padding: 10px;

    margin-bottom: 14px;

    border: 1px solid rgba(255,255,255,0.04);
}

/* ===== SCROLLBAR ===== */

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {

    background: linear-gradient(
        #7c3aed,
        #ec4899
    );

    border-radius: 20px;
}

/* ===== HIDE STREAMLIT ===== */

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

</style>
""", unsafe_allow_html=True)

st.title("NeuraDocs")
st.caption(
    "Next-generation AI workspace for intelligent document conversations."
)

st.markdown("""
<div style="
padding:22px;
border-radius:22px;
background:rgba(255,255,255,0.04);
backdrop-filter:blur(20px);
border:1px solid rgba(255,255,255,0.05);
margin-bottom:25px;
">

<h3 style="margin-top:0;">
Intelligent PDF Conversations
</h3>

<p style="opacity:0.8;">
Upload documents, retrieve semantic answers,
manage persistent AI chats, and interact
with your knowledge base in real time.
</p>

</div>
""", unsafe_allow_html=True)

db = TinyDB("chat_db.json")
Chat = Query()

# Ensure upload folder exists
os.makedirs("uploaded_pdfs", exist_ok=True)


def create_chat() -> str:
    chat_id = str(uuid.uuid4())
    db.insert({
        "chat_id": chat_id,
        "name": "New Chat",
        "created_at": datetime.now().strftime("%b %d, %I:%M %p"),
        "messages": [],
        "pdf_text": ""
    })
    return chat_id


def get_chat_display_name(chat_item: dict) -> str:
    return chat_item.get("name", "Untitled Chat")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Prefer the stronger model but fall back to the smaller one on resource errors
try:
    qa_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1,
    )
except Exception as e:
    st.warning(f"Could not load flan-t5-base, falling back to small. ({e})")
    qa_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1,
    )

st.sidebar.markdown(
    """
    <div style="padding: 0.25rem 0 0.75rem 0;">
        <div style="font-size: 1.35rem; font-weight: 800; letter-spacing: -0.04em; color: #f8fafc;">
            NeuraDocs
        </div>
        <div style="margin-top: 0.25rem; color: #cbd5e1; font-size: 0.9rem; line-height: 1.4;">
            AI PDF workspace for chat, question generation, and simple explanations.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.sidebar.button("+ New Chat", use_container_width=True):
    st.session_state.chat_id = create_chat()
    st.session_state.question_input = ""
    st.rerun()

if "chat_id" not in st.session_state:
    existing_chats = db.all()
    st.session_state.chat_id = existing_chats[0]["chat_id"] if existing_chats else create_chat()

if "question_batch" not in st.session_state:
    st.session_state.question_batch = 1

if "generated_questions" not in st.session_state:
    st.session_state.generated_questions = ""

current_chat = db.get(Chat.chat_id == st.session_state.chat_id)
if not current_chat:
    st.session_state.chat_id = create_chat()
    current_chat = db.get(Chat.chat_id == st.session_state.chat_id)

if st.session_state.get("active_chat_id") != st.session_state.chat_id:
    st.session_state.active_chat_id = st.session_state.chat_id
    st.session_state.question_batch = 1
    st.session_state.generated_questions = ""

    if current_chat:
        auto_questions = [
            message["answer"]
            for message in current_chat.get("messages", [])
            if message.get("question") in {"[Auto] Generated Questions", "[Auto] More Questions"}
        ]
        if auto_questions:
            st.session_state.generated_questions = "\n\n".join(auto_questions)
            st.session_state.question_batch = len(auto_questions) + 1

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='font-size: 0.9rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #c4b5fd; margin: 0.5rem 0 0.75rem;'>Recent Chats</div>", unsafe_allow_html=True)

all_chats = sorted(db.all(), key=lambda x: x.get("created_at", ""), reverse=True)
for chat in all_chats:
    with st.sidebar.expander(chat.get("name", "Untitled Chat")):
        # OPEN CHAT
        if st.button(
            "Open",
            key=f"open_{chat['chat_id']}",
            use_container_width=True
        ):
            st.session_state.chat_id = chat["chat_id"]
            st.session_state.question_input = ""
            st.rerun()

        st.divider()

        # DELETE CHAT
        if st.button(
            "Delete",
            key=f"delete_{chat['chat_id']}",
            use_container_width=True
        ):
            db.remove(
                lambda x: x["chat_id"] == chat["chat_id"]
            )
            chats = db.all()
            if chats:
                st.session_state.chat_id = chats[0]["chat_id"]
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("NeuraDocs")

# Main content starts here

st.subheader("Upload Documents")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # read uploaded bytes and save file locally
    uploaded_bytes = uploaded_file.read()

    # save file to uploaded_pdfs folder
    save_path = os.path.join("uploaded_pdfs", uploaded_file.name)
    try:
        with open(save_path, "wb") as f:
            f.write(uploaded_bytes)
    except Exception as e:
        st.error(f"Failed to save uploaded PDF: {e}")

    # extract text from the saved bytes
    try:
        pdf_reader = PdfReader(io.BytesIO(uploaded_bytes))
        texts = []
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                texts.append(content)

        text = " ".join(texts)
        text = text[:50000]
    except Exception:
        text = ""

    # persist text and saved path(s) in DB
    existing_paths = current_chat.get("pdf_paths", []) if current_chat else []
    new_paths = existing_paths + [save_path]

    db.update(
        {
            "pdf_text": text,
            "pdf_paths": new_paths,
        },
        Chat.chat_id == st.session_state.chat_id,
    )

    current_chat["pdf_text"] = text
    current_chat["pdf_paths"] = new_paths

    st.success("PDF uploaded and saved")

    # show previews from saved pdf paths
    st.subheader("Document Preview")
    for pdf_path in new_paths:
        try:
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as pdf_file:
                    pdf_viewer(pdf_file.read(), width=700)
        except Exception:
            st.info("PDF preview not available for this document")

current_chat = db.get(Chat.chat_id == st.session_state.chat_id)

# If there are saved pdf paths for this chat, show previews
if current_chat:
    if current_chat.get("pdf_paths"):
        st.subheader("Document Preview")
        for pdf_path in current_chat.get("pdf_paths", []):
            if os.path.exists(pdf_path):
                try:
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_viewer(pdf_file.read(), width=700)
                except Exception:
                    st.info("PDF preview not available for this document")

text = current_chat.get("pdf_text", "")

# Build semantic chunks/index once so all actions can reuse them.
chunks = []
index = None
if text:
    chunk_size = 300
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i : i + chunk_size])

    if chunks:
        embeddings = embedding_model.encode(chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype("float32"))


def get_relevant_text(query: str) -> str:
    if not chunks or index is None:
        return ""

    query_embedding = embedding_model.encode([query])
    _, indices = index.search(np.array(query_embedding).astype("float32"), k=5)

    relevant_text = ""
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            relevant_text += chunks[idx]

    return relevant_text

st.divider()

if not text:
    st.info("Upload a PDF in this chat to start asking questions.")

st.subheader("Ask Questions")

input_col, button_col1, button_col2 = st.columns([5, 1.6, 1.6])

with input_col:
    question = st.text_input(
        "Ask something about the PDF", key="question_input", disabled=not bool(text), label_visibility="collapsed"
    )

with button_col1:
    generate_questions = st.button("Generate Questions", use_container_width=True)

with button_col2:
    explain_simple = st.button("Explain Simply", use_container_width=True)

if generate_questions:
    if not text or not chunks or index is None:
        st.info("Upload a PDF first.")
    else:
        st.session_state.question_batch = 1

        with st.spinner("Generating questions..."):
            relevant_text = get_relevant_text("important concepts and topics")

            prompt = f"""
            You must generate questions ONLY from the provided PDF context.

            STRICT RULES:
            - Do not use outside knowledge
            - Questions must come only from the PDF
            - Do not repeat questions
            - Number all questions clearly

            Generate EXACTLY 10 questions.

            Include:
            - MCQs
            - Short Questions
            - Viva Questions

            PDF Context:
            {relevant_text}
            """

            try:
                response = qa_pipeline(prompt, max_length=700)
                generated_questions = response[0]["generated_text"]
            except Exception as e:
                st.error(f"Question generation failed: {e}")
                generated_questions = ""

            generated_questions = " ".join(generated_questions.split())
            st.session_state.generated_questions = generated_questions

            st.markdown(
                """
                <div class="bot-message">
                <h3>Generated Questions</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(st.session_state.generated_questions)

            updated_messages = current_chat.get("messages", [])
            updated_messages.append({"question": "[Auto] Generated Questions", "answer": generated_questions})
            db.update({"messages": updated_messages}, Chat.chat_id == st.session_state.chat_id)
            st.session_state.question_batch = 2
            st.rerun()

if st.session_state.generated_questions and text and chunks and index is not None:
    if st.button("More Questions", use_container_width=True):
        with st.spinner("Generating more questions..."):
            relevant_text = get_relevant_text("important concepts and topics")

            prompt = f"""
            Generate 10 NEW questions ONLY from the PDF context.

            STRICT RULES:
            - No repeated questions
            - No outside knowledge
            - Questions must come only from the PDF
            - Number all questions clearly

            Previous Questions:
            {st.session_state.generated_questions}

            PDF Context:
            {relevant_text}
            """

            try:
                response = qa_pipeline(prompt, max_length=700)
                new_questions = response[0]["generated_text"]
            except Exception as e:
                st.error(f"More questions generation failed: {e}")
                new_questions = ""

            new_questions = " ".join(new_questions.split())
            st.session_state.generated_questions += "\n\n" + new_questions
            st.session_state.question_batch += 1

            st.markdown(
                """
                <div class="bot-message">
                <h3>More Questions</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(new_questions)

            updated_messages = current_chat.get("messages", [])
            updated_messages.append({"question": "[Auto] More Questions", "answer": new_questions})
            db.update({"messages": updated_messages}, Chat.chat_id == st.session_state.chat_id)
            st.rerun()

if explain_simple:
    if not text or not chunks or index is None:
        st.info("Upload a PDF first.")
    else:
        with st.spinner("Simplifying content..."):
            relevant_text = get_relevant_text("important topics and concepts")

            prompt = f"""
            Explain this PDF content in very easy language.

            Rules:
            - Use beginner-friendly explanations
            - Use simple examples
            - Break complex ideas simply
            - Make it easy for students

            PDF Context:
            {relevant_text}
            """

            try:
                response = qa_pipeline(prompt, max_length=700)
                easy_explanation = response[0]["generated_text"]
            except Exception as e:
                st.error(f"Simplification failed: {e}")
                easy_explanation = ""

            easy_explanation = " ".join(easy_explanation.split())

            st.markdown(
                """
                <div class="bot-message">
                <h3>Easy Explanation</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(easy_explanation)

            updated_messages = current_chat.get("messages", [])
            updated_messages.append({"question": "[Auto] Easy Explanation", "answer": easy_explanation})
            db.update({"messages": updated_messages}, Chat.chat_id == st.session_state.chat_id)
            st.rerun()

if question and text:
    with st.spinner("Generating response..."):
        if not chunks or index is None:
            st.warning("No readable text found in this PDF.")
            st.stop()

        question_embedding = embedding_model.encode([question])
        _, indices = index.search(np.array(question_embedding).astype("float32"), k=5)

        relevant_text = ""
        for idx in indices[0]:
            if 0 <= idx < len(chunks):
                relevant_text += chunks[idx]

        prompt = f"""
        Answer the question using the context below.

        Context:
        {relevant_text}

        Question:
        {question}
        """

        response = qa_pipeline(prompt, max_length=200)
        answer = response[0]["generated_text"]

        updated_messages = current_chat.get("messages", [])
        updated_messages.append({"question": question, "answer": answer})

        db.update({"messages": updated_messages}, Chat.chat_id == st.session_state.chat_id)
        st.session_state.question_input = ""
        st.rerun()

current_chat = db.get(Chat.chat_id == st.session_state.chat_id)

st.subheader("Conversation")

def render_chat_card(role: str, content: str) -> None:
    safe_content = html.escape(content).replace("\n", "<br>")
    is_user = role == "You"
    role_label = "You" if is_user else "Assistant"
    accent = "#ec4899" if is_user else "#8b5cf6"
    role_bg = "rgba(236,72,153,0.15)" if is_user else "rgba(139,92,246,0.15)"

    st.markdown(
        f"""
        <div class="{'user-message' if is_user else 'bot-message'}" style="position: relative; overflow: hidden;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                <span style="display:inline-flex; align-items:center; justify-content:center; width: 2rem; height: 2rem; border-radius: 999px; background: {role_bg}; color: {accent}; font-weight: 800; font-size: 0.8rem;">
                    {"Y" if is_user else "A"}
                </span>
                <span style="display:inline-flex; align-items:center; padding: 0.3rem 0.65rem; border-radius: 999px; background: rgba(255,255,255,0.06); color: #e2e8f0; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;">
                    {role_label}
                </span>
            </div>
            <div style="color: #f8fafc; line-height: 1.7; white-space: normal;">
                {safe_content}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for chat_message in current_chat.get("messages", []):
    render_chat_card("You", chat_message["question"])
    render_chat_card("Assistant", chat_message["answer"])
