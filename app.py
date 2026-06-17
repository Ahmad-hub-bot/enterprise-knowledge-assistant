import streamlit as st
from rag_pipeline import build_vector_store, load_vector_store, build_graph, get_llm
import os

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="🧠", layout="centered")

st.title("🧠 Enterprise Knowledge Assistant")
st.caption("Ask questions about company policy, financials, or IT procedures.")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Setup")
    if st.button("🔄 Rebuild Vector Store", use_container_width=True):
        with st.spinner("Building vector store... this may take a minute."):
            st.session_state["vector_store"] = build_vector_store()
        st.success("Vector store ready!")

    st.markdown("---")
    st.markdown("**Document Categories**")
    st.markdown("📄 Policy — HR & business continuity")
    st.markdown("💰 Financial — UK borrowing & expense data")
    st.markdown("🖥️ SOP — IT procedures")

# ---------- Init ----------
@st.cache_resource(show_spinner="Loading pipeline...")
def init_pipeline():
    llm = get_llm()
    if os.path.exists("/tmp/faiss_index"):
        vector_store = load_vector_store()
    else:
        vector_store = build_vector_store()
    app = build_graph(vector_store, llm)
    return app

app = init_pipeline()

# ---------- Confidence badge ----------
def confidence_badge(score):
    if score >= 0.75:
        return "🟢 High confidence"
    elif score >= 0.5:
        return "🟡 Medium confidence"
    else:
        return "🔴 Low confidence"

# ---------- Main UI ----------
question = st.text_input("Ask a question:", placeholder="e.g. How does a user report an IT problem?")

if st.button("Ask", use_container_width=True) and question.strip():
    with st.spinner("Thinking..."):
        result = app.invoke({"question": question})

    st.markdown("---")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"**Category:** `{result['category']}`")
    with col2:
        confidence = result.get("confidence", 0)
        st.markdown(f"**{confidence_badge(confidence)}** ({confidence:.2f})")

    st.markdown("### Answer")
    st.write(result["answer"])

    st.markdown(f"**Source:** `{result['citation']}`")
