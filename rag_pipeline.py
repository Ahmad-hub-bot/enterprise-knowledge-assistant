import os
import glob
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional

FAISS_PATH = "/tmp/faiss_index"
DOCS_PATH = "documents"

# ---------- State ----------
class GraphState(TypedDict):
    question: str
    category: Optional[str]
    retrieved_docs: Optional[List[Document]]
    confidence: Optional[float]
    answer: Optional[str]
    citation: Optional[str]

# ---------- Load documents ----------
def load_documents():
    all_docs = []

    for pdf in glob.glob(f"{DOCS_PATH}/*.pdf"):
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        for d in docs:
            d.metadata["source_type"] = "policy"
            d.metadata["filename"] = pdf.split("/")[-1]
        all_docs.extend(docs)

    for docx in glob.glob(f"{DOCS_PATH}/*.docx"):
        loader = Docx2txtLoader(docx)
        docs = loader.load()
        for d in docs:
            d.metadata["source_type"] = "sop"
            d.metadata["filename"] = docx.split("/")[-1]
        all_docs.extend(docs)

    for csv_path in glob.glob(f"{DOCS_PATH}/*.csv"):
        filename = csv_path.split("/")[-1]
        try:
            if "EXPENSE" in filename.upper():
                df = pd.read_csv(csv_path, skiprows=4, header=0)
            else:
                raw = pd.read_csv(csv_path, header=None, nrows=10)
                header_row = None
                for i, row in raw.iterrows():
                    vals = [str(v) for v in row if str(v) != "nan"]
                    if any("LGF" in v or "Borrowing" in v or "Expense category" in v for v in vals):
                        header_row = i
                        break
                if header_row is None:
                    continue
                df = pd.read_csv(csv_path, skiprows=header_row, header=0)

            df.dropna(how="all", inplace=True)
            df.dropna(axis=1, how="all", inplace=True)
            first_col = df.columns[0]
            df = df[~df[first_col].astype(str).str.contains(
                "Back to|Note|Source|£|Contents|nan|Select|This worksheet|Some|Columns|A blank",
                case=False, na=True
            )]
            df = df[df[first_col].astype(str).str.strip() != ""]

            for _, row in df.iterrows():
                parts = []
                for col in df.columns:
                    val = row[col]
                    if pd.notna(val) and str(val).strip() not in ["", "nan", "[z]", "[x]", "[i]", "[r]"]:
                        parts.append(f"{col.strip()}: {str(val).strip()}")
                if len(parts) >= 2:
                    all_docs.append(Document(
                        page_content=" | ".join(parts),
                        metadata={"source_type": "financial", "filename": filename}
                    ))
        except Exception:
            continue

    return all_docs


# ---------- Vector store ----------
def build_vector_store():
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_PATH)
    return vector_store


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)


# ---------- LLM ----------
def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0)


# ---------- Graph nodes ----------
def make_classify_node(llm):
    def classify_query(state: GraphState) -> GraphState:
        question = state["question"]
        prompt = f"""Classify the following question into exactly one category: policy, financial, sop, or out_of_scope.

Rules:
- "policy" = questions about company policies, HR rules, leave, conduct, business continuity
- "financial" = questions about budgets, expenses, borrowing, investments, financial figures
- "sop" = questions about standard operating procedures, IT processes, work orders
- "out_of_scope" = anything unrelated to the above

Question: {question}

Respond with ONLY one word: policy, financial, sop, or out_of_scope"""

        response = llm.invoke(prompt)
        category = response.content.strip().lower()
        if category not in ["policy", "financial", "sop"]:
            category = "out_of_scope"
        state["category"] = category
        return state
    return classify_query


def make_retrieve_node(vector_store):
    def retrieve_documents(state: GraphState) -> GraphState:
        question = state["question"]
        category = state["category"]
        results = vector_store.similarity_search_with_score(
            question, k=3, filter={"source_type": category}
        )
        state["retrieved_docs"] = [doc for doc, _ in results]
        if results:
            avg_score = sum(score for _, score in results) / len(results)
            state["confidence"] = max(0, 1 - (avg_score / 2))
        else:
            state["confidence"] = 0.0
        return state
    return retrieve_documents


def make_generate_node(llm):
    def generate_answer(state: GraphState) -> GraphState:
        question = state["question"]
        docs = state["retrieved_docs"]
        confidence = state["confidence"]
        context = "\n\n".join([d.page_content for d in docs])
        sources = list(set([d.metadata["filename"] for d in docs]))

        prompt = f"""Answer the question using ONLY the context below. If the context does not contain the answer, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

        response = llm.invoke(prompt)
        answer = response.content

        if confidence < 0.5:
            answer += "\n\n⚠️ Low confidence — recommend escalation to a human reviewer."

        state["answer"] = answer
        state["citation"] = ", ".join(sources)
        return state
    return generate_answer


def out_of_scope_response(state: GraphState) -> GraphState:
    state["answer"] = "This question is outside the scope of the available documents (policy, financial, SOP)."
    state["citation"] = "N/A"
    state["confidence"] = 0.0
    return state


# ---------- Build graph ----------
def build_graph(vector_store, llm):
    def route_decision(state: GraphState) -> str:
        return "out_of_scope" if state["category"] == "out_of_scope" else "retrieve"

    graph = StateGraph(GraphState)
    graph.add_node("classify", make_classify_node(llm))
    graph.add_node("retrieve", make_retrieve_node(vector_store))
    graph.add_node("generate", make_generate_node(llm))
    graph.add_node("out_of_scope", out_of_scope_response)

    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", route_decision, {
        "retrieve": "retrieve",
        "out_of_scope": "out_of_scope"
    })
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("out_of_scope", END)

    return graph.compile()
