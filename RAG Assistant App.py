!pip install --upgrade langchain-google-genai langchain-pinecone langchain -qqq
!pip install --upgrade langchain langchain-core langchain-community langchain-google-genai langchain-pinecone streamlit -qqq


import os

# Injecting keys directly into the environment state
os.environ['GOOGLE_API_KEY'] = 'AQ.Ab8RN6LdQIhYYVh50FyPPZ0zF0LwRvHCAniM1owQWFnRBXHueQ'
os.environ['PINECONE_API_KEY'] = 'pcsk_5YUQ8L_7NWNfC4219AfEjqygUGFJcreC9MWPuqbFUMUPii4GCi48n7AP5Tasc4MHuaDgk9'
os.environ['PINECONE_INDEX_NAME'] = 'pureplate-rag-index'

"""
app.py - Domain Support Assistant Conversational RAG Engine
Author: PurePlate AgroFood AI Team
Description: Modern, environment-safe Streamlit UI built with LangChain Expression 
             Language (LCEL), engineered with robust multi-turn embedding guardrails.
"""

import os
import streamlit as st

# Modern LangChain Core Imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser

# Partner Vector Store & Resilient LLM Integrations
from langchain_pinecone import PineconeVectorStore

# Resilient Fallback Architecture for Google GenAI Package Versions
try:
    from langchain_google_genai import GoogleGenAIEmbeddings, ChatGoogleGenerativeAI
except ImportError:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_google_genai import GoogleGenerativeAIEmbeddings as GoogleGenAIEmbeddings

# Page Configuration
st.set_page_config(page_title="PurePlate Support AI", page_icon="🌱", layout="centered")
st.title("🌱 PurePlate AgroFood Assistant")
st.caption("Official enterprise workspace chatbot for policies, FAQs, and vacancies.")

# 1. Environment Guardrails Validation
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pureplate-rag-index")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    st.error("Missing environment variables configuration! Please verify GOOGLE_API_KEY and PINECONE_API_KEY are configured.")
    st.stop()

# 2. Session Context State Tracking Setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Tracks interaction history states cleanly

@st.cache_resource
def initialize_rag_pipeline():
    """Initializes explicit 768-dim embeddings and Gemini 2.5 Flash-Lite safely."""
    try:
        # Enforcing output_dimensionality=768 prevents the Pinecone dimension mismatch error
        # Removing global task_type parameters prevents downstream 500 INTERNAL errors on multi-turns
        embeddings = GoogleGenAIEmbeddings(
            model="models/gemini-embedding-001",
            output_dimensionality=768
        )
        
        # Establishing connection to Pinecone Index
        vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Explicit model routing configured to leverage Gemini 2.5 Flash-Lite
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)
        return retriever, llm
    except Exception as e:
        st.error(f"Failed to bootstrap underlying RAG components: {str(e)}")
        return None, None

retriever, llm = initialize_rag_pipeline()

if retriever and llm:
    
    # --- STEP 1: MODERN LCEL CONTEXT REWRITER (Query Condenser) ---
    context_rephrase_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it exactly as-is."
    )
    context_rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", context_rephrase_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Declarative LCEL Chain for question condensation (Output is parsed cleanly as a string)
    condense_question_chain = context_rephrase_prompt | llm | StrOutputParser()
    
    # Optimize latency by bypassing condenser when message history is empty
    query_condenser = RunnableBranch(
        (lambda x: len(x.get("chat_history", [])) > 0, condense_question_chain),
        RunnablePassthrough() | (lambda x: x["input"])
    )

    # --- STEP 2: GROUNDED QA CHAIN (Strict Prompt Guardrails) ---
    qa_system_prompt = (
        "You are a strict PurePlate AgroFood Customer and Employee Support Assistant. "
        "Your priority is safety, truthfulness, and strict alignment to corporate resources.\n"
        "Answer the user's question ONLY using the provided retrieved context snippets below.\n"
        "Do not extrapolate, do not use external training data assumptions, and do not make up facts.\n\n"
        "If the true answer cannot be confidently derived entirely from the provided context chunks, you MUST "
        "reply exactly with: 'I don’t have enough information in the provided documents.'\n\n"
        "Retrieved Context Chunks:\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # --- STEP 3: ASSEMBLE FULL CONVERSATIONAL LCEL RAG PIPELINE ---
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Pre-computation payload assembly chain using modern pipeline operators and protected string casting
    rag_chain = (
        RunnablePassthrough.assign(
            condensed_query=query_condenser
        )
        | RunnablePassthrough.assign(
            context=lambda x: retriever.invoke(str(x["condensed_query"]))
        )
        | RunnablePassthrough.assign(
            context_str=lambda x: format_docs(x["context"])
        )
    )

    # Render History Logs onto active UI state on script refresh ticks
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Active Interactive Prompt Input Loop
    if user_query := st.chat_input("How can I assist you today regarding PurePlate?"):
        # Display immediate client mutation block
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Append directly to state prior to chain invoke to preserve temporal UI logic
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.chat_message("assistant"):
            try:
                # Format memory state arrays into strict explicit LangChain message objects
                langchain_history = []
                for msg in st.session_state.chat_history[:-1]:  # Exclude current frame
                    if msg["role"] == "user":
                        langchain_history.append(HumanMessage(content=msg["content"]))
                    else:
                        langchain_history.append(AIMessage(content=msg["content"]))

                with st.spinner("Analyzing corporate knowledge base..."):
                    # Process documentation extraction and translation layers
                    processed_payload = rag_chain.invoke({
                        "input": user_query,
                        "chat_history": langchain_history
                    })
                    
                    # Empty context guardrail to bypass model costs when data is completely absent
                    retrieved_docs = processed_payload.get("context", [])
                    if not retrieved_docs:
                        answer = "I don’t have enough information in the provided documents."
                    else:
                        # Feed arguments cleanly downstream into strict validation prompt
                        final_qa_chain = qa_prompt | llm
                        qa_response = final_qa_chain.invoke({
                            "input": str(processed_payload["condensed_query"]),
                            "chat_history": langchain_history,
                            "context": processed_payload["context_str"]
                        })
                        answer = qa_response.content
                
                # Render Synthesized Output
                st.markdown(answer)
                
                # Render source validation metadata within expanding UI panels
                if retrieved_docs:
                    with st.expander("🔍 View Checked Source Context Citations"):
                        for idx, doc in enumerate(retrieved_docs):
                            source_name = doc.metadata.get('source', 'Unknown Document')
                            page_num = doc.metadata.get('page', 0)
                            st.write(f"**Source [{idx+1}]:** {os.path.basename(source_name)} (Page {int(page_num)+1})")
                            st.caption(doc.page_content[:200] + "...")
                
                # Sync final assistant data point back onto active memory state array
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                # Instantly force a page rerun to clean layout alignment stuttering
                st.rerun()

            except Exception as e:
                st.error(f"An unexpected loop exception stopped execution processing safely: {str(e)}")
