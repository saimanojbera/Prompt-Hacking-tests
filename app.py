import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from datetime import date
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Load vector store and retriever
def get_retriever():
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# Format retrieved chunks for prompt
def format_chunks(docs):
    formatted_chunks = []
    for i, doc in enumerate(docs):
        text = doc.page_content.strip()
        source = doc.metadata.get("source", "Unknown Source")
        chunk_id = f"chk-{i+1:02d}"
        chunk = f"Content: {text}\nsource_name: {source}\nchunk_id: {chunk_id}"
        formatted_chunks.append(chunk)
    return "\n\n".join(formatted_chunks), formatted_chunks


# Streamlit UI
st.title("🔍 RAG QA System")
st.markdown("Enter your query and system prompt below:")

custom_system_prompt = st.text_area("System Prompt", "", height=300)
user_query = st.text_area("User Query", "What types of loans are available for education?", height=100)
submit = st.button("Run Query")

if submit:
    if not custom_system_prompt.strip():
        st.error("System prompt is required.")
    else:
        retriever = get_retriever()
        docs = retriever.invoke(user_query)
        context_block, readable_chunks = format_chunks(docs)

        user_message = f"Context: {context_block}\nQuery: '{user_query}'"

        messages = [
            {"role": "system", "content": custom_system_prompt},
            {"role": "user", "content": user_message}
        ]

        with st.spinner("🧠 Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=1068
            )

        st.subheader("💬 Response")
        st.markdown(response.choices[0].message.content)

        with st.expander("📚 Retrieved Chunks"):
            for chunk in readable_chunks:
                st.markdown(f"```\n{chunk}\n```")