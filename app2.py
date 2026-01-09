import streamlit as st
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# -------------------- Streamlit Setup --------------------
st.title("RAG PDF Q&A with OpenAI & Chroma")
st.write("Upload PDFs and ask questions.")

# -------------------- OpenAI LLM --------------------
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

if openai_api_key:
    llm = ChatOpenAI(
        api_key=openai_api_key,
        model="gpt-3.5-turbo",
        temperature=0
    )

    # -------------------- Chat History --------------------
    session_id = st.text_input("Session ID", value="default_session")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}

    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = []

    def get_session_history(sess_id):
        return st.session_state.chat_history[sess_id]

    # -------------------- PDF Upload --------------------
    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        # Save PDFs locally
        temp_dir = "./temp_pdfs"
        os.makedirs(temp_dir, exist_ok=True)

        for f in uploaded_files:
            with open(os.path.join(temp_dir, f.name), "wb") as file:
                file.write(f.getvalue())

        # Load documents
        loader = PyPDFDirectoryLoader(temp_dir)
        documents = loader.load()

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        final_docs = splitter.split_documents(documents)

        # Create embeddings + vectorstore
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        vectorstore = Chroma.from_documents(final_docs, embeddings)
        retriever = vectorstore.as_retriever()

        st.success("Vector database ready!")

        # -------------------- Query Pipeline --------------------
        def pipeline_query(user_query):
            # ✅ FIX: use invoke() (NO Document import needed)
            docs = retriever.invoke(user_query)

            context = "\n\n".join(doc.page_content for doc in docs)

            chat_history = get_session_history(session_id)
            history_text = "\n".join(
                f"User: {m['user']}\nAssistant: {m['assistant']}"
                for m in chat_history
            )

            prompt = ChatPromptTemplate.from_template("""
            Answer the question based on the following context and chat history.
            If you don't know the answer, say you don't know.

            Chat History:
            {history}

            Context:
            {context}

            Question:
            {query}
            """)

            input_prompt = prompt.format(
                history=history_text,
                context=context,
                query=user_query
            )

            response = llm.invoke(input_prompt).content

            chat_history.append({
                "user": user_query,
                "assistant": response
            })

            return response

        # -------------------- User Input --------------------
        user_input = st.text_input("Your question:")
        if user_input:
            answer = pipeline_query(user_input)
            st.write("### Assistant")
            st.write(answer)

            with st.expander("Chat History"):
                for chat in get_session_history(session_id):
                    st.write("**User:**", chat["user"])
                    st.write("**Assistant:**", chat["assistant"])
                    st.write("---")
