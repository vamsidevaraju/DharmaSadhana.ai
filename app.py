import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Set your Google API key here
os.environ['GOOGLE_API_KEY']= "Your_api_key"

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Define the file paths for the PDFs
pdf_file_paths = ["/Users/vamsi/venv-metal/DharmaSadhana/COI...pdf"]  # Add your PDF file paths here

def get_pdf_text(pdf_file_paths):
    text = ""
    for pdf_path in pdf_file_paths:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "I Can't help you with irrelevent context", don't provide the wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the FAISS index with dangerous deserialization allowed
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    return response["output_text"]

def main():
    st.set_page_config(page_title="DharmaSadhana.ai")
    st.header("DharmaSadhana.ai - Your AI Legal Companion ")
    st.info("⚠️ **Disclaimer:**\nThis AI Agent is trained with the Constitution of India. On this basis, The agent will be responding to your legal queries.")

    if "history" not in st.session_state:
        st.session_state.history = []

    user_question = st.chat_input(placeholder="Your message")
    if user_question:
        response = user_input(user_question)
        st.session_state.history.append({"user": user_question, "bot": response})

    for chat in st.session_state.history:
        st.chat_message(name="User", avatar="👤").write(chat['user'])
        st.chat_message(name="DharmaSadhana", avatar="🤖").write(chat['bot'])

    with st.sidebar:
        st.title("Memory:")
        if st.button("Activate"):
            with st.spinner("Activating Memory..."):
                try:
                    raw_text = get_pdf_text(pdf_file_paths)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Memory Activated successfully!")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()




