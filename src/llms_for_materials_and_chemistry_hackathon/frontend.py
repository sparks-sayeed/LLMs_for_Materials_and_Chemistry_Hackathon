import streamlit as st
import os
import shutil
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

DATA_DIR = 'uploaded_pdfs'

def clean_and_create_directory():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join(DATA_DIR, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return os.path.join(DATA_DIR, uploaded_file.name)
    except Exception as e:
        st.sidebar.error(f"Error saving file: {e}")
        return None

def load_files(folder_path):
    loaders = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loaders.append(PyPDFLoader(os.path.join(folder_path, filename)))
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs

def process_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    return splits

def initialize_embeddings():
    model_kwargs = {'device': 'mps'}  # Adjust according to your available device
    embedding_model_name = "WhereIsAI/UAE-Large-V1"
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)
    return embedding

def initialize_chroma(embedding, splits):
    vectordb = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory="chroma_db/")
    return vectordb, vectordb.as_retriever()

def get_llm_response(question, retriever, hf_llm):
    template = """<bos><start_of_turn>user
            {question} \You are a helpful assistant, your name is XYZ. Answer the query by using the reports in quote:\n "{context}" \n 
            If you are not confident then say I don't know.<end_of_turn>
            <start_of_turn>model
            """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["question", "context"], template=template)
    qa_chain = RetrievalQA.from_chain_type(hf_llm, retriever=retriever, chain_type_kwargs={"prompt":QA_CHAIN_PROMPT}, return_source_documents=False)
    result = qa_chain({"query": question})
    response = result["result"]
    if response:
        return response.split("<start_of_turn>model")[-1]
    return "I don't know."

MODEL_LLM_NAME = "google/gemma-2b-it"
def setup_huggingface_pipeline():
    return HuggingFacePipeline.from_model_id(model_id=MODEL_LLM_NAME, task="text-generation", pipeline_kwargs={"max_new_tokens": 500})



def main():
    
    st.title("PDF Query With Vector Database")
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True)

    if 'embedding' not in st.session_state:
        st.session_state.embedding = initialize_embeddings()
        st.session_state.hf_llm = setup_huggingface_pipeline()

    # uploaded_files = st.sidebar.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True)
    if uploaded_files:
        if st.sidebar.button("Process and Load PDFs"):
            with st.spinner('Processing and loading PDFs...'):
                clean_and_create_directory()
                saved_files = [save_uploaded_file(f) for f in uploaded_files]
                docs = load_files(DATA_DIR)
                splits = process_documents(docs)
                embedding = st.session_state.embedding
                vectordb, retriever = initialize_chroma(embedding, splits)
                st.session_state.retriever = retriever
                st.session_state.processed = True
                st.success("PDFs processed and data loaded into the vector database.")

    if 'retriever' in st.session_state and st.session_state.get('processed', False):
        select_box = st.sidebar.selectbox("prompts",("HEA","Super Conductor","Thermo Electric"))
        if select_box =="HEA":
            st.write('prompt: i am here HEA expert')
        if select_box =="Super Conductor":
            st.write('prompt: i am here SC expert')
        if select_box =="Thermo Electric":
            st.write('prompt: i am here TC expert')

        query = st.text_input("Ask a question:")
        if st.button('Submit'):
            with st.spinner('Getting response...'):
                response = get_llm_response(query, st.session_state.retriever, st.session_state.hf_llm)
                st.write('Response:', response)

if __name__ == "__main__":
    main()

