import os
import shutil

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from openai import OpenAI

OPENAI_API_KEY = "api_key"

DATA_DIR = "uploaded_pdfs"


def clean_and_create_directory():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)


# def save_uploaded_file(uploaded_file):
#     try:
#         with open(os.path.join(DATA_DIR, uploaded_file.name), "wb") as f:
#             f.write(uploaded_file.getbuffer())
#         return os.path.join(DATA_DIR, uploaded_file.name)
#     except Exception as e:
#         st.sidebar.error(f"Error saving file: {e}")
#         return None


def load_files(folder_path):
    loaders = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loaders.append(PyPDFLoader(os.path.join(folder_path, filename)))
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs


def extract_JSON(paragraph, json_schema):
    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Create completion request
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        temperature=0,
        seed=42,
        # response_format={"type": "json_object"},
        max_tokens=4096,
        messages=[
            {
                "role": "system",
                "content": f"""Below is a chunk of text from a scientific literature
                discussing the critical temperature of a superconductor. You are a
                helpful assistant for researchers needing to extract data for their
                analyses. Your task is to fill out the provided empty JSON schema
                with relevant information extracted from the text. Make sure to fill
                out a JSON for each materials present in the text. If data for a
                specific material is absent, put 'NONE' in the corresponding field.
                Fill out multiple JSON if multiplematerial is present.
                \n\nParagraph: {paragraph}\n\nEmpty JSON Schema:
                {json_schema}\n\nAnswer:""",
            },
            {
                "role": "user",
                "content": """Fill out the JSON schema with the appropriate information
                extracted from the paragraph. Ensure that the keys match the provided
                schema, and provide the values for composition name, property name,
                value, unit, and measurement condition accordingly.""",
            },
        ],
    )

    # Extract and return the completed JSON content
    return response.choices[0].message.content


json_schema = """{
  "composition": "",
  "property": "",
  "value": "",
  "unit": "",
  "measurement_condition": ""
}"""


def main():
    st.title("PDF Query With Vector Database")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        if st.sidebar.button("Process and Load PDFs"):
            with st.spinner("Processing and loading PDFs..."):
                clean_and_create_directory()
                # saved_files = [save_uploaded_file(f) for f in uploaded_files]
                docs = load_files(DATA_DIR)
            response = extract_JSON(docs, json_schema)
            st.write("Response:", response)

    if "retriever" in st.session_state and st.session_state.get("processed", False):
        select_box = st.sidebar.selectbox(
            "prompts", ("HEA", "Super Conductor", "Thermo Electric")
        )
        if select_box == "HEA":
            st.write("prompt: i am here HEA expert")
        if select_box == "Super Conductor":
            st.write("prompt: i am here SC expert")
        if select_box == "Thermo Electric":
            st.write("prompt: i am here TC expert")


if __name__ == "__main__":
    main()
