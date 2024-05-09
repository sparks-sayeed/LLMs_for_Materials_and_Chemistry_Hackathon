import os
import shutil

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from openai import OpenAI

DATA_DIR = "uploaded_pdfs"


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
    paragraphs = [doc.page_content for doc in docs]
    return paragraphs


OPENAI_API_KEY = "api_key"


def extract_JSON(paragraph, json_schema, system_content=None, user_content=None):
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
                "content": system_content
                + f"{paragraph}\n\nEmpty JSON Schema: {json_schema}\n\nAnswer:",
            },
            {"role": "user", "content": user_content},
        ],
    )

    # Extract and return the completed JSON content
    return response.choices[0].message.content


json_schema_default = """{
  "composition": "",
  "property": "",
  "value": "",
  "unit": "",
  "measurement_condition": ""
}"""

json_schema_hea = """{
  "composition": [
    {
      "chemical_composition": "",
      "element": "",
      "percentage": ""
    }
  ],
  "phase_structure": "",
  "properties_measured": [
    {
      "property": "",
      "value": "",
      "unit": ""
    }
  ],
  "measurement_conditions": "",
  "treatment": ""
}"""


def main():

    # Custom CSS to inject into the Streamlit page
    st.markdown(
        """
    <style>
    /* Targeting the JSON formatter class */
    .json-container {
        color: green;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Upload the PDF")
    st.title("Decoding PDFs: Retrieving Structured JSON")

    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF file", type="pdf", accept_multiple_files=False
    )
    paragraphs = (
        []
    )  # Initialize paragraphs as an empty list to ensure it's always defined

    if uploaded_file:

        select_box = st.sidebar.selectbox(
            "Prompts", ("Default", "High Entropy Alloy", "Pervoskite")
        )

        if select_box == "Default":
            system_content = """Below is a chunk of text from a scientific literature.
            You are a helpful assistant for researchers needing to extract data for
            their analyses. Your task is to fill out the provided empty JSON schema
            with relevant information extracted from the text. Make sure to fill out
            a JSON for each materials present in the text. If data for a specific
            material is absent, put 'NONE' in the corresponding field. Fill out
            multiple JSON if multiple material is present.\n\nParagraph:"""

            user_content = """Fill out the JSON schema with the appropriate
            information extracted from the paragraph. Ensure that the keys
            match the provided schema, and provide the values for composition
            name, property name, value, unit, and measurement condition
            accordingly."""

            json_schema = json_schema_default
            st.sidebar.write("Selected Prompt:", system_content)
            st.sidebar.write("json_schema")
            st.sidebar.json(json_schema)

        # if select_box =="Super Conductor":
        #     system_content = """Below is a chunk of text from a
        #     scientific literature discussing the critical
        #     temperature of a superconductor. You are a helpful
        #     assistant for researchers needing to extract data
        #     for their analyses. Your task is to fill out the
        #     provided empty JSON schema with relevant information
        #     extracted from the text. Make sure to fill out a
        #     JSON for each materials present in the text. If
        #     data for a specific material is absent, put
        #     'NONE' in the corresponding field. Fill out
        #     multiple JSON if multiple material is
        #     present.\n\nParagraph:"""

        #     user_content = """Fill out the JSON schema with the
        #     appropriate information extracted from the paragraph.
        #     Ensure that the keys match the provided schema, and
        #     provide the values for composition name, property
        #     name, value, unit, and measurement condition
        #     accordingly."""

        #     st.sidebar.write("Selected Prompt:",system_content)
        #     st.sidebar.write("json_schema")
        #     st.sidebar.json(json_schema)

        if select_box == "High Entropy Alloy":
            system_content = """Below is an excerpt from a scientific paper
            discussing High-Entropy Alloys (HEAs). You are a helpful assistant
            for researchers needing to extract data for their analyses.
            Your task is to fill out the provided empty JSON schema with
            relevant information extracted from the text. Make sure to
            fill out a JSON for each materials present in the text.
            Please focus on the composition of the alloys, phase
            structures, properties measured, conditions under which
            these properties were measured, and any treatment processes
            applied. If data for a specific material is absent, put
            'NONE' in the corresponding field. Fill out multiple
            JSON if multiple material is present.\n\nParagraph:"""

            user_content = """Fill out the JSON schema with the appropriate
            information extracted from the paragraph. Ensure that the keys
            match the provided schema, and provide the values for
            composition name, property name, value, unit, and
            measurement condition accordingly."""

            json_schema = json_schema_hea
            st.sidebar.write("Selected Prompt:", system_content)
            st.sidebar.write("json_schema")
            st.sidebar.json(json_schema)

        if select_box == "Pervoskite":
            system_content = """Below is a chunk of text from a
            scientific literature discussing Pervoskite. You
            are a helpful assistant for researchers needing
            to extract data for their analyses. Your task is
            to fill out the provided empty JSON schema with
            relevant information extracted from the text.
            Make sure to fill out a JSON for each materials
            present in the text. If data for a specific
            material is absent, put 'NONE' in the corresponding
            field. Fill out multiple JSON if multiple material
            is present.\n\nParagraph:"""

            user_content = """Fill out the JSON schema with the
            appropriate information extracted from the paragraph.
            Ensure that the keys match the provided schema,
            and provide the values for composition name, property
            name, value, unit, and measurement condition
            accordingly."""

            json_schema = json_schema_hea
            st.sidebar.write("Selected Prompt:", system_content)
            st.sidebar.write("json_schema")
            st.sidebar.json(json_schema)

        if st.sidebar.button("Load and Process PDF"):
            with st.spinner("Loading and Processing PDF..."):
                clean_and_create_directory()
                # saved_file = save_uploaded_file(uploaded_file)
                paragraphs = load_files(
                    DATA_DIR
                )  # Now paragraphs will only be updated if this block executes
                st.session_state.processed = True
                st.success("PDF loaded successfully ...")

    if st.session_state.get("processed", False):
        with st.spinner("Extracting JSON"):
            responses = []
            page_number = st.empty()
            if len(paragraphs) > 0:
                count = 0
                for i, paragraph in enumerate(paragraphs):
                    page_number.text(f"Processing page number: {i + 1}")
                    # print(system_content)
                    response = extract_JSON(
                        paragraph, json_schema, system_content, user_content
                    )
                    responses.append(response)
                    count = count + 1
                st.success("Extracted JSON succesfully")
                st.json(responses)

            else:
                st.write(
                    """nothing to display select a pdf file and press
                    'Load and Process PDF' button"""
                )


if __name__ == "__main__":
    main()
