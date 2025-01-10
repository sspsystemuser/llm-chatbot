import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from langchain_chroma import Chroma 
import openai
from dotenv import load_dotenv
import os
import shutil
import argparse
import magic
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
CHROMA_PATH = "C:\ProjectData\GitProject\coax\llm-bot\chroma"
DATA_PATH = "C:\ProjectData\GitProject\coax\llm-bot\data"

def reduce_markdown_file_size(file_path: str):
    output_file_path = file_path.replace('.md', '_reduced.md')
    try:
        with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as file:
            lines = file.readlines()

        with open(output_file_path, 'w', encoding='utf-8') as file:
            for line in lines:
                # Remove empty lines and lines that are comments
                if line.strip() and not line.strip().startswith('#'):
                    file.write(line.strip() + '\n')
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

    return output_file_path

def load_documents():
    jq_schema = ". as $content | {page_content: $content, metadata: {url: .url}}"

    loader = DirectoryLoader(DATA_PATH, glob="*.json",loader_cls=lambda p: JSONLoader(file_path=p, jq_schema=jq_schema,text_content=False))
    #documents = loader.load()
    documents = []
    try:
        loaded_documents = loader.load()  # Load all documents at once
        for doc in loaded_documents:
            # Assuming each document has a 'metadata' attribute with the file path
            file_path = doc.metadata.get("source", None)  # Adjust based on your actual metadata structure
            if file_path:
                # Reduce the size of the markdown file
                # reduced_file_path = reduce_markdown_file_size(file_path)
                # Instead of loading the reduced file, process the loaded document
                documents.append(doc)
    except Exception as e:
        print(f"Error loading documents: {e}")

    return documents


def split_text(documents: list[Document]):    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    if not chunks:  # Check if chunks is empty
        print("No chunks were created. Please check the input documents.")
        return [] 


    document = chunks[3]
    print(document.page_content)
    print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    if not chunks:  # Check if chunks is empty
        print("No chunks to save to Chroma. Exiting.")
        return 
    save_to_chroma(chunks)

PROMPT_TEMPLATE = """
Answer the question based on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    st.title("CO-AX Research Tool")
    # generate_data_store()
    query_text = st.text_input("Your input:")
    if query_text:
        # Prepare the DB.
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=10)
        if len(results) == 0 or results[0][1] < 0.7:
            print(f"Unable to find matching results.")
            st.write("Unable to find matching results from vector database.")    


        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        print(f"*************************************************************************")
        print(f"Context Text: {context_text}")
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        print(f"*************************************************************************")
        print(f"prompt Text: {prompt}")    

        model = ChatOpenAI()
        response_text = model.predict(prompt)

        print("URL Metadata:", results[0])

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        print(f"*************************************************************************")
        print(formatted_response)

        st.header("Answer:")
        if response_text: 
            st.write(response_text)  
  
        else:
            st.write("Not found anything related to your query")    
            

if __name__ == "__main__":
    main()