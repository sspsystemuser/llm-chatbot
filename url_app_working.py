import streamlit as st
import os
import pickle
import time
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import faiss
import tempfile
import re 
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

def is_valid_url(url):
    """check if the URL is valid"""
    if not url or not url.strip():
        return False
    try:
        result = urlparse(url)
        return all([result.scheme,result.netloc])
    except:
        return False

st.title("News Research Tool")
st.sidebar.title("News Article URLS")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = os.path.join(tempfile.gettempdir(),"faiss_store_openai.pkl")
main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)
if process_url_clicked:
    # Validate URLs before processing
    valid_urls = [url.strip() for url in urls if is_valid_url(url)]

    if not valid_urls:
        st.error("Please enter at least one valid URL. URLs should start with http:// or https://")
    else:
        try:
            # show which URLS are being prcessed
            st.info(f"Processing {len(valid_urls)} valid URLs:")
            for url in valid_urls:
                st.write(f"- {url}")
            # load data
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text("Data Loading...Started...✅✅✅")
            data = loader.load()
            # split data
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Text Splitter...Started...✅✅✅")
            docs = text_splitter.split_documents(data)
            # create embeddings and save it to FAISS index
            embeddings = OpenAIEmbeddings()
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            main_placeholder.text("Embedding Vector Started Building...✅✅✅")
            time.sleep(20)

            # Save the FAISS index to a pickle file
            #save with error handling
            try:
                vectorstore_openai.save_local(file_path)
                main_placeholder.success(f"Index saved successfully to {file_path}")
            except PermissionError:
                st.error(f"Permission denied: unable to save to {file_path}. Please check your file permisson.")
            except Exception as e:
                st.error(f"Error saving index: {str(e)}")
            
        except Exception as e:
            st.error(f"Error processing URLs: {str(e)}")


query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        # Add a warning message
        st.warning("⚠️ Loading saved index. Make sure you trust the source of the index file.")
        try:
            # Ensure we have read permissions
            if os.access(file_path, os.R_OK):
                # Load the FAISS index with safe deserialization
                vectorstore = FAISS.load_local(
                    file_path, 
                    OpenAIEmbeddings(),
                    allow_dangerous_deserialization=True
                )
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain.invoke({"question": query}, return_only_outputs=True)
                
                st.header("Answer")
                st.write(result["answer"])

                # Display sources, if available
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")
                    for source in sources_list:
                        st.write(source)
            else:
                st.error(f"cannot read the index file. Please check your file permissions for {file_path}.")
        except Exception as e:
            st.error(f"Error loading index: {str(e)}")

