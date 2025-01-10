import streamlit as st
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import requests
import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

load_dotenv()

def scrape_url_with_suburls(main_url, max_depth=1):
    """
    Scrape the main URL and its sub-URLs up to a specified depth.
    """
    scraped_data = {}
    visited_urls = set()
    def scrape(url, depth):
        if depth > max_depth or url in visited_urls:
            return
        visited_urls.add(url)
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            scraped_data[url] = soup.get_text()
            for link in soup.find_all('a', href=True):
                sub_url = link['href']
                if sub_url.startswith('/') or sub_url.startswith(main_url):
                    full_url = sub_url if sub_url.startswith('http') else main_url + sub_url
                    scrape(full_url, depth + 1)
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    scrape(main_url, 0)
    return scraped_data


def create_vector_store(data):
    """
    Create a FAISS vector store from scraped data.
    """
    texts = []
    metadata = []
    for url, content in data.items():
        texts.append(content)
        metadata.append({"source": url})

    # Split large texts
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.create_documents(texts, metadata)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def build_qa_chain(vector_store):
    """
    Build a Q&A retrieval chain.
    """
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)
    return qa_chain

def get_user_input(prompt):
    """
    Helper function to get user input in Streamlit.
    """
    return prompt.strip()


st.title("Web Scraper Chatbot")

# Input URL
main_url = st.text_input("Enter the main URL to scrape:", placeholder="https://www.vinnova.se/en/apply-for-funding/find-the-right-funding/")

# Input maximum depth for sub-URL scraping
max_depth = st.slider("Select scraping depth:", min_value=1, max_value=3, value=1)

if st.button("Scrape and Build Chatbot"):
    if not main_url:
        st.error("Please enter a valid URL.")
    else:
        st.write("Scraping the website...")
        scraped_data = scrape_url_with_suburls(main_url, max_depth)
        if not scraped_data:
            st.error("No data found. Please check the URL.")
        else:
            st.write("Data scraped successfully! Building chatbot...")
            vector_store = create_vector_store(scraped_data)
            qa_chain = build_qa_chain(vector_store)

            st.success("Chatbot is ready!")

            # Chat interface
            st.subheader("Chat with your bot:")
            user_input = st.text_input("Enter your question:")
            if user_input:
                response = qa_chain.run(user_input)
                st.write("**Answer:**", response)