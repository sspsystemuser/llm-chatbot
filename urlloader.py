import os
from langchain_community.document_loaders import AsyncChromiumLoader
import asyncio
from langchain_community.document_transformers import Html2TextTransformer, BeautifulSoupTransformer
from langchain_openai import ChatOpenAI
from langchain.chains import create_extraction_chain
import pprint
import numpy as np
import streamlit as st
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from playwright.async_api import async_playwright
from playwright.async_api import Page
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

# Set USER_AGENT environment variable
#os.environ['USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# Define constants for user agent and viewport
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
VIEWPORT = {'width': 1920, 'height': 1080}


llm = ChatOpenAI(temperature=0, model="gpt-4o")
embeddings_model = OpenAIEmbeddings()  # Initialize OpenAI embeddings model

# Initialize FAISS index
faiss_index = None
urls = [] 

MAX_DEPTH = 3  # Set a maximum recursion depth
DELAY_BETWEEN_REQUESTS = 1  

async def wait_for_page_load(page: Page):
    await page.wait_for_load_state('networkidle')
    await page.wait_for_load_state('domcontentloaded')
    await page.wait_for_load_state('load')


async def extract_links(page: Page):
    await wait_for_page_load(page)
    
    # Extract links from the current page
    links = await page.evaluate(""" 
        () => {
            const links = Array.from(document.querySelectorAll('a'));
            return links.map(link => link.href).filter(url => url);  // Filter out empty URLs
        }
    """)
    return links

async def scrape_all_links(url: str, visited: set, depth: int=0):
    if depth > MAX_DEPTH:  # Check if maximum depth is reached
        return set()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport=VIEWPORT, user_agent=USER_AGENT)
        page = await context.new_page()
        
        visited.add(url)
        await page.goto(url, wait_until='networkidle', timeout=60000)
        
        links = await extract_links(page)
        all_links = set(links)  # Use a set to avoid duplicates
        
        for link in links:
            if link not in visited:
                visited.add(link)
                await asyncio.sleep(DELAY_BETWEEN_REQUESTS)  # Throttle requests
                sub_links = await scrape_all_links(link, visited,depth+1)  # Recursively scrape sub-URLs
                all_links.update(sub_links)
        
        await page.close()
        await browser.close()
        
        return all_links

def embed_urls(urls):
    # Use OpenAI embeddings to convert URLs to vectors
    vectors = []
    for url in urls:
        try:
            print(f"Embedding URL: {url}") 
            vector = embeddings_model.embed(url)  # Get the embedding for the URL
            if vector is not None and len(vector) > 0:
                vectors.append(vector)
                print(f"Successfully embedded: {url}")
            else:
                print(f"Embedding failed for URL: {url}")
        except Exception as e:
            print(f"Error embedding URL {url}: {e}")
    return np.array(vectors).astype('float32') if vectors else np.array([])

def store_in_faiss(vectors):
    if vectors.size == 0:
        print("No vectors to store in FAISS.")
        return None
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Create a FAISS index
    index.add(vectors)  # Add vectors to the index
    return index

async def run_scraping(start_url):
    # Run the scraping function in the event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    visited = set()
    all_links = loop.run_until_complete(scrape_all_links(start_url, visited))
    print(f"Scraped {len(all_links)} links.")

    # Store URLs for reference
    global urls
    urls = list(all_links)

    # Convert URLs to vectors
    if urls:
        print(f"Embedding {len(urls)} URLs.")
        vectors = embed_urls(urls)
        print(f"Generated {len(vectors)} vectors.")
    
        # Store vectors in FAISS
        global faiss_index
        if len(vectors) > 0:
            faiss_index = store_in_faiss(vectors)
            print("Stored vectors in FAISS.")
        else:
            print("No vectors generated.")
    else:
        print("No URLs scraped.")

def find_similar_urls(query):
    # Embed the query
    query_vector = embeddings_model.embed(query)
    query_vector = np.array([query_vector]).astype('float32')

    # Search in FAISS
    distances, indices = faiss_index.search(query_vector, k=5)  # Get top 5 results
    similar_urls = [urls[i] for i in indices[0]]
    return similar_urls

def generate_answer(query, context):
    # Use the LLM to generate an answer based on the context
    response = llm(f"Answer the question: '{query}' based on the following context: {context}")
    return response

# Streamlit UI
st.title("LLM Chatbot with URL Scraping")
start_url = st.text_input("Enter the URL to scrape:")
if st.button("Scrape"):
    if start_url:
        run_scraping(start_url)
        st.success("Scraping completed!")
        st.session_state.scraped = True

user_query = st.text_input("Ask a question:")
if st.button("Get Answer"):
    if 'scraped' in st.session_state and st.session_state.scraped:
        if faiss_index is not None:
            similar_urls = find_similar_urls(user_query)
            context = " ".join(similar_urls)  # Combine URLs for context
            answer = generate_answer(user_query, context)
            st.write("Answer:", answer)
            st.write("Source URLs:", similar_urls)
        else:
            st.error("No vectors found in FAISS.")
    else:
        st.error("Please scrape a URL first.")



