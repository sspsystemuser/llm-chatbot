from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from langchain.llms import OpenAI
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)
# Function to set up Playwright
def setup_playwright():
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=True)
    return browser

# Function to fetch rendered HTML content
def fetch_page_content(url, browser):
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')  # Wait for page to fully load
    content = page.content()
    page.close()
    return content

# Function to extract answers based on a question
def extract_answer_from_content(html, question):
    soup = BeautifulSoup(html, 'html.parser')
    paragraphs = soup.find_all(['p', 'li', 'div'])
    for paragraph in paragraphs:
        text = paragraph.get_text().strip()
        if question.lower() in text.lower():
            return text
    return "No relevant answer found."

# Initialize OpenAI LLM
llm = OpenAI(model_name="gpt-3.5-turbo")

# Function to get a summarized answer
def get_summary_answer(content, question):
    prompt = f"Here is the content extracted from the web page:\n{content}\n\nAnswer the question: {question}"
    response = llm(prompt)
    return response

# Main function for Streamlit UI
def main():
    st.title("LangChain LLM Bot with Playwright Scraping")
    st.write("Enter a URL and a question to get an answer from the webpage content.")

    # User inputs
    url = st.text_input("Enter the URL:")
    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if url and question:
            st.write("Fetching and processing content. Please wait...")

            # Initialize Playwright
            browser = setup_playwright()
            try:
                # Fetch content using Playwright
                html_content = fetch_page_content(url, browser)

                # Extract answer from HTML content
                extracted_answer = extract_answer_from_content(html_content, question)

                # Get summarized answer using LangChain LLM
                summary_answer = get_summary_answer(extracted_answer, question)

                # Display the result
                st.subheader("Extracted Answer:")
                st.write(extracted_answer)

                st.subheader("Summarized Answer from LLM:")
                st.write(summary_answer)
            finally:
                # Close browser
                browser.close()
        else:
            st.error("Please provide both URL and question.")

if __name__ == "__main__":
    main()