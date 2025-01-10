import os
from langchain_community.document_loaders import AsyncChromiumLoader
import asyncio
from langchain_community.document_transformers import Html2TextTransformer, BeautifulSoupTransformer
from langchain_openai import ChatOpenAI
from langchain.chains import create_extraction_chain
import pprint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from playwright.async_api import async_playwright
from playwright.async_api import Page
from bs4 import BeautifulSoup 
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

# Set USER_AGENT environment variable
#os.environ['USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# Define constants for user agent and viewport
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
VIEWPORT = {'width': 1280, 'height': 800}

schema = {
    "properties": {
        "article_title": {"type": "string"},
        "article_url": {"type": "string"},
    },
    "required": ["article_title", "article_url"],
}

def extract(content: str, schema: dict):
    chain = create_extraction_chain(schema=schema,llm=llm)
    return chain.invoke(content)


llm = ChatOpenAI(temperature=0, model="gpt-4o")

async def wait_for_page_load(page: Page):
    await page.wait_for_load_state('networkidle')
    await page.wait_for_load_state('domcontentloaded')
    await page.wait_for_load_state('load')

async def custom_page_handler(page):
    # Wait for the page to load completely
    await wait_for_page_load(page)
    
    # Scroll multiple times to trigger lazy loading
    for _ in range(3):  # Scroll 3 times
        await page.evaluate("""
            window.scrollTo(0, document.body.scrollHeight);
            new Promise((resolve) => setTimeout(resolve, 1000));  // Wait 1s after each scroll
        """)
        await page.wait_for_timeout(1000)  # Additional wait after scroll
    
    # Wait for common content containers
    selectors = ['article', '.article', 'main', '.content', '#content']
    for selector in selectors:
        try:
            await page.wait_for_selector(selector, timeout=5000)
            break  # Stop if any selector is found
        except:
            continue
    
    # Extract links and headings
    links_and_headings = await page.evaluate("""
        () => {
            const links = Array.from(document.querySelectorAll('a'));
            return links.map(link => ({
                url: link.href,
                heading: link.textContent.trim()
            })).filter(item => item.url && item.heading);  // Filter out empty items
        }
    """)
    return links_and_headings

async def extract_links(urls):
    all_links_and_headings = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Set headless=True in production
        context = await browser.new_context(
            viewport=VIEWPORT,  # Larger viewport
            user_agent=USER_AGENT
        )
        
        for url in urls:
            try:
                page = await context.new_page()
                await page.goto(url, wait_until='networkidle', timeout=60000)  # Increased timeout
                links_and_headings = await custom_page_handler(page)
                all_links_and_headings.extend(links_and_headings)
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
            finally:
                await page.close()
                
        await browser.close()
    return all_links_and_headings

async def scrape_sub_url_content():
    sub_urls=["https://www.vinnova.se/en/calls-for-proposals/global-cooperation-2024/planning-grant-for-international-proposal-2024/",
              "https://www.vinnova.se/en/calls-for-proposals/partnership-futurefoods/european-collaborations-for-sustainable-2024-03369/"
              ]
    data = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False,args=["--disable-blink-features=AutomationControlled"])
        context = await browser.new_context(
            viewport=VIEWPORT,
            user_agent=USER_AGENT, # Ensure this is 
            java_script_enabled=True,
            locale="en-US"  
        )
        # Set request interception on the context
        await context.route("**/*", lambda route, request: route.continue_()) 
        page = await context.new_page()
        

        for sub in sub_urls:
            await page.goto(sub,wait_until='networkidle')
            await page.wait_for_timeout(5000) 
            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")
            
            # Extract full content (adjust selectors based on website structure)
            content = " ".join([p.text for p in soup.find_all("p")])
            data.append({'url': sub, 'heading': '', 'content': content})

        await browser.close()

    return data
async def main():
    sub_url_data = await scrape_sub_url_content()
    print(sub_url_data)
        
        

if __name__ == "__main__":
    asyncio.run(main())

