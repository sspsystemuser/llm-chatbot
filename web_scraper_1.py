import getpass
from typing import List, Optional, TypedDict, Any
from typing_extensions import Annotated
from operator import or_
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from IPython.display import Image, display
import logging

# === Logging Configuration ===
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set the root logger level to INFO

# Remove all handlers associated with the root logger object.
if logger.hasHandlers():
    logger.handlers.clear()

# Create a new handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)  # Set handler level to INFO

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the root logger
logger.addHandler(handler)
# === End of Logging Configuration ===

def first_non_null(a, b):
    return a if a is not None else b

# FIRECRAWL_API_KEY = getpass.getpass("Enter your FireCrawal API Key:")
FIRECRAWL_API_KEY = "fc-6fc0d7c6b69746869c0478af483bf874"

# Define the InputState schema
class InputState(TypedDict):
    url: str
    keyword: str

# Define the OverallState schema with reducers
class OverallState(TypedDict):
    urls: List[str]
    current_url_index: int
    total_urls: int
    urls_to_scrape: List[str]
    extracted_info: Annotated[Optional[str], first_non_null]
    extracted_from_url: Annotated[Optional[str], first_non_null]
    is_information_found: Annotated[Optional[bool], or_]
    keyword: str

def first_node(state: InputState) -> OverallState:
    logging.info("Executing node: first_node")
    return {
        "urls": [state["url"]],
        "current_url_index": 0,
        "total_urls": 0,
        "urls_to_scrape": [],
        "keyword": state["keyword"],
        "extracted_info": None,
        "extracted_from_url": None,
        "is_information_found": False,
    }

from firecrawl import FirecrawlApp

def get_sitemap(state: OverallState, config):
    logging.info("Executing node: get_sitemap")
    api_key = config.get("configurable", {}).get("firecrawl_api_key")
    if not api_key:
        logging.error("Firecrawl API key is missing in the configuration.")
        raise ValueError("Firecrawl API key is missing in the configuration.")

    app = FirecrawlApp(api_key=api_key)

    # Map the website to get the list of URLs
    map_result = app.map_url(state["urls"][0])
    logging.info(f"Map result: {map_result}")

    # Handle the new response format which is a dictionary
    if isinstance(map_result, dict):
        if map_result.get('success') and isinstance(map_result.get('links'), list):
            sitemap = map_result['links']
            state["urls"] = sitemap
            state["total_urls"] = len(sitemap)
            logging.info(f"Found {len(sitemap)} URLs to scrape.")
        else:
            logging.error(f"Error: Invalid response structure in map_result: {map_result}")
            state["urls"] = []
    # Keep the original list handling for backward compatibility
    elif isinstance(map_result, list) and len(map_result) > 0:
        sitemap = map_result
        state["urls"] = sitemap
        state["total_urls"] = len(sitemap)
        logging.info(f"Found {len(sitemap)} URLs to scrape.")
    else:
        logging.error(f"Error: Unexpected format in map_result: {map_result}")
        state["urls"] = []

    return state

def scrape_manager(state: OverallState) -> OverallState:
    logging.info("Executing node: scrape_manager")
    total_urls = state.get("total_urls", 0)
    current_index = state["current_url_index"]
    next_index = min(current_index + 3, total_urls)
    urls_to_scrape = state["urls"][current_index:next_index]
    state["current_url_index"] = next_index  # Update the current index
    state["urls_to_scrape"] = urls_to_scrape  # Store the URLs to scrape

    if total_urls > 0:
        progress = (current_index / total_urls) * 100
        logging.info(f"Progress: {current_index}/{total_urls} URLs processed ({progress:.2f}%)")
    return state

# Function to return Send objects to scraper nodes
def continue_to_scraper(state: OverallState):
    logging.info(f"Preparing to scrape URLs {state['current_url_index'] - len(state['urls_to_scrape'])} to {state['current_url_index'] - 1}")
    # Return list of Send objects to 'scraper' node
    return [Send("scraper", {"url": url, "keyword": state["keyword"]}) for url in state["urls_to_scrape"]]

from firecrawl import FirecrawlApp

def scraper(state: OverallState, config):
    url = state["url"]
    keyword = state["keyword"]
    logging.info(f"Executing node: scraper for URL: {url}")
    api_key = config.get("configurable", {}).get("firecrawl_api_key")
    if not api_key:
        logging.error("Firecrawl API key is missing in the configuration.")
        raise ValueError("Firecrawl API key is missing in the configuration.")

    app = FirecrawlApp(api_key=api_key)

    # Scrape the URL and get markdown
    try:
        logging.info(f"Started scraping: {url}")
        scrape_result = app.scrape_url(url, params={'formats': ['markdown']})
        logging.info(f"Scraped URL: {url}")
    except Exception as e:
        logging.error(f"Exception occurred while scraping URL: {url}. Error: {e}")
        scrape_result = {}

    if isinstance(scrape_result, dict):
        markdown = scrape_result.get("markdown", "")
        # Check if keyword is in markdown
        if markdown and keyword in markdown:
            # Information found
            return {
                "extracted_info": markdown,
                "extracted_from_url": url,
                "is_information_found": True
            }
        else:
            # Information not found
            return {"is_information_found": False}
    else:
        logging.warning(f"Failed to scrape URL: {url}")
        return {"is_information_found": False}


def evaluate(state: OverallState):
    logging.info("Executing node: evaluate")
    # Logic is handled in the routing function
    return state

# Conditional routing function for evaluate node
def evaluate_routing(state: OverallState):
    if state.get("is_information_found", False):
        logging.info(f"Information found in URL: {state.get('extracted_from_url')}")
        logging.info("Terminating graph execution.")
        return "END"
    elif state["current_url_index"] < len(state["urls"]):
        logging.info("Information not found, continuing to scrape next batch.")
        return "scrape_manager"
    else:
        logging.info("No more URLs to scrape, terminating graph execution.")
        return "END"
    

# Build the graph with configuration schema
class ConfigSchema(TypedDict):
    firecrawl_api_key: str

builder = StateGraph(OverallState, input=InputState, config_schema=ConfigSchema)

builder.add_node("first_node", first_node)
builder.add_node("get_sitemap", get_sitemap)
builder.add_node("scrape_manager", scrape_manager)
builder.add_node("scraper", scraper)
builder.add_node("evaluate", evaluate)

# Add edges between nodes
builder.add_edge(START, "first_node")
builder.add_edge("first_node", "get_sitemap")
builder.add_edge("get_sitemap", "scrape_manager")

# Add conditional edges from scrape_manager to scraper nodes
builder.add_conditional_edges("scrape_manager", continue_to_scraper, ["scraper"])

builder.add_edge("scraper", "evaluate")

# Add conditional edge from evaluate node with explicit mapping
builder.add_conditional_edges(
    "evaluate",
    evaluate_routing,
    {"END": END, "scrape_manager": "scrape_manager"}
)

# Compile the graph
graph = builder.compile()

# Visualize the graph
display(Image(graph.get_graph().draw_mermaid_png()))

config = {"configurable": {"firecrawl_api_key": FIRECRAWL_API_KEY}, "recursion_limit": 50}

input_state = {
    "url": "https://www.vinnova.se/en/apply-for-funding/find-the-right-funding/",
    "keyword": ""
}

result = graph.invoke(input_state, config=config)    