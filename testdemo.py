from playwright.sync_api import sync_playwright

def scrape_page_content(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Set headless=True for scraping
        page = browser.new_page()


        # Navigate to the URL
        page.goto(url)
        page.wait_for_selector("text='Allow cookies'", timeout=60000)  # Wait up to 60 seconds
        page.locator("text='Allow cookies'").click()

        # Extract all headings (h1, h2, h3, etc.)
        headings = page.locator("h1, h2, h3, h4, h5, h6").all_inner_texts()

        # Extract all links along with their text
        links = page.locator("a").evaluate_all(
            "(elements) => elements.map(el => ({text: el.innerText, href: el.href}))"
        )

        # Extract the main content (e.g., paragraphs or body)
        main_content = page.locator("body").inner_text()

        browser.close()
        return {
            "headings": headings,
            "links": links,
            "main_content": main_content,
        }

# Example usage
data = scrape_page_content("https://www.vinnova.se/en/calls-for-proposals/global-cooperation-2024/planning-grant-for-international-proposal-2024/")
print(data)