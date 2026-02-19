import os
from dotenv import load_dotenv
from firecrawl import FirecrawlApp

load_dotenv()


class FirecrawlService:
    def __init__(self):
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError("Missing FIRECRAWL_API_KEY environment variable")
        self.app = FirecrawlApp(api_key=api_key)

    def search_companies(self, query: str, num_results: int = 5):
        try:
            result = self.app.search(
                query=query,
                limit=num_results,
                scrape_options={"formats": ["markdown"]}
            )
            return result.data if hasattr(result, "data") else []
        except Exception as e:
            print(e)
            return []

    def scrape_company_pages(self, url: str):
        try:
            if not isinstance(url, str) or not url.startswith("http"):
                return None

            result = self.app.scrape(
                url,
                formats=["markdown"]
            )
            return result
        except Exception as e:
            print(e)
            return None
