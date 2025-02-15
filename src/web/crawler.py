import requests
from bs4 import BeautifulSoup
import re
from fake_useragent import UserAgent


class KnowledgeCrawler:
    def __init__(self, search_engine="bing"):
        self.ua = UserAgent()
        self.headers = {'User-Agent': self.ua.random}
        self.search_engine = search_engine

    def _get_search_url(self, query):
        return {
            "bing": f"https://www.bing.com/search?q={query}&count=10"
        }[self.search_engine]

    def search(self, query, max_results=5):
        try:
            res = requests.get(
                self._get_search_url(query),
                headers=self.headers,
                timeout=5
            )
            soup = BeautifulSoup(res.text, 'html.parser')

            results = []
            for item in soup.select('li.b_algo')[:max_results]:
                link = item.find('a')['href']
                if content := self._crawl_page(link):
                    results.append(content)
            return results
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

    def _crawl_page(self, url):
        try:
            page = requests.get(url, headers=self.headers, timeout=5)
            soup = BeautifulSoup(page.text, 'html.parser')
            # 移除无用元素
            for tag in soup(['script', 'style', 'nav', 'footer']):
                tag.decompose()
            text = ' '.join([p.get_text() for p in soup.find_all('p')])
            return re.sub(r'\s+', ' ', text).strip()
        except:
            return ""
