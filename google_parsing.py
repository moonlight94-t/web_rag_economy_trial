import urllib.parse
import feedparser
from langchain.docstore.document import Document
from googlesearch import search
import trafilatura
import requests
import time
from email.utils import parsedate_to_datetime


def search_googlenews(user_query, max_count=10, blacklist=None, us_flag=False):
    # template = (
    #     "https://news.google.com/rss/search?q={query}&ceid={ceid}&hl={hl}&gl={gl}"
    # )
    template = "https://news.google.com/rss/search?q={query}+when:30d&ceid={ceid}&hl={hl}&gl={gl}"  # when 조건 추가

    req_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    }

    if us_flag:
        ceid = "US:en"
        hl = "en"
        gl = "US"
    else:
        ceid = "KR:ko"
        hl = "ko"
        gl = "KR"

    encoded_query = urllib.parse.quote(user_query)

    rss_url = template.format(query=encoded_query, ceid=ceid, hl=hl, gl=gl)

    feed = feedparser.parse(rss_url)

    results = []
    count = 0
    for entry in feed.entries:
        if blacklist:
            if entry["title"].rsplit(" - ", 1)[-1] in blacklist:
                continue
        time.sleep(2)
        url = next(search(entry["title"], stop=2, pause=2.0), None)
        text = None
        try:
            time.sleep(1)
            html = requests.get(url, timeout=10, headers=req_headers)
            text = trafilatura.extract(html.text)
        except:
            continue

        if text is None:
            continue

        # if keyword_match(user_query, text):
        dt = parsedate_to_datetime(entry.published)
        results.append(
            Document(
                page_content=text,
                metadata={
                    "source": url,
                    "publisher": (
                        entry["title"].rsplit(" - ", 1)[-1]
                        if " - " in entry["title"]
                        else entry["title"]
                    ),
                    "title": (
                        entry["title"].rsplit(" - ", 1)[0]
                        if " - " in entry["title"]
                        else entry["title"]
                    ),
                    "published": dt,
                    "published_ts": int(dt.timestamp()),
                },
            )
        )
        count += 1
        if count >= max_count:
            break

    if len(results) == 0:
        return None
    else:
        return results
