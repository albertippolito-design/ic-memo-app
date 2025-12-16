import requests
from pathlib import Path
from datetime import datetime
import re
import time

# Try to use yfinance which exposes news in a stable way. If not available,
# fall back to scraping Yahoo Finance pages.
try:
    import yfinance as yf
except Exception:
    yf = None
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

OUTPUT = Path("news.txt")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
                  " Chrome/120.0.0.0 Safari/537.36"
}


def fetch_yahoo_news_links(symbol, max_links=6):
    """Return a list of (title, url) tuples for the symbol's Yahoo Finance news page.

    This is only used as a fallback if `yfinance` is not available or returns no news.
    """
    # Try a generic quote page and look for headline anchors
    url = f"https://finance.yahoo.com/quote/{symbol}"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    if BeautifulSoup is None:
        return []
    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    seen = set()
    # Yahoo often includes headlines in specific sections; search for anchors with role or data-test attributes
    candidates = soup.find_all("a", href=True)
    for a in candidates:
        href = a["href"]
        text = a.get_text(separator=" ", strip=True)
        if not text or len(text) < 10:
            continue
        if href.startswith("/"):
            full = "https://finance.yahoo.com" + href
        elif href.startswith("http"):
            full = href
        else:
            continue
        if full in seen:
            continue
        # Heuristic: news/story/article in path or link text that looks like a headline
        if any(part in href for part in ("/news/", "/story/", "/article/")) or len(text) > 30:
            links.append((text, full))
            seen.add(full)
        if len(links) >= max_links:
            break
    return links


def fetch_snippet(url, max_sentences=2):
    """Fetch an article URL and return the first few sentences as a short snippet."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # prefer article <p> tags
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
        if not text.strip():
            return ""
        # simple sentence split
        sentences = re.split(r'(?<=[.!?])\s+', text)
        snippet = " ".join(s.strip() for s in sentences[:max_sentences])
        # truncate to a reasonable length
        if len(snippet) > 600:
            snippet = snippet[:600].rsplit(" ", 1)[0] + "..."
        return snippet.strip()
    except Exception:
        return ""


def summarize_stock_news(symbol):
    symbol = symbol.strip().upper()
    header = f"News summary for {symbol} — fetched {datetime.now().isoformat(sep=' ', timespec='seconds')}\n"
    header += "=" * 80 + "\n\n"

    # Prefer yfinance if available (more stable). Fallback to scraping.
    links = []
    if yf is not None:
        try:
            t = yf.Ticker(symbol)
            # yfinance returns a list of dicts with keys like 'title' and 'link'
            items = getattr(t, "news", None)
            if items:
                for it in items[:6]:
                    title = it.get("title") or it.get("publisher", "")
                    link = it.get("link") or it.get("providerPublishTime")
                    if title and link:
                        links.append((title, link))
        except Exception:
            links = []

    if not links:
        try:
            links = fetch_yahoo_news_links(symbol, max_links=6)
        except Exception as e:
            return header + f"Failed to fetch news page: {e}\n"

    if not links:
        return header + "No news links found on Yahoo Finance.\n"

    out_lines = [header]
    for i, (title, url) in enumerate(links, start=1):
        out_lines.append(f"{i}. {title}\n")
        snippet = fetch_snippet(url, max_sentences=2)
        if snippet:
            out_lines.append(snippet + "\n")
        out_lines.append(f"Link: {url}\n\n")
        # be polite
        time.sleep(0.5)

    # Very short extractive summary: join headlines and first fragments
    summary_lines = ["Top headlines:"]
    for i, (title, url) in enumerate(links, start=1):
        summary_lines.append(f"{i}. {title}")
    out_lines.append("---\nShort digest:\n")
    out_lines.extend(summary_lines)
    out_lines.append("\nEnd of summary.\n")

    return "\n".join(out_lines)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Yahoo Finance news for a ticker and save a short summary to news.txt")
    parser.add_argument("symbol", nargs="?", help="Ticker symbol (e.g. AAPL). If omitted, you'll be prompted.")
    args = parser.parse_args()

    if args.symbol:
        sym = args.symbol
    else:
        print("This script fetches major news for a stock from Yahoo Finance and saves a short summary to news.txt")
        sym = input("Enter the stock symbol (e.g. AAPL, TSLA): ").strip()

    if not sym:
        print("No symbol provided — exiting.")
    else:
        print(f"Fetching news for {sym}... (this may take a few seconds)")
        summary = summarize_stock_news(sym)
        try:
            OUTPUT.write_text(summary, encoding="utf-8")
            print(f"Saved summary to {OUTPUT.resolve()}")
        except Exception as e:
            print(f"Failed to write file: {e}")
        print("Done.")
