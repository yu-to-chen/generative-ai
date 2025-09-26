# makuake_top10.py
import os
import re
import json
import time
import base64
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
from playwright.sync_api import sync_playwright, Browser

BASE = "https://www.makuake.com"
RANKING_URL = f"{BASE}/discover/ranking/"
HOMEPAGE_URL = BASE + "/"
DISCOVER_CANDIDATES = [
    f"{BASE}/discover/most-funded/",
    f"{BASE}/discover/",
    f"{BASE}/discover/technology/",
    f"{BASE}/discover/fashion/",
    f"{BASE}/discover/food/",
    f"{BASE}/discover/outdoors/",
    f"{BASE}/discover/home-living/",
]

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/122.0.0.0 Safari/537.36")

def text_or_none(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    x = x.strip()
    return x if x else None

def translate_ja_to_zh(translator, text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    try:
        return translator.translate(text)
    except Exception as e:
        print(f"[WARN] Translation failed; keeping JA. Err={e}")
        return text

def parse_json_ld(soup: BeautifulSoup) -> List[dict]:
    out = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string)
            if isinstance(data, list):
                out.extend(data)
            elif isinstance(data, dict):
                out.append(data)
        except Exception:
            pass
    return out

def extract_brand_from_page(soup: BeautifulSoup) -> Optional[str]:
    for block in parse_json_ld(soup):
        if isinstance(block, dict):
            for key in ("brand", "author", "organizer", "creator", "publisher"):
                val = block.get(key)
                if isinstance(val, dict):
                    name = val.get("name")
                    if name:
                        return name
                elif isinstance(val, str):
                    return val
            if block.get("@type") in ("Organization", "Brand") and block.get("name"):
                return block.get("name")
    for label in soup.find_all(string=re.compile(r"実行者")):
        parent = label.parent
        if parent:
            sib = parent.find_next(string=True)
            if sib:
                cand = text_or_none(str(sib))
                if cand and "実行者" not in cand:
                    return cand
    return None

def extract_meta(soup: BeautifulSoup, prop: str) -> Optional[str]:
    tag = soup.find("meta", attrs={"property": prop})
    if tag and tag.get("content"):
        return text_or_none(tag["content"])
    return None

def is_clean_project_link(href: str) -> bool:
    return bool(re.search(r"^/project/[^/]+/?$", href))

def collect_project_links_from_dom(html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    hrefs = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("http"):
            if re.search(r"https?://(www\.)?makuake\.com/project/[^/]+/?$", href):
                hrefs.add(href if href.endswith("/") else href + "/")
        elif href.startswith("/project/"):
            if is_clean_project_link(href):
                hrefs.add(BASE + (href if href.endswith("/") else href + "/"))
    return list(hrefs)

def render_and_get_html(browser: Browser, url: str, timeout_s: int) -> Optional[str]:
    """
    Launches a new context & page from an already-launched Browser.
    (Fixes: 'Playwright' object has no attribute 'new_context')
    """
    try:
        ctx = browser.new_context(
            user_agent=UA,
            locale="ja-JP",
            timezone_id="Asia/Tokyo",
            viewport={"width": 1366, "height": 900},
        )
        # reduce obvious automation signals
        ctx.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        page = ctx.new_page()
        page.set_default_timeout(timeout_s * 1000)
        page.goto(url, wait_until="domcontentloaded")
        # Let hydration/lazy-load finish; brief scroll to trigger content
        page.wait_for_timeout(1500)
        for y in (600, 1200, 2400):
            try:
                page.mouse.wheel(0, y)
                page.wait_for_timeout(250)
            except Exception:
                pass
        # If network-heavy, you can switch to 'networkidle' once:
        # page.goto(url, wait_until="networkidle")
        html = page.content()
        page.close()
        ctx.close()
        return html
    except Exception as e:
        print(f"[WARN] Playwright render failed for {url}: {e}")
        try:
            ctx.close()
        except Exception:
            pass
        return None

def get_project_links_playwright(browser: Browser, url: str, limit: int, timeout_s: int) -> List[str]:
    html = render_and_get_html(browser, url, timeout_s=timeout_s)
    if not html:
        return []
    links = collect_project_links_from_dom(html)
    # preserve DOM order via regex scan
    ordered = []
    seen = set()
    for m in re.finditer(r'href="([^"]+)"', html):
        href = m.group(1)
        if href.startswith("/project/") and is_clean_project_link(href):
            full = BASE + (href if href.endswith("/") else href + "/")
            if full not in seen and full in links:
                seen.add(full)
                ordered.append(full)
    if ordered:
        links = ordered
    uniq = []
    seen = set()
    for u in links:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq[:limit]

def get_any_10_projects_playwright(browser: Browser, timeout_s: int) -> List[str]:
    for url in [RANKING_URL, HOMEPAGE_URL] + DISCOVER_CANDIDATES:
        print(f"[*] Trying source: {url}")
        links = get_project_links_playwright(browser, url, limit=20, timeout_s=timeout_s)
        if links:
            if url == RANKING_URL and len(links) >= 10:
                print(f"[OK] Ranking links found: {len(links)}")
                return links[:10]
            pool = set(links)
            for more_url in DISCOVER_CANDIDATES:
                if len(pool) >= 24:
                    break
                extra = get_project_links_playwright(browser, more_url, limit=20, timeout_s=timeout_s)
                for x in extra:
                    pool.add(x)
            if len(pool) >= 10:
                sample = random.sample(list(pool), 10)
                print(f"[OK] Collected {len(pool)} candidates; sampled 10.")
                return sample
    return []

def download_image(url: str, out_path: Path, timeout_s: int) -> Optional[Path]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout_s)
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            f.write(r.content)
        return out_path
    except Exception as e:
        print(f"[WARN] Image download failed {url}: {e}")
        return None

def base64_image_from_path(p: Path) -> Optional[str]:
    try:
        with p.open("rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return None

def scrape_project(url: str, out_dir: Path, download_images: bool, embed_b64: bool,
                   timeout_s: int, translator: GoogleTranslator) -> Dict:
    print(f"  -> scraping project: {url}")
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout_s)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        return {"product_url": url, "error": f"fetch_failed: {e}"}

    soup = BeautifulSoup(html, "lxml")
    title_ja = extract_meta(soup, "og:title") or text_or_none(
        soup.find("h1").get_text(strip=True) if soup.find("h1") else None
    )
    desc_ja = extract_meta(soup, "og:description")
    image_url = extract_meta(soup, "og:image")
    brand = extract_brand_from_page(soup)

    title_zh = translate_ja_to_zh(translator, title_ja)
    desc_zh = translate_ja_to_zh(translator, desc_ja)
    brand_zh = translate_ja_to_zh(translator, brand) if brand else None

    item = {
        "source_site": "Makuake",
        "product_url": url,
        "image_url": image_url,
        "brand_ja": brand,
        "brand_zh_tw": brand_zh,
        "product_name_ja": title_ja,
        "product_name_zh_tw": title_zh,
        "short_description_ja": desc_ja,
        "short_description_zh_tw": desc_zh,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    if image_url and download_images:
        ext = os.path.splitext(image_url.split("?")[0])[1] or ".jpg"
        safe_slug = re.sub(r"[^a-zA-Z0-9_-]", "-", url.strip("/").split("/")[-1])
        img_path = out_dir / "images" / f"{safe_slug}{ext}"
        saved = download_image(image_url, img_path, timeout_s=timeout_s)
        if saved:
            item["local_image_path"] = str(saved)
            if embed_b64:
                b64 = base64_image_from_path(saved)
                if b64:
                    item["image_base64"] = b64
    return item

def main():
    parser = argparse.ArgumentParser(description="Scrape Makuake Top10 (or random 10)")
    parser.add_argument("--out-dir", default="./makuake_out", help="Output directory")
    parser.add_argument("--timeout", type=int, default=25, help="HTTP/Render timeout (seconds)")
    parser.add_argument("--no-download-images", action="store_true", help="Skip image downloads")
    parser.add_argument("--embed-base64", action="store_true", help="Embed base64 image in JSON (larger file)")
    parser.add_argument("--engine", choices=["chromium", "firefox", "webkit"], default="chromium",
                        help="Playwright engine to use (default: chromium)")
    args = parser.parse_args()

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    translator = GoogleTranslator(source="ja", target="zh-TW")

    with sync_playwright() as p:
        # --- launch a browser, then create contexts per page ---
        if args.engine == "chromium":
            browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
        elif args.engine == "firefox":
            browser = p.firefox.launch(headless=True)
        else:
            browser = p.webkit.launch(headless=True)

        try:
            print("[*] Rendering ranking/home/discover with Playwright...")
            project_urls = get_any_10_projects_playwright(browser, timeout_s=args.timeout)
        finally:
            browser.close()

    if not project_urls:
        raise SystemExit(
            "No project links found after Playwright render. "
            "Try --engine firefox, increase --timeout, or run on a non-proxy network."
        )

    # If we got them from ranking in order, include rank
    ranked = True if len(project_urls) == 10 else False
    results = []
    for i, url in enumerate(project_urls, start=1):
        item = scrape_project(
            url,
            out_dir=OUT_DIR,
            download_images=not args.no_download_images,
            embed_b64=args.embed_base64,
            timeout_s=args.timeout,
            translator=translator
        )
        if ranked:
            item["rank"] = i
        results.append(item)

    out_json = {
        "source": {
            "ranking_url": RANKING_URL,
            "homepage_url": HOMEPAGE_URL,
            "method": "ranking_top10" if ranked else "mixed_random10",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "notes": "Playwright browser launched correctly; contexts created per page."
        },
        "items": results
    }

    out_path = OUT_DIR / "makuake_top10.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCCESS] Wrote JSON: {out_path.resolve()}")
    if not args.no_download_images:
        print(f"[INFO] Images saved under: {(OUT_DIR / 'images').resolve()}")

if __name__ == "__main__":
    main()

