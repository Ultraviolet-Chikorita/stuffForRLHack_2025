import argparse
import base64
import json
import os
import time
from typing import Dict, Iterable, Optional, Set
from urllib.parse import urlparse

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from webdriver_manager.chrome import ChromeDriverManager

try:
    # Optional: load API_KEY from a .env file if python-dotenv is installed.
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


def build_driver(window_width: int = 1280, window_height: int = 720) -> webdriver.Chrome:
    """Create a visible (non-headless) Chrome webdriver instance (plain Selenium)."""

    options = ChromeOptions()
    options.add_argument(f"--window-size={window_width},{window_height}")

    driver_path = ChromeDriverManager().install()
    service = ChromeService(driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def is_clickable_like(el: WebElement) -> bool:
    if not el.is_displayed():
        return False

    tag = (el.tag_name or "").lower()
    if tag == "a" and el.get_attribute("href"):
        return True

    if tag in {"button", "input"}:
        return True

    role = (el.get_attribute("role") or "").lower()
    if role in {"button", "link", "menuitem"}:
        return True

    if el.get_attribute("onclick"):
        return True

    return False


def find_clickable_elements(driver: webdriver.Chrome, root: WebElement | None = None) -> Iterable[WebElement]:
    ctx = root if root is not None else driver
    candidates = ctx.find_elements(By.XPATH, "//*[self::a or self::button or self::input or @role='button' or @role='link' or @onclick]")
    seen_ids: Set[int] = set()
    results: list[WebElement] = []

    for el in candidates:
        try:
            if not is_clickable_like(el):
                continue
            el_id = id(el)
            if el_id in seen_ids:
                continue
            seen_ids.add(el_id)
            results.append(el)
        except Exception:
            continue

    return results


def find_modals(driver: webdriver.Chrome) -> Iterable[WebElement]:
    modals = driver.find_elements(
        By.XPATH,
        "//*[contains(@role,'dialog') or contains(@class,'modal') or contains(@class,'popup') or @aria-modal='true']",
    )
    return [m for m in modals if m.is_displayed()]


def hover_and_collect_urls(
    driver: webdriver.Chrome,
    root: Optional[WebElement],
    visited_urls: Set[str],
    base_netloc: str,
    hover_delay: float = 0.3,
) -> None:
    actions = ActionChains(driver)
    elements = list(find_clickable_elements(driver, root))

    for el in elements:
        try:
            actions.move_to_element(el).perform()
            time.sleep(hover_delay)
        except Exception:
            continue

        try:
            href = el.get_attribute("href")
            if not href:
                raise ValueError("no href")

            # Ignore mailto links entirely.
            if href.strip().lower().startswith("mailto:"):
                continue

            parsed = urlparse(href)
            # Drop any URL fragment so we don't treat anchor variants as
            # different targets in the crawl graph or curriculum.
            canonical_href = parsed._replace(fragment="").geturl()

            # Keep only URLs on the same domain (or relative URLs which Selenium may
            # resolve to absolute with the same netloc).
            if parsed.netloc and parsed.netloc != base_netloc:
                continue

            if canonical_href not in visited_urls:
                visited_urls.add(canonical_href)
        except Exception:
            pass

        try:
            # After hover, check for modals/popups and recursively scan inside them.
            for modal in find_modals(driver):
                hover_and_collect_urls(
                    driver,
                    modal,
                    visited_urls,
                    base_netloc=base_netloc,
                    hover_delay=hover_delay,
                )
        except Exception:
            continue


def _create_summary_folder_for_url(url: str) -> str:
    """Create an output folder for scrolling screenshots based on the URL."""

    parsed = urlparse(url)
    parts = [parsed.netloc or "page"]
    if parsed.path and parsed.path != "/":
        parts.append(parsed.path.lstrip("/"))

    raw_name = "_".join(parts)
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in raw_name)

    folder = os.path.join(os.getcwd(), "screenshots", safe_name or "page")
    os.makedirs(folder, exist_ok=True)
    return folder


def _capture_scrolling_screenshots_with_driver(
    driver: webdriver.Chrome,
    output_folder: str,
    scroll_pause: float = 0.7,
max_screenshots: Optional[int] = None,
) -> list[str]:
    """Capture incremental screenshots while scrolling using an existing driver.

    Returns a list of file paths to the saved PNG screenshots.
    """

    # Determine total page height and viewport height.
    total_height = driver.execute_script(
        "return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);"
    )
    viewport_height = driver.execute_script("return window.innerHeight;")

    if not isinstance(total_height, (int, float)) or total_height <= 0:
        total_height = viewport_height
    if not isinstance(viewport_height, (int, float)) or viewport_height <= 0:
        viewport_height = 720

    scroll_position = 0
    screenshot_index = 0
    paths: list[str] = []

    while scroll_position <= total_height:
        if max_screenshots is not None and screenshot_index >= max_screenshots:
            break

        driver.execute_script(f"window.scrollTo(0, {scroll_position});")
        time.sleep(scroll_pause)

        file_path = os.path.join(output_folder, f"{screenshot_index}.png")
        driver.save_screenshot(file_path)
        paths.append(file_path)

        screenshot_index += 1
        scroll_position += viewport_height

    # Final screenshot at the very bottom, in case the loop ended early.
    if max_screenshots is None or screenshot_index < max_screenshots:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause)
        file_path = os.path.join(output_folder, f"{screenshot_index}.png")
        driver.save_screenshot(file_path)
        paths.append(file_path)

    return paths


def summarise_page_purpose(driver: webdriver.Chrome, url: str) -> Optional[str]:
    """Capture scrolling screenshots and ask Claude for a short description.

    Uses the same scrolling screenshot approach as in base.py, then sends all
    screenshots in a single Claude call and asks for a brief textual summary
    (no JSON) of the page's purpose.
    """

    api_key = os.getenv("API_KEY")
    if not api_key:
        print("API_KEY is not set in environment; skipping Claude summary.")
        return None

    output_folder = _create_summary_folder_for_url(url)
    print(f"Saving scrolling screenshots to: {output_folder}")

    try:
        image_paths = _capture_scrolling_screenshots_with_driver(driver, output_folder)
    except Exception as exc:
        print(f"Failed to capture scrolling screenshots: {exc}")
        return None

    if not image_paths:
        print("No screenshots captured; skipping Claude summary.")
        return None

    images_content = []
    for path in image_paths:
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            continue

        images_content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64,
                },
            }
        )

    if not images_content:
        print("Failed to read any screenshots from disk; skipping Claude summary.")
        return None

    instructions = (
        "You are given a sequence of full-page screenshots from a single web page, "
        "ordered from top to bottom. In 2-3 sentences, describe the main purpose of "
        "this page and what a typical visitor can do here. Return ONLY plain text, "
        "no JSON or bullet points."
    )

    payload = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instructions},
                    *images_content,
                ],
            }
        ],
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:  # type: ignore[attr-defined]
        # If the request is too large, fall back to a batched summarisation
        # strategy that sends smaller groups of screenshots and then combines
        # the partial summaries.
        status = getattr(exc.response, "status_code", None)
        if status == 413:
            print("Claude summary request too large (413). Retrying in batches.")
            return _summarise_page_purpose_batched(api_key, image_paths)
        print(f"Error calling Claude for summary: {exc}")
        return None
    except Exception as exc:
        print(f"Error calling Claude for summary: {exc}")
        return None

    data = response.json()
    try:
        return data["content"][0]["text"].strip()
    except Exception as exc:
        print("Unexpected Claude response format for summary:")
        print(data)
        print(f"Error: {exc}")
        return None


def _summarise_page_purpose_batched(api_key: str, image_paths: list[str], batch_size: int = 5) -> Optional[str]:
    """Fallback summarisation that batches screenshots to avoid 413 errors.

    - Sends smaller groups of screenshots to Claude, asking for a short
      partial summary of each group.
    - Then sends a final text-only prompt to Claude to combine those partial
      summaries into a single concise page-purpose description.
    """

    def _chunks(seq: list[str], size: int) -> list[list[str]]:
        return [seq[i : i + size] for i in range(0, len(seq), size)]

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    partial_summaries: list[str] = []

    for batch in _chunks(image_paths, batch_size):
        images_content = []
        for path in batch:
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                continue

            images_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                }
            )

        if not images_content:
            continue

        instructions = (
            "You are given a subset of full-page screenshots from one web page, "
            "ordered from top to bottom. In 1-2 sentences, describe the main "
            "content visible in just these screenshots. Return ONLY plain text."
        )

        payload = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 128,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instructions},
                        *images_content,
                    ],
                }
            ],
        }

        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["content"][0]["text"].strip()
            if text:
                partial_summaries.append(text)
        except Exception as exc:
            print(f"Error calling Claude for batched summary: {exc}")
            continue

    if not partial_summaries:
        return None

    # Now combine the partial summaries into a single concise description.
    combine_instructions = (
        "You are given several short summaries of different vertical sections "
        "of the same web page, from top to bottom. In 2-3 sentences, write a "
        "single overall description of the page's main purpose and what a "
        "typical visitor can do there. Return ONLY plain text."
    )

    combined_text = "\n".join(f"Section {i+1}: {s}" for i, s in enumerate(partial_summaries))

    payload = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": combine_instructions},
                    {"type": "text", "text": combined_text},
                ],
            }
        ],
    }

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"].strip()
    except Exception as exc:
        print(f"Error calling Claude to combine batched summaries: {exc}")
        return None


def detect_auth_requirement(driver: webdriver.Chrome, url: str, screenshots_folder: str) -> bool:
    """Ask Claude (Y/N) if this page requires user authentication details.

    Uses a very short response (1 token) and expects 'Y' or 'N'. Returns True
    if Claude indicates authentication is needed.
    """

    api_key = os.getenv("API_KEY")
    if not api_key:
        print("API_KEY is not set in environment; skipping auth detection.")
        return False

    # Reuse the first screenshot as context if available.
    png_b64: Optional[str] = None
    try:
        files = [f for f in os.listdir(screenshots_folder) if f.lower().endswith(".png")]
        if files:
            first = sorted(files)[0]
            with open(os.path.join(screenshots_folder, first), "rb") as f:
                png_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        png_b64 = None

    instructions = (
        "You will be given a screenshot (if available) and URL of a web page. "
        "Answer with a single character only: 'Y' if a typical user must enter "
        "authentication details (e.g. login, password, MFA) on THIS page to "
        "proceed, or 'N' otherwise. No explanation, just 'Y' or 'N'."
    )

    content = [{"type": "text", "text": instructions + f"\nURL: {url}"}]
    if png_b64 is not None:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": png_b64,
                },
            }
        )

    payload = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 1,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
    except Exception as exc:
        print(f"Error calling Claude for auth detection: {exc}")
        return False

    try:
        data = response.json()
        text = data["content"][0]["text"].strip().upper()
        return text.startswith("Y")
    except Exception as exc:
        print("Unexpected Claude response format for auth detection:")
        print(response.text)
        print(f"Error: {exc}")
        return False


def get_authenticated_info(driver: webdriver.Chrome, url: str, screenshots_folder: str) -> Optional[Dict]:
    """Ask Claude for a JSON object describing info only retrievable when signed in.

    Uses existing screenshots as context and returns a parsed JSON object, or
    None if the call fails.
    """

    api_key = os.getenv("API_KEY")
    if not api_key:
        print("API_KEY is not set in environment; skipping authenticated info call.")
        return None

    # Use the first screenshot in the folder as context, if present.
    png_b64: Optional[str] = None
    try:
        files = [f for f in os.listdir(screenshots_folder) if f.lower().endswith(".png")]
        if files:
            first = sorted(files)[0]
            with open(os.path.join(screenshots_folder, first), "rb") as f:
                png_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        png_b64 = None

    instructions = (
        "You are given context for a web page that is only accessible when a user "
        "is signed in. Return a concise JSON object describing the kinds of data "
        "and actions that are available ONLY to authenticated users on this page.\n\n"
        "The JSON should be of the form:\n"
        "{\n"
        "  \"dataTypes\": [<list of types of personal or account data visible>],\n"
        "  \"keyActions\": [<list of important actions a signed-in user can perform>],\n"
        "  \"notes\": <optional short string with any other important auth-only details>\n"
        "}\n\n"
        "Return ONLY JSON, with double-quoted keys and string values. No prose."
    )

    content = [{"type": "text", "text": instructions + f"\nURL: {url}"}]
    if png_b64 is not None:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": png_b64,
                },
            }
        )

    payload = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
    except Exception as exc:
        print(f"Error calling Claude for authenticated info on {url}: {exc}")
        return None

    def extract_json(text: str) -> str:
        """Extract a JSON substring from Claude output (handles code fences)."""

        text = text.strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            if text.endswith("```"):
                text = text[: -3]
            text = text.strip()

        start_candidates = [idx for idx in (text.find("{"),) if idx != -1]
        end_candidates = [idx for idx in (text.rfind("}"),) if idx != -1]
        if start_candidates and end_candidates:
            start = min(start_candidates)
            end = max(end_candidates) + 1
            return text[start:end]
        return text

    try:
        data = response.json()
        text = data["content"][0]["text"]
        json_text = extract_json(text)
        return json.loads(json_text)
    except Exception as exc:
        print("Failed to parse authenticated info JSON:")
        print(response.text)
        print(f"Error: {exc}")
        return None


def crawl_site(root_url: str, max_depth: int = 2, hover_delay: float = 0.3) -> Dict:
    """Recursively crawl pages starting from root_url, building a directed graph.

    - Uses hover-based URL extraction per page (same-domain, no mailto:).
    - For each unique page, captures scrolling screenshots and gets a Claude summary.
    - Detects whether the page requires authentication via a Y/N Claude call.
    - If auth is required, pauses for manual login before continuing.
    - Captures JSON API responses via selenium-wire and stores real + dummy examples.
    - Tracks nodes and edges in a graph structure.
    """

    driver = build_driver()

    visited: Set[str] = set()
    base_netloc = urlparse(root_url).netloc

    nodes: Dict[str, Dict] = {}
    edges: list[Dict] = []

    # Paths for incremental persistence of the graph and mirror.
    json_path = os.path.join(os.getcwd(), "crawl_result.json")
    mirror_root = os.path.join(os.getcwd(), "site_mirror")

    def persist_graph() -> None:
        """Write the current graph to disk and regenerate the static mirror."""

        graph = {"root": root_url, "nodes": nodes, "edges": edges}
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(graph, f, indent=2)
        except Exception as exc:
            print(f"Failed to write graph JSON: {exc}")

        try:
            # This regenerates all HTML pages based on the current graph state.
            generate_static_html_mirror(graph, output_root=mirror_root)
        except Exception as exc:
            print(f"Failed to generate static HTML mirror: {exc}")

    def add_edge(src: str, dst: str, authenticated_only: bool) -> None:
        # We currently do not persist authenticated_only on edges; we just
        # record connectivity between pages.
        edges.append({"from": src, "to": dst})
        # Persist after each new edge so the graph on disk is always up to date.
        persist_graph()

    def dfs(url: str, depth: int, from_url: Optional[str], inherited_auth: bool) -> None:
        # Treat URLs that differ only by a fragment (e.g. #section) as the same
        # page for crawling, to avoid reloading and re-summarising identical
        # content. We still record edges to the full URL (including fragment),
        # but use a canonical URL (no fragment) for the visited set.
        parsed = urlparse(url)
        canonical_url = parsed._replace(fragment="").geturl()

        if depth > max_depth:
            if from_url is not None:
                add_edge(from_url, url, authenticated_only=inherited_auth)
            return

        if canonical_url in visited:
            if from_url is not None:
                # Edge to an already-visited node
                add_edge(from_url, url, authenticated_only=inherited_auth)
            return

        visited.add(canonical_url)

        if from_url is not None:
            add_edge(from_url, url, authenticated_only=inherited_auth)

        try:
            print(f"\n=== Visiting {url} (depth {depth}) ===")
            # Load the canonical URL without fragment so we don't reload the
            # same page for different anchor links.
            driver.get(canonical_url)
            time.sleep(2.0)
        except Exception as exc:
            print(f"Failed to load {url}: {exc}")
            nodes[url] = {
                "summary": None,
                "urls_on_page": [],
                "requires_auth": False,
                "auth_only_info": None,
            }
            return

        # First: scrolling screenshots + summary for the current URL as loaded.
        summary = summarise_page_purpose(driver, url)

        # Second: auth detection
        screenshots_folder = _create_summary_folder_for_url(url)
        requires_auth = detect_auth_requirement(driver, url, screenshots_folder)

        if requires_auth:
            # Record this URL as an auth-requiring page using the pre-login view.
            nodes[url] = {
                "summary": summary,
                "urls_on_page": [],
                "requires_auth": True,
                "auth_only_info": None,
            }
            persist_graph()

            print("\a")  # bell sound in many terminals
            input(
                "Claude indicates this page likely requires authentication. "
                "Please complete login in the browser (the resulting page may have a new URL), "
                "then press Enter here to continue..."
            )

            # After login, if the browser has navigated to a different URL, treat
            # that new URL as a separate page in the graph and traverse it next.
            # We keep the same logical depth for the post-login page so that it
            # is not pruned when the login page is already at max_depth.
            new_url = driver.current_url
            if new_url and new_url != url:
                # Inherit authenticated context when visiting the post-login page.
                dfs(new_url, depth, from_url=url, inherited_auth=True)

            # Do not overwrite the login page's screenshots or summary, and do not
            # continue collecting URLs for this node.
            return

        # If we are in an authenticated context (either this page or an ancestor
        # required auth), ask Claude for an auth-only info object based on the
        # current page view.
        auth_context = inherited_auth or requires_auth
        auth_info: Optional[Dict] = None
        if auth_context:
            auth_info = get_authenticated_info(driver, url, screenshots_folder)

        # Collect URLs via hover from the current page state.
        page_urls: Set[str] = set()
        hover_and_collect_urls(
            driver,
            None,
            page_urls,
            base_netloc=base_netloc,
            hover_delay=hover_delay,
        )

        nodes[url] = {
            "summary": summary,
            "urls_on_page": sorted(page_urls),
            "requires_auth": requires_auth,
            "auth_only_info": auth_info,
        }

        # Persist graph and mirror after each node update as well.
        persist_graph()

        next_auth_flag = auth_context
        for child in sorted(page_urls):
            dfs(child, depth + 1, from_url=url, inherited_auth=next_auth_flag)

    try:
        dfs(root_url, depth=0, from_url=None, inherited_auth=False)
    finally:
        driver.quit()

    return {"root": root_url, "nodes": nodes, "edges": edges}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively crawl a site starting from a URL, hovering over elements "
            "that might be links (including popups/modals), capturing page "
            "summaries, auth requirements, and API responses into a graph."
        )
    )

    parser.add_argument("url", help="Base URL of the site to analyse")
    parser.add_argument(
        "--hover-delay",
        dest="hover_delay",
        type=float,
        default=0.3,
        help="Seconds to pause after each hover (default: 0.3)",
    )
    parser.add_argument(
        "--max-depth",
        dest="max_depth",
        type=int,
        default=1,
        help="Maximum crawl depth from the root URL (default: 1)",
    )

    return parser.parse_args()


def generate_static_html_mirror(graph: Dict, output_root: str = "site_mirror") -> None:
    """Generate a simple static HTML mirror from the crawl graph.

    - One HTML file per URL (filename is a safe version of the URL).
    - Each page shows its summary and links to neighbors.
    - Pages requiring auth show a simple login form and mark protected edges.
    """

    os.makedirs(output_root, exist_ok=True)

    nodes: Dict[str, Dict] = graph.get("nodes", {})
    edges = graph.get("edges", [])

    # Build adjacency list with auth flags
    outgoing: Dict[str, list[Dict]] = {url: [] for url in nodes.keys()}
    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if src in outgoing:
            outgoing[src].append({
                "to": dst,
                "authenticated_only": edge.get("authenticated_only", False),
            })

    def safe_filename(url: str) -> str:
        parsed = urlparse(url)
        raw = (parsed.netloc or "") + "_" + (parsed.path or "/")
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in raw).strip("_") + ".html"

    for url, info in nodes.items():
        fname = safe_filename(url)
        path = os.path.join(output_root, fname)

        summary = info.get("summary") or "(no summary)"
        requires_auth = bool(info.get("requires_auth"))
        neighbors = outgoing.get(url, [])

        with open(path, "w", encoding="utf-8") as f:
            f.write("<!DOCTYPE html>\n<html><head><meta charset='utf-8'>")
            f.write(f"<title>Mirror: {url}</title></head><body>")
            f.write(f"<h1>Page: {url}</h1>")
            f.write("<h2>Summary</h2>")
            f.write(f"<p>{summary}</p>")

            if requires_auth:
                f.write("<h2>Authentication</h2>")
                f.write(
                    "<p>This page appears to require user authentication. "
                    "The following dummy form represents the login interaction.</p>"
                )
                f.write(
                    "<form><label>Username <input type='text' name='username'></label><br>"
                    "<label>Password <input type='password' name='password'></label><br>"
                    "<button type='submit'>Log In</button></form>"
                )

            f.write("<h2>Outgoing Links</h2><ul>")
            for edge in neighbors:
                dst = edge["to"]
                if dst not in nodes:
                    continue
                dst_fname = safe_filename(dst)
                label = dst
                if edge.get("authenticated_only"):
                    label += " (requires auth)"
                f.write(f"<li><a href='{dst_fname}'>{label}</a></li>")
            f.write("</ul>")

            f.write("</body></html>")


def main() -> None:
    args = parse_args()

    graph = crawl_site(args.url, max_depth=args.max_depth, hover_delay=args.hover_delay)

    # Export graph JSON
    json_path = os.path.join(os.getcwd(), "crawl_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
    print(f"Saved crawl graph JSON to {json_path}")

    # Generate static HTML mirror
    mirror_root = os.path.join(os.getcwd(), "site_mirror")
    generate_static_html_mirror(graph, output_root=mirror_root)
    print(f"Generated static HTML mirror under {mirror_root}")


if __name__ == "__main__":
    main()
