import argparse
import base64
import json
import os
import time
from datetime import datetime
from pprint import pprint
from urllib.parse import urlparse

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

try:
    # Optional: load API_KEY from a .env file if python-dotenv is installed.
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


def create_output_folder(base_folder: str | None, url: str) -> str:
    """Create (or reuse) an output folder for screenshots.

    If base_folder is None/empty, a folder is created under ./screenshots based on the URL.
    Example: https://example.com/path -> ./screenshots/example.com_path
    """

    if base_folder is None or base_folder.strip() == "":
        parsed = urlparse(url)
        # Build a filesystem-friendly name from netloc + path
        parts = [parsed.netloc or "page"]
        if parsed.path and parsed.path != "/":
            parts.append(parsed.path.lstrip("/"))

        raw_name = "_".join(parts)
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in raw_name)

        base_folder = os.path.join(os.getcwd(), "screenshots", safe_name or "page")

    os.makedirs(base_folder, exist_ok=True)
    return base_folder


def build_driver(window_width: int = 1280, window_height: int = 720) -> webdriver.Chrome:
    """Create a visible (non-headless) Chrome webdriver instance."""

    options = ChromeOptions()
    # Do NOT set headless: we want the real window so a future pydirectinput
    # controller can interact with it.
    options.add_argument(f"--window-size={window_width},{window_height}")

    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    return driver


def capture_scrolling_screenshots(
    url: str,
    output_folder: str,
    scroll_pause: float = 0.7,
    max_screenshots: int | None = None,
) -> None:
    """Open the URL and capture incremental screenshots while scrolling.

    - url: page to open
    - output_folder: where to store PNG files
    - scroll_pause: seconds to wait after each scroll before screenshot
    - max_screenshots: optional hard cap on number of screenshots
    """

    driver = build_driver()
    try:
        driver.get(url)

        # Small wait so the page can finish initial layout / JS.
        time.sleep(2.0)

        # Determine total page height and viewport height.
        total_height = driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);")
        viewport_height = driver.execute_script("return window.innerHeight;")

        # Fallbacks if JS returns something unexpected.
        if not isinstance(total_height, (int, float)) or total_height <= 0:
            total_height = viewport_height

        if not isinstance(viewport_height, (int, float)) or viewport_height <= 0:
            viewport_height = 720

        scroll_position = 0
        screenshot_index = 0

        while scroll_position <= total_height:
            if max_screenshots is not None and screenshot_index >= max_screenshots:
                break

            # Scroll and wait for content to render/stabilize.
            driver.execute_script(f"window.scrollTo(0, {scroll_position});")
            time.sleep(scroll_pause)

            file_path = os.path.join(output_folder, f"{screenshot_index}.png")
            driver.save_screenshot(file_path)

            screenshot_index += 1
            scroll_position += viewport_height

        # Final screenshot at the very bottom, in case the loop ended early.
        if max_screenshots is None or screenshot_index < max_screenshots:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause)
            file_path = os.path.join(output_folder, f"{screenshot_index}.png")
            driver.save_screenshot(file_path)

    finally:
        driver.quit()


def call_claude_with_screenshots(folder: str, batch_size: int = 5) -> None:
    """Send PNG screenshots in a folder to Claude in multiple batches.

    Each batch processes up to `batch_size` images. The JSON outputs from all
    batches are combined into a single list and pretty-printed.

    Expected JSON schema (per element in the list):
    {
        index: number,          # index of the screenshot file
        summary: string,        # main content summary (ignore header/footer, max one sentence)
        thingsToClick: [        # array of clickable element bounding boxes
            { left, top, right, bottom }
        ],
    }
    """

    api_key = os.getenv("API_KEY")
    if not api_key:
        print("API_KEY is not set in environment; skipping Claude call.")
        return

    files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    if not files:
        print("No PNG screenshots found; nothing to send to Claude.")
        return

    # Sort files numerically based on filename (without extension) when possible.
    def sort_key(name: str):
        stem, _ = os.path.splitext(name)
        try:
            return int(stem)
        except ValueError:
            return stem

    files.sort(key=sort_key)

    # Helper shared by all batches
    def extract_json(text: str) -> str:
        """Extract a JSON substring from Claude text output.

        Handles cases where the model wraps content in ```json code fences or
        adds minor prose around the JSON.
        """

        text = text.strip()

        # Strip Markdown code fences like ```json ... ``` if present.
        if text.startswith("```"):
            # Remove leading fence line
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            if text.endswith("```"):
                text = text[: -3]
            text = text.strip()

        # Heuristic: take from first '[' or '{' to last ']' or '}'
        start_candidates = [idx for idx in (text.find("["), text.find("{")) if idx != -1]
        end_candidates = [idx for idx in (text.rfind("]"), text.rfind("}")) if idx != -1]

        if start_candidates and end_candidates:
            start = min(start_candidates)
            end = max(end_candidates) + 1
            return text[start:end]
        return text

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    all_results: list[dict] = []

    # Process images in batches
    for batch_start in range(0, len(files), batch_size):
        batch_files = files[batch_start : batch_start + batch_size]

        images_content = []
        for name in batch_files:
            path = os.path.join(folder, name)
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
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

        # Let Claude know exactly which indices are in this batch.
        batch_indices = []
        for name in batch_files:
            stem, _ = os.path.splitext(name)
            try:
                idx = int(stem)
            except ValueError:
                continue
            batch_indices.append(idx)

        indices_str = ", ".join(str(i) for i in batch_indices)

        instructions = (
            "You are given a batch of full-page screenshots from a website, "
            "ordered from top to bottom. This batch contains screenshots with "
            f"indices: [{indices_str}]. Return a single JSON array where each "
            "element has the form:\n\n"
            "[\n"
            "  {\n"
            "    \"index\": <number>,\n"
            "    \"summary\": <string>,\n"
            "    \"thingsToClick\": [\n"
            "      { \"left\": <number>, \"top\": <number>, \"right\": <number>, \"bottom\": <number> }\n"
            "    ]\n"
            "  }\n"
            "]\n\n"
            "Where:\n"
            "- index: the numeric index of the screenshot (matching the indices listed above).\n"
            "- summary: ONE sentence summarising the main page content in that screenshot, "
            "  ignoring header and footer, prioritising larger text.\n"
            "- thingsToClick: an array (max 5 items per screenshot) of bounding boxes for "
            "  elements that could reasonably be clickable (buttons, links, cards, etc.), "
            "  each with left/top/right/bottom coordinates in pixels relative to the "
            "  screenshot image.\n\n"
            "The JSON must be compact enough to avoid truncation. Do NOT include "
            "Markdown code fences or any commentary. Return ONLY valid JSON."
        )

        payload = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 4096,
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
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
        except Exception as exc:
            print(f"Error calling Claude API for batch starting at {batch_start}: {exc}")
            continue

        data = response.json()

        try:
            text = data["content"][0]["text"]
            json_text = extract_json(text)
            batch_parsed = json.loads(json_text)
        except Exception as exc:
            print(f"Failed to parse Claude response as JSON for batch starting at {batch_start}:")
            print(data)
            print(f"Error: {exc}")
            continue

        if isinstance(batch_parsed, list):
            all_results.extend(batch_parsed)
        else:
            all_results.append(batch_parsed)

    print("\n=== Combined Claude JSON Analysis ===")
    pprint(all_results, sort_dicts=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open a URL in a visible browser window and capture incremental "
            "screenshots while scrolling down the page."
        )
    )

    parser.add_argument("url", help="Base URL of the page to capture")
    parser.add_argument(
        "--out",
        dest="output_folder",
        default=None,
        help="Folder to save screenshots to (default: ./screenshots/<timestamp>)",
    )
    parser.add_argument(
        "--pause",
        dest="scroll_pause",
        type=float,
        default=0.7,
        help="Seconds to wait after each scroll before taking a screenshot (default: 0.7)",
    )
    parser.add_argument(
        "--max-shots",
        dest="max_screenshots",
        type=int,
        default=None,
        help="Optional maximum number of screenshots to capture",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_folder = create_output_folder(args.output_folder, args.url)

    capture_scrolling_screenshots(
        url=args.url,
        output_folder=out_folder,
        scroll_pause=args.scroll_pause,
        max_screenshots=args.max_screenshots,
    )

    # After screenshots are captured, analyse them with Claude and pretty-print the JSON result.
    call_claude_with_screenshots(out_folder)


if __name__ == "__main__":
    main()

