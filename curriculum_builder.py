import argparse
import json
import os
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple

import requests

try:
    # Optional: load environment variables (including API_KEY) from a .env file
    # if python-dotenv is installed.
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


def load_graph(crawl_json_path: str) -> Dict[str, Any]:
    with open(crawl_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_adjacency(edges: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if not src or not dst:
            continue
        adj[src].append(dst)
    return adj


def bfs_shortest_paths(root: str, adj: Dict[str, List[str]], max_depth: int) -> Tuple[Dict[str, str | None], Dict[str, int]]:
    parents: Dict[str, str | None] = {root: None}
    depth: Dict[str, int] = {root: 0}
    q: deque[str] = deque([root])

    while q:
        u = q.popleft()
        if depth[u] >= max_depth:
            continue
        for v in adj.get(u, []):
            if v not in parents:
                parents[v] = u
                depth[v] = depth[u] + 1
                q.append(v)

    return parents, depth


def reconstruct_path(goal: str, parents: Dict[str, str | None]) -> List[str]:
    path: List[str] = []
    u: str | None = goal
    while u is not None:
        path.append(u)
        u = parents[u]
    path.reverse()
    return path


def classify_difficulty(path: List[str], nodes: Dict[str, Any]) -> int:
    auth = any(nodes.get(url, {}).get("requires_auth") for url in path)
    L = len(path)
    if auth:
        return 3
    if L == 2:
        return 1
    if 3 <= L <= 4:
        return 2
    return 3


def build_tasks_from_graph(graph: Dict[str, Any], max_depth: int) -> List[Dict[str, Any]]:
    root = graph["root"]
    nodes: Dict[str, Any] = graph["nodes"]
    edges: List[Dict[str, Any]] = graph["edges"]

    adj = build_adjacency(edges)
    parents, depth = bfs_shortest_paths(root, adj, max_depth=max_depth)

    tasks: List[Dict[str, Any]] = []
    task_id = 1

    for url, d in depth.items():
        if url == root or d == 0:
            continue
        path = reconstruct_path(url, parents)
        difficulty = classify_difficulty(path, nodes)
        auth_only_info_used = [u for u in path if nodes.get(u, {}).get("auth_only_info")]
        requires_auth = any(nodes.get(u, {}).get("requires_auth") for u in path)

        # Internal route description (for evaluators / training scaffolding), not
        # exposed as the agent's goal. Keep it abstract where possible.
        if len(path) == 2:
            route_description = "Navigate from the main page to a directly linked page."
        else:
            route_description = "Navigate from the main page through intermediate pages to a specific target page."

        # Initial user_goal is a rough placeholder; Claude will tailor it based on
        # the start/goal URLs and their summaries.
        user_goal = "Ask the assistant to help you achieve what a typical user would want on the goal page."

        tasks.append(
            {
                "id": f"T{task_id:03d}",
                "difficulty": difficulty,
                "start_url": root,
                "goal_url": url,
                "path": path,
                "user_goal": user_goal,
                "route_description": route_description,
                "requires_auth": requires_auth,
                "auth_only_info_used": auth_only_info_used,
            }
        )
        task_id += 1

    return tasks


def refine_task_descriptions_with_claude(tasks: List[Dict[str, Any]], graph: Dict[str, Any]) -> None:
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("API_KEY is not set; skipping Claude refinement.")
        return

    nodes: Dict[str, Any] = graph["nodes"]

    for task in tasks:
        start = task["start_url"]
        goal = task["goal_url"]
        path = task["path"]

        instructions = (
            "You are designing natural-language goals that a human might ask "
            "an AI assistant when using a website.\n\n"
            "I will give you: the start and goal URLs, a start page summary, "
            "optional intermediate page summaries, a goal page summary, and an "
            "internal route description.\n"
            "Use BOTH the goal URL (its path/fragment) and the goal summary to infer "
            "a concrete intent. For example, URLs containing words like 'pricing', "
            "'docs', 'changelog', or a version number should lead to specific goals "
            "such as checking prices, reading documentation, or seeing what's new in "
            "a particular release.\n\n"
            "Write a SINGLE, natural-sounding request the user might say that:\n"
            "- Reflects what they want to accomplish on the goal page,\n"
            "- Does NOT mention raw URLs or specific clicks (no 'go to /changelog'),\n"
            "- Could reasonably be satisfied by navigating from the start to the goal.\n"
            "Examples: 'Show me my account statistics', 'Show me what changed in "
            "Windsurf version 1.6.4', 'Explain what this product does'.\n"
            "Return ONLY that single sentence, no bullet points or numbering."
        )

        content = [
            {"type": "text", "text": instructions},
            {"type": "text", "text": f"Start URL: {start}"},
            {"type": "text", "text": f"Goal URL: {goal}"},
            {"type": "text", "text": f"Start page summary: {nodes.get(start, {}).get('summary', '')}"},
        ]

        for mid in path[1:-1]:
            content.append({"type": "text", "text": f"Intermediate page summary: {nodes.get(mid, {}).get('summary', '')}"})

        content.append({"type": "text", "text": f"Goal page summary: {nodes.get(goal, {}).get('summary', '')}"})
        content.append({"type": "text", "text": f"Internal route description: {task.get('route_description', '')}"})

        payload = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 64,
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
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["content"][0]["text"].strip()
            if text:
                task["user_goal"] = text
        except Exception as exc:
            print(f"Claude refinement failed for {task['id']}: {exc}")
            # Keep the rough goal in this case.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an RL curriculum from crawl_result.json, using the graph "
            "structure and summaries, with optional light Claude refinement."
        )
    )

    parser.add_argument(
        "--crawl-json",
        dest="crawl_json",
        default="crawl_result.json",
        help="Path to crawl_result.json (default: crawl_result.json)",
    )
    parser.add_argument(
        "--max-depth",
        dest="max_depth",
        type=int,
        default=3,
        help="Maximum graph depth from root to consider for tasks (default: 3)",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        default="curriculum.json",
        help="Where to write the generated curriculum JSON (default: curriculum.json)",
    )
    parser.add_argument(
        "--refine-with-claude",
        dest="refine_with_claude",
        action="store_true",
        help="If set, use Claude to refine natural-language goal descriptions.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    graph = load_graph(args.crawl_json)
    tasks = build_tasks_from_graph(graph, max_depth=args.max_depth)

    if args.refine_with_claude:
        refine_task_descriptions_with_claude(tasks, graph)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)

    print(f"Wrote {len(tasks)} tasks to {args.output_path}")


if __name__ == "__main__":
    main()
