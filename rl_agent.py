import argparse
import json
import os
import random
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
from hover_crawler import build_driver
from selenium.webdriver.chrome.webdriver import WebDriver


# -----------------------------
# HTML parsing over site_mirror
# -----------------------------


class MirrorPageParser(HTMLParser):
    """Parse a single mirror HTML file to get its canonical URL and outgoing hrefs."""

    def __init__(self) -> None:
        super().__init__()
        self.in_h1 = False
        self.in_title_h1 = False
        self.url: Optional[str] = None
        self.hrefs: List[str] = []

    def handle_starttag(self, tag: str, attrs):  # type: ignore[override]
        if tag.lower() == "h1":
            # We look for <h1>Page: {url}</h1>
            self.in_h1 = True
        if tag.lower() == "a":
            href = None
            for k, v in attrs:
                if k.lower() == "href":
                    href = v
                    break
            if href:
                self.hrefs.append(href)

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag.lower() == "h1":
            self.in_h1 = False

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self.in_h1 and data.strip().startswith("Page:"):
            # Expect format: "Page: {url}"
            rest = data.split("Page:", 1)[1].strip()
            self.url = rest


def parse_mirror_directory(mirror_root: str) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str]]:
    """Build adjacency from site_mirror HTML.

    Returns:
      - adj: url -> list[url] outgoing
      - file_to_url: filename -> url (just for debugging/inspection)
    """

    # First pass: map each file to its canonical URL from <h1>Page: ...</h1>
    file_to_url: Dict[str, str] = {}
    for name in os.listdir(mirror_root):
        if not name.lower().endswith(".html"):
            continue
        path = os.path.join(mirror_root, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        parser = MirrorPageParser()
        try:
            parser.feed(content)
        except Exception:
            continue

        if parser.url:
            file_to_url[name] = parser.url

    # Second pass: build adjacency using href filenames and the mapping
    adj: Dict[str, List[str]] = {}
    for name in os.listdir(mirror_root):
        if not name.lower().endswith(".html"):
            continue
        path = os.path.join(mirror_root, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        parser = MirrorPageParser()
        try:
            parser.feed(content)
        except Exception:
            continue

        src_url = parser.url
        if not src_url:
            continue

        neighbors: List[str] = []
        for href in parser.hrefs:
            # Hrefs in mirror are filenames like other_page.html
            fname = os.path.basename(href)
            dst_url = file_to_url.get(fname)
            if dst_url:
                neighbors.append(dst_url)

        # Deduplicate while preserving order
        seen = set()
        uniq_neighbors: List[str] = []
        for u in neighbors:
            if u not in seen:
                seen.add(u)
                uniq_neighbors.append(u)

        adj[src_url] = uniq_neighbors

    # Also build inverse mapping: url -> filename (for loading via Selenium)
    url_to_file: Dict[str, str] = {url: fname for fname, url in file_to_url.items()}

    return adj, file_to_url, url_to_file


# -----------------------------
# Curriculum loading
# -----------------------------


def load_curriculum(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def remap_task_goals_via_mirror_text(
    tasks: List[Dict], mirror_root: str, file_to_url: Dict[str, str], known_urls: Dict[str, List[str]]
) -> None:
    """For any task whose goal_url is not a node in the mirror graph, try to
    map it to the canonical URL of a page whose HTML contains that goal_url
    string.

    This is useful when curriculum goal URLs are textual representations shown
    on pages rather than the exact crawled routes.
    """

    # Build a cache of file contents so we only read each once.
    content_cache: Dict[str, str] = {}

    def file_contains(path: str, needle: str) -> bool:
        if path not in content_cache:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content_cache[path] = f.read()
            except Exception:
                content_cache[path] = ""
        return needle in content_cache[path]

    for task in tasks:
        goal = task.get("goal_url")
        if not isinstance(goal, str):
            continue
        if goal in known_urls:
            # Already a known node; nothing to do.
            continue

        # Search for a page whose HTML contains this goal string.
        mapped_url: Optional[str] = None
        for fname, page_url in file_to_url.items():
            path = os.path.join(mirror_root, fname)
            if file_contains(path, goal):
                mapped_url = page_url
                break

        if mapped_url is not None:
            task["goal_url"] = mapped_url


# -----------------------------
# RL environment
# -----------------------------


@dataclass
class StepResult:
    state: str
    reward: float
    done: bool


class WebsiteNavEnv:
    """Simple tabular navigation environment over the mirror graph.

    - State: canonical page URL (string)
    - Action: integer index into outgoing neighbors list for current state
    - Reward: step_penalty each step; +goal_reward when reaching goal_url
    """

    def __init__(
        self,
        adj: Dict[str, List[str]],
        url_to_file: Dict[str, str],
        mirror_root: str,
        driver: WebDriver,
        max_steps: int = 10,
        step_penalty: float = -0.01,
        goal_reward: float = 1.0,
    ) -> None:
        self.adj = adj
        self.url_to_file = url_to_file
        self.mirror_root = mirror_root
        self.driver = driver
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward

        # Ensure every URL appears as a state key
        for src, nbrs in list(adj.items()):
            for dst in nbrs:
                if dst not in self.adj:
                    self.adj[dst] = []

        self._state: Optional[str] = None
        self._goal: Optional[str] = None
        self._steps: int = 0
        self._visited: Set[str] = set()

    def _load_state_in_browser(self, url: str) -> None:
        """Load the mirror HTML for the given URL in the Selenium driver."""
        fname = self.url_to_file.get(url)
        if not fname:
            # If we do not have a file mapping, do nothing; this is a safety net.
            return
        path = os.path.join(self.mirror_root, fname)
        if not os.path.isfile(path):
            return
        file_url = f"file://{os.path.abspath(path)}"
        try:
            self.driver.get(file_url)
        except Exception:
            # Ignore navigation errors during training.
            pass

    def reset(self, start_url: str, goal_url: str) -> str:
        if start_url not in self.adj:
            raise ValueError(f"Unknown start_url in mirror graph: {start_url}")
        if goal_url not in self.adj:
            raise ValueError(f"Unknown goal_url in mirror graph: {goal_url}")
        self._state = start_url
        self._goal = goal_url
        self._steps = 0
        self._visited = {start_url}
        self._load_state_in_browser(start_url)
        return self._state

    def actions(self, state: Optional[str] = None) -> List[str]:
        url = state if state is not None else self._state
        if url is None:
            return []
        return self.adj.get(url, [])

    def step(self, action_index: int) -> StepResult:
        if self._state is None or self._goal is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")

        neighbors = self.adj.get(self._state, [])
        if not neighbors:
            # Dead end: stay put, incur penalty and treat as failure when max_steps hit.
            self._steps += 1
            reward = self.step_penalty
            done = self._steps >= self.max_steps
            if done and self._state != self._goal:
                # Extra penalty for timing out without success.
                reward += -0.5
            return StepResult(state=self._state, reward=reward, done=done)

        if not (0 <= action_index < len(neighbors)):
            raise ValueError(f"Invalid action index {action_index} for state {self._state}")

        next_state = neighbors[action_index]
        self._state = next_state
        self._steps += 1

        # Interact with the environment by loading the next page in Selenium.
        self._load_state_in_browser(next_state)

        reward = self.step_penalty
        done = False

        # Penalize revisiting states in the same episode to discourage loops.
        if next_state in self._visited:
            reward += -0.05
        else:
            self._visited.add(next_state)

        if next_state == self._goal:
            # Successful navigation: strong positive signal.
            reward += self.goal_reward
            done = True
        elif self._steps >= self.max_steps:
            # Episode timed out without reaching the goal: extra penalty.
            reward += -2.0
            done = True
        return StepResult(state=next_state, reward=reward, done=done)


# -----------------------------
# Q-learning agent
# -----------------------------


class QLearningAgent:
    def __init__(
        self,
        env: WebsiteNavEnv,
        learning_rate: float = 0.3,
        discount: float = 0.95,
        epsilon_start: float = 0.2,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 10_000,
    ) -> None:
        self.env = env
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0
        # Q-table: (state_url, action_index) -> value
        self.q: Dict[Tuple[str, int], float] = {}

    def _epsilon(self) -> float:
        t = min(self.total_steps, self.epsilon_decay_steps)
        frac = t / float(self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _get_q(self, state: str, action_index: int) -> float:
        return self.q.get((state, action_index), 0.0)

    def _set_q(self, state: str, action_index: int, value: float) -> None:
        self.q[(state, action_index)] = value

    def _best_action(self, state: str) -> Tuple[int, float]:
        neighbors = self.env.actions(state)
        if not neighbors:
            return 0, 0.0
        best_idx = 0
        best_q = self._get_q(state, 0)
        for i in range(1, len(neighbors)):
            val = self._get_q(state, i)
            if val > best_q:
                best_q = val
                best_idx = i
        return best_idx, best_q

    def select_action(self, state: str) -> int:
        neighbors = self.env.actions(state)
        if not neighbors:
            return 0
        eps = self._epsilon()
        if random.random() < eps:
            return random.randrange(len(neighbors))
        idx, _ = self._best_action(state)
        return idx

    def update(self, state: str, action_index: int, reward: float, next_state: str, done: bool) -> None:
        old_q = self._get_q(state, action_index)
        if done:
            target = reward
        else:
            _, next_best = self._best_action(next_state)
            target = reward + self.gamma * next_best
        new_q = old_q + self.lr * (target - old_q)
        self._set_q(state, action_index, new_q)
        self.total_steps += 1


# -----------------------------
# Training and evaluation
# -----------------------------


def train_agent(
    env: WebsiteNavEnv,
    agent: QLearningAgent,
    tasks: List[Dict],
    episodes: int = 2000,
) -> List[float]:
    if not tasks:
        raise ValueError("No tasks found in curriculum.")

    episode_rewards: List[float] = []

    for ep in range(episodes):
        task = random.choice(tasks)
        start = task["start_url"]
        goal = task["goal_url"]
        state = env.reset(start, goal)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action_idx = agent.select_action(state)
            result = env.step(action_idx)
            agent.update(state, action_idx, result.reward, result.state, result.done)
            state = result.state
            done = result.done
            total_reward += result.reward
            steps += 1

        episode_rewards.append(total_reward)

        if (ep + 1) % 200 == 0:
            print(f"Episode {ep + 1}/{episodes}: reward={total_reward:.3f}, steps={steps}")

    return episode_rewards


def find_task_for_goal(tasks: List[Dict], goal_substring: str) -> Optional[Dict]:
    goal_substring = goal_substring.lower()
    for t in tasks:
        if goal_substring in t.get("user_goal", "").lower():
            return t
    return None


def run_policy_for_task(env: WebsiteNavEnv, agent: QLearningAgent, task: Dict) -> List[str]:
    start = task["start_url"]
    goal = task["goal_url"]
    path: List[str] = []
    state = env.reset(start, goal)
    path.append(state)
    done = False

    while not done:
        neighbors = env.actions(state)
        if not neighbors:
            break
        # Greedy at evaluation time
        idx, _ = agent._best_action(state)
        result = env.step(idx)
        state = result.state
        path.append(state)
        done = result.done

    return path


# -----------------------------
# CLI
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple RL navigation agent over the site_mirror.")
    parser.add_argument(
        "--mirror-root",
        default=os.path.join(os.getcwd(), "site_mirror"),
        help="Path to the site_mirror directory (default: ./site_mirror)",
    )
    parser.add_argument(
        "--curriculum",
        default="curriculum_tailored.json",
        help="Path to curriculum_tailored.json (default: curriculum_tailored.json)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2000,
        help="Number of training episodes (default: 2000)",
    )
    parser.add_argument(
        "--demo-goal",
        type=str,
        default="",
        help="Substring to match in user_goal to run a demo episode after training.",
    )
    args = parser.parse_args()

    print(f"Parsing mirror from: {args.mirror_root}")
    adj, file_to_url, url_to_file = parse_mirror_directory(args.mirror_root)
    print(f"Loaded {len(adj)} pages from mirror.")

    tasks = load_curriculum(args.curriculum)
    print(f"Loaded {len(tasks)} curriculum tasks from {args.curriculum}.")

    # Some curriculum goal_url fields may be textual URLs shown on pages rather than
    # exact crawled routes. Attempt to remap those by searching the mirror HTML.
    remap_task_goals_via_mirror_text(tasks, args.mirror_root, file_to_url, adj)

    # After remapping, keep only tasks whose start/goal URLs exist in the mirror graph.
    filtered_tasks = [
        t
        for t in tasks
        if t.get("start_url") in adj and t.get("goal_url") in adj
    ]
    if not filtered_tasks:
        raise ValueError("No curriculum tasks have both start_url and goal_url in the mirror graph after remapping.")
    print(f"Using {len(filtered_tasks)} tasks that are present in the mirror graph after remapping.")

    driver = build_driver()
    try:
        env = WebsiteNavEnv(adj, url_to_file, args.mirror_root, driver)
        agent = QLearningAgent(env)

        print("Training agent...")
        episode_rewards = train_agent(env, agent, filtered_tasks, episodes=args.episodes)

        # Plot reward vs. episode
        try:
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)
            plt.xlabel("Episode")
            plt.ylabel("Total reward")
            plt.title("Navigation agent training reward over episodes")
            plt.tight_layout()
            out_path = os.path.join(os.getcwd(), "reward_curve.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Saved reward curve to {out_path}")
        except Exception as exc:
            print(f"Failed to plot reward curve: {exc}")

        if args.demo_goal:
            task = find_task_for_goal(filtered_tasks, args.demo_goal)
            if not task:
                print(f"No task found matching goal substring: {args.demo_goal!r}")
                return
            print("\nDemo task:")
            print(f"  id: {task['id']}")
            print(f"  user_goal: {task['user_goal']}")
            print(f"  start_url: {task['start_url']}")
            print(f"  goal_url: {task['goal_url']}")

            path = run_policy_for_task(env, agent, task)
            print("\nTraversed path:")
            for url in path:
                print("  ", url)
    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
