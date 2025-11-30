"""Train an RL navigation agent over the Windsurf site mirror.

This module wires together several pieces:

- Parses the static HTML mirror under `site_mirror/` into a graph.
- Loads a curriculum of navigation tasks (start_url, goal_url, demo path).
- Defines a tabular Q-learning environment with optional path-based shaping.
- Runs Optuna to tune hyperparameters for the D_potential shaping mode.
- Retrains the best configuration and materializes plots / logs for analysis.

The design goal is to make it easy to swap in a different curriculum or
shaping strategy without touching the Optuna wiring, and to keep all
"experiment diary" artifacts (reward curves, success rates, Q-tables)
under a single output directory.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import optuna
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
    """Normalize curriculum goal URLs against the mirrored graph.

    The curriculum sometimes stores `goal_url` as a literal string that
    appears on the page (e.g. a full route including query/fragment), while
    the crawl has already canonicalized URLs. Here we:

    - Leave goal_url unchanged if it is already a known graph node.
    - Otherwise, treat it as a search key and map to the first mirror HTML
      file whose contents contain that string.

    This lets us reuse the authored curriculum without regenerating it from
    scratch every time the crawl changes slightly.
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
    """Tabular navigation environment over the mirror graph.

    - State: canonical page URL (string).
    - Action: integer index into the adjacency list for the current state.
    - Base reward: small step penalty, plus a terminal spike on success.

    Path-based shaping is layered on top of this using the curriculum demo
    path per task (which you can think of as "Devin's expert trace"):

    - "A": forgiving progressive reward as we advance along the demo path.
    - "B": strict sequence reward; one deviation disables further shaping.
    - "C": dense shaping via prefix-match similarity to the demo path.
    - "D": potential-based shaping from demo-path distance to goal.

    The intent is that you can enable/disable shaping modes without touching
    the core transition or Selenium logic.
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
        path_shaping_mode: str = "A",
        path_reward: float = 0.05,
        potential_gamma: float = 0.95,
    ) -> None:
        self.adj = adj
        self.url_to_file = url_to_file
        self.mirror_root = mirror_root
        self.driver = driver
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward

        # Path shaping configuration
        self.path_shaping_mode = path_shaping_mode  # "A", "B", "C", or "D"
        self.path_reward = path_reward
        self.potential_gamma = potential_gamma

        # Ensure every URL appears as a state key
        for src, nbrs in list(adj.items()):
            for dst in nbrs:
                if dst not in self.adj:
                    self.adj[dst] = []

        # Episode state
        self._state: Optional[str] = None
        self._goal: Optional[str] = None
        self._steps: int = 0
        self._visited: Set[str] = set()

        # Demo path shaping state (per episode)
        self._demo_path: Optional[List[str]] = None
        self._demo_index_by_url: Dict[str, int] = {}
        self._demo_progress: int = 0  # used by modes A/B
        self._off_demo_path: bool = False  # used by mode B
        self._trajectory: List[str] = []  # used by mode C
        self._similarity_progress: int = 0  # used by mode C

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

    def _reset_demo_path_state(self, start_url: str, goal_url: str, demo_path: Optional[List[str]]) -> None:
        """Initialize per-episode path-shaping state from the curriculum demo path."""
        if demo_path and isinstance(demo_path, list) and len(demo_path) > 0:
            self._demo_path = demo_path
            self._demo_index_by_url = {u: i for i, u in enumerate(demo_path)}
        else:
            self._demo_path = None
            self._demo_index_by_url = {}

        self._demo_progress = 0
        self._off_demo_path = False
        self._trajectory = [start_url]
        self._similarity_progress = 0

        # Initialize progress if start_url is on the path
        if self._demo_path:
            if start_url in self._demo_index_by_url:
                self._demo_progress = self._demo_index_by_url[start_url]
                # For similarity shaping, prefix match length starts at 1 if prefix[0] == start_url
                if self._demo_path[0] == start_url:
                    self._similarity_progress = 1
                else:
                    self._similarity_progress = 0
            else:
                self._demo_progress = 0
                self._similarity_progress = 0

    def _potential(self, state: str) -> float:
        """Potential function Φ(s) for potential-based shaping (mode D).

        Φ(s) = -distance along demo_path from s to goal.
        If s or goal not on the demo path, potential is 0.
        """
        if (
            not self._demo_path
            or state not in self._demo_index_by_url
            or self._goal not in self._demo_index_by_url
        ):
            return 0.0
        idx = self._demo_index_by_url[state]
        goal_idx = self._demo_index_by_url[self._goal]  # type: ignore[arg-type]
        distance = max(0, goal_idx - idx)
        return -float(distance)

    def _compute_prefix_match_length(self) -> int:
        """Length of the common prefix between trajectory and demo_path (for mode C)."""
        if not self._demo_path:
            return 0
        max_k = min(len(self._trajectory), len(self._demo_path))
        k = 0
        for i in range(max_k):
            if self._trajectory[i] == self._demo_path[i]:
                k += 1
            else:
                break
        return k

    def reset(self, start_url: str, goal_url: str, demo_path: Optional[List[str]] = None) -> str:
        if start_url not in self.adj:
            raise ValueError(f"Unknown start_url in mirror graph: {start_url}")
        if goal_url not in self.adj:
            raise ValueError(f"Unknown goal_url in mirror graph: {goal_url}")
        self._state = start_url
        self._goal = goal_url
        self._steps = 0
        self._visited = {start_url}
        self._reset_demo_path_state(start_url, goal_url, demo_path)
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
            # Dead end: stay put, small penalty per step, episode ends only on timeout.
            self._steps += 1
            reward = self.step_penalty  # e.g. -0.01
            done = self._steps >= self.max_steps
            return StepResult(state=self._state, reward=reward, done=done)

        if not (0 <= action_index < len(neighbors)):
            raise ValueError(f"Invalid action index {action_index} for state {self._state}")

        prev_state = self._state
        next_state = neighbors[action_index]

        # Potential at current state (for mode D)
        phi_prev = 0.0
        if self.path_shaping_mode == "D":
            phi_prev = self._potential(prev_state)

        # Transition
        self._state = next_state
        self._steps += 1

        # Interact with the environment by loading the next page in Selenium.
        self._load_state_in_browser(next_state)

        # Base per-step penalty
        reward = self.step_penalty  # e.g. -0.01
        done = False

        # Penalize revisiting states in the same episode to discourage loops.
        if next_state in self._visited:
            reward += -0.04  # total ≈ -0.05 on revisits
        else:
            self._visited.add(next_state)

        # Path-based shaping (modes A, B, C)
        if self._demo_path:
            # Mode A: forgiving progressive reward along demo path
            if self.path_shaping_mode == "A":
                if next_state in self._demo_index_by_url:
                    idx = self._demo_index_by_url[next_state]
                    # Reward only when we advance to the next unseen path step
                    if idx == self._demo_progress + 1:
                        reward += self.path_reward
                        self._demo_progress = idx

            # Mode B: strict sequence along demo path; one deviation disables shaping
            elif self.path_shaping_mode == "B":
                if not self._off_demo_path:
                    cur_idx = self._demo_index_by_url.get(prev_state)
                    next_idx = self._demo_index_by_url.get(next_state)
                    if cur_idx is None or next_idx is None or next_idx != cur_idx + 1:
                        # Once we deviate, we stay "off path" for this episode
                        self._off_demo_path = True
                    else:
                        reward += self.path_reward
                        self._demo_progress = next_idx

            # Mode C: dense similarity shaping via prefix match with demo path
            elif self.path_shaping_mode == "C":
                # Update trajectory and reward improvements in prefix match length
                self._trajectory.append(next_state)
                new_k = self._compute_prefix_match_length()
                if new_k > self._similarity_progress:
                    reward += self.path_reward * (new_k - self._similarity_progress)
                    self._similarity_progress = new_k

        # Mode D: potential-based shaping
        if self.path_shaping_mode == "D" and self._demo_path:
            phi_next = self._potential(next_state)
            reward += self.potential_gamma * phi_next - phi_prev

        # Check terminal conditions
        if next_state == self._goal:
            # On success, override with a strong positive signal.
            reward = self.goal_reward  # e.g. +1.0
            done = True
        elif self._steps >= self.max_steps:
            # Episode timed out without reaching the goal: no massive extra penalty.
            done = True

        # If we didn't append in mode C (or D), still keep trajectory updated
        if self.path_shaping_mode != "C":
            self._trajectory.append(next_state)

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
) -> Tuple[List[float], List[float]]:
    if not tasks:
        raise ValueError("No tasks found in curriculum.")

    episode_rewards: List[float] = []
    success_rates: List[float] = []
    successes = 0

    for ep in range(episodes):
        task = random.choice(tasks)
        start = task["start_url"]
        goal = task["goal_url"]
        demo_path = task.get("path")
        state = env.reset(start, goal, demo_path=demo_path)
        done = False
        total_reward = 0.0
        steps = 0
        reached_goal = False

        while not done:
            action_idx = agent.select_action(state)
            result = env.step(action_idx)
            agent.update(state, action_idx, result.reward, result.state, result.done)
            state = result.state
            done = result.done
            total_reward += result.reward
            steps += 1

            if done and state == goal:
                reached_goal = True

        if reached_goal:
            successes += 1
        success_rate = successes / float(ep + 1)
        episode_rewards.append(total_reward)
        success_rates.append(success_rate)

        if (ep + 1) % 200 == 0:
            print(
                f"Episode {ep + 1}/{episodes}: "
                f"reward={total_reward:.3f}, steps={steps}, success_rate={success_rate:.3f}"
            )

    return episode_rewards, success_rates


def find_task_for_goal(tasks: List[Dict], goal_substring: str) -> Optional[Dict]:
    goal_substring = goal_substring.lower()
    for t in tasks:
        if goal_substring in t.get("user_goal", "").lower():
            return t
    return None


def run_policy_for_task(env: WebsiteNavEnv, agent: QLearningAgent, task: Dict) -> List[str]:
    start = task["start_url"]
    goal = task["goal_url"]
    demo_path = task.get("path")
    path: List[str] = []
    state = env.reset(start, goal, demo_path=demo_path)
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
# Optuna objective for hyperparameter tuning (D_potential only)
# -----------------------------


def objective(
    trial: optuna.trial.Trial,
    adj: Dict[str, List[str]],
    url_to_file: Dict[str, str],
    mirror_root: str,
    driver: WebDriver,
    tasks: List[Dict],
    episodes: int,
    demo_task: Optional[Dict],
    out_dir: str,
) -> float:
    """Optuna objective for D_potential shaping.

    We treat the environment + agent hyperparameters as a search space and
    optimize for the mean success rate over the tail of training episodes.

    Rationale:
    - Early reward spikes are common in RL and often misleading.
    - Using the final 10% of episodes as the metric biases the search toward
      configurations that actually converge to a stable, successful policy.
    """

    # Env hyperparameters
    max_steps = trial.suggest_int("max_steps", 6, 12)
    step_penalty = -trial.suggest_float("step_penalty_mag", 0.001, 0.02, log=True)
    goal_reward = trial.suggest_float("goal_reward", 0.5, 3.0)
    path_reward = trial.suggest_float("path_reward", 0.01, 0.1)
    potential_gamma = trial.suggest_float("potential_gamma", 0.85, 0.99)

    # Agent hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 0.1, 0.5)
    discount = trial.suggest_float("discount", 0.9, 0.999)
    epsilon_start = 0.2
    epsilon_end = trial.suggest_float("epsilon_end", 0.01, 0.1)

    env = WebsiteNavEnv(
        adj,
        url_to_file,
        mirror_root,
        driver,
        max_steps=max_steps,
        step_penalty=step_penalty,
        goal_reward=goal_reward,
        path_shaping_mode="D",
        path_reward=path_reward,
        potential_gamma=potential_gamma,
    )
    agent = QLearningAgent(
        env,
        learning_rate=learning_rate,
        discount=discount,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
    )

    episode_rewards, success_rates = train_agent(env, agent, tasks, episodes=episodes)

    # Objective: mean success rate over the last 10% of episodes
    tail = max(1, len(success_rates) // 10)
    score = sum(success_rates[-tail:]) / float(tail)

    # Optionally, attach intermediate artifacts for the best trials
    trial.set_user_attr("episode_rewards", episode_rewards)
    trial.set_user_attr("success_rates", success_rates)

    # For the current best trial, also persist plots / Q-table / path
    # (Optuna will overwrite user_attrs when a new best is found; persistence
    # happens in main() once after optimization.)

    return score


# -----------------------------
# Plotting and saving utilities
# -----------------------------


def plot_reward_curve(episode_rewards: List[float], label: str, out_dir: str) -> None:
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.title(f"Training reward over episodes ({label})")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"reward_curve_{label}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved reward curve to {out_path}")
    except Exception as exc:
        print(f"Failed to plot reward curve for {label}: {exc}")


def plot_success_rate(success_rates: List[float], label: str, out_dir: str) -> None:
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(success_rates) + 1), success_rates)
        plt.xlabel("Episode")
        plt.ylabel("Success rate")
        plt.title(f"Training success rate over episodes ({label})")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"success_rate_{label}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved success rate curve to {out_path}")
    except Exception as exc:
        print(f"Failed to plot success rate for {label}: {exc}")


def save_q_table(agent: QLearningAgent, label: str, out_dir: str) -> None:
    try:
        data = [
            {"state": state, "action_index": action_index, "value": value}
            for (state, action_index), value in agent.q.items()
        ]
        out_path = os.path.join(out_dir, f"q_table_{label}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved Q-table to {out_path}")
    except Exception as exc:
        print(f"Failed to save Q-table for {label}: {exc}")


def save_demo_path(path: List[str], label: str, out_dir: str) -> None:
    try:
        out_path = os.path.join(out_dir, f"demo_path_{label}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for url in path:
                f.write(url + "\n")
        print(f"Saved demo path to {out_path}")
    except Exception as exc:
        print(f"Failed to save demo path for {label}: {exc}")


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
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of Optuna hyperparameter trials (default: 10)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Directory to write plots and model outputs (default: current directory).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

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
        t for t in tasks if t.get("start_url") in adj and t.get("goal_url") in adj
    ]
    if not filtered_tasks:
        raise ValueError(
            "No curriculum tasks have both start_url and goal_url in the mirror graph after remapping."
        )
    print(f"Using {len(filtered_tasks)} tasks that are present in the mirror graph after remapping.")

    driver = build_driver()
    try:
        demo_task: Optional[Dict] = None
        if args.demo_goal:
            demo_task = find_task_for_goal(filtered_tasks, args.demo_goal)
            if not demo_task:
                print(f"No task found matching goal substring: {args.demo_goal!r}")
            else:
                print("\nDemo task used for evaluation:")
                print(f"  id: {demo_task.get('id')}")
                print(f"  user_goal: {demo_task.get('user_goal')}")
                print(f"  start_url: {demo_task.get('start_url')}")
                print(f"  goal_url: {demo_task.get('goal_url')}")

        # Optuna hyperparameter search over D_potential configs.
        # new_rl_agent.py is intentionally "single-study": we let Optuna own
        # the experiment loop and only retrain the best configuration once
        # for high-signal logging and artifact generation.
        def optuna_objective(trial: optuna.trial.Trial) -> float:
            return objective(
                trial,
                adj,
                url_to_file,
                args.mirror_root,
                driver,
                filtered_tasks,
                args.episodes,
                demo_task,
                args.out_dir,
            )

        study = optuna.create_study(direction="maximize")
        print(f"Starting Optuna study with {args.trials} trials...")
        study.optimize(optuna_objective, n_trials=args.trials)

        best_trial = study.best_trial
        print("\nOptuna search complete.")
        print(f"Best trial value (mean tail success_rate): {best_trial.value:.3f}")
        print("Best hyperparameters:")
        for k, v in best_trial.params.items():
            print(f"  {k}: {v}")

        # Reconstruct best env/agent and retrain once for full logging
        max_steps = int(best_trial.params["max_steps"])
        step_penalty = -float(best_trial.params["step_penalty_mag"])
        goal_reward = float(best_trial.params["goal_reward"])
        path_reward = float(best_trial.params["path_reward"])
        potential_gamma = float(best_trial.params["potential_gamma"])
        learning_rate = float(best_trial.params["learning_rate"])
        discount = float(best_trial.params["discount"])
        epsilon_start = 0.2
        epsilon_end = float(best_trial.params["epsilon_end"])

        print("\nRetraining best configuration for logging (label: optuna_best)...")
        best_env = WebsiteNavEnv(
            adj,
            url_to_file,
            args.mirror_root,
            driver,
            max_steps=max_steps,
            step_penalty=step_penalty,
            goal_reward=goal_reward,
            path_shaping_mode="D",
            path_reward=path_reward,
            potential_gamma=potential_gamma,
        )
        best_agent = QLearningAgent(
            best_env,
            learning_rate=learning_rate,
            discount=discount,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
        )

        best_rewards, best_success = train_agent(best_env, best_agent, filtered_tasks, episodes=args.episodes)

        plot_reward_curve(best_rewards, "optuna_best", args.out_dir)
        plot_success_rate(best_success, "optuna_best", args.out_dir)
        save_q_table(best_agent, "optuna_best", args.out_dir)

        if demo_task is not None:
            path = run_policy_for_task(best_env, best_agent, demo_task)
            save_demo_path(path, "optuna_best", args.out_dir)

    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
