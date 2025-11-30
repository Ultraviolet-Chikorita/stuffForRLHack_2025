# Website RL Navigation Agent
This project implements a **tabular Q-learning agent** that learns to navigate a mirrored version of the Windsurf website - as an example, obviously replaceable with any base URL as set in the program, using a curriculum of navigation tasks.

It is structured to match three tracks of work:

- **Track 1 – Building Environments**: turning a real website into an RL Gym-like environment.
- **Track 2 – Building Task Curricula**: generating and refining navigation tasks and rewards.
- **Track 3 – Training Agents**: training and tuning agents on those tasks to make the metrics go up.

Taken together, they form an **end-to-end pipeline**:

> Live Website → Static Mirror → Curriculum of Tasks → RL Environment + Agent → Trained Policies + Evaluation Artifacts

This README is organized along those three tracks and shows where each file fits.

---

## 0. Quickstart (End-to-End Pipeline)

If you just want to go from website → trained agent as fast as possible:

```bash
# 0. Create & activate virtualenv (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. Track 1: Build / refresh environment
python hover_crawler.py \
  --start-url https://windsurf.com \
  --out-dir site_mirror

# 2. Track 2: Build / refresh curriculum
python curriculum_builder.py \
  --crawl-json crawl_result.json \
  --out curriculum_tailored.json

# 3. Track 3: Train and (optionally) tune the agent
python new_rl_agent.py \
  --mirror-root ./site_mirror \
  --curriculum curriculum_tailored.json \
  --episodes 5000 \
  --trials 20 \
  --out-dir ./outputs
```

After this, inspect `./outputs/` for reward curves, success rates, Q-tables, and demo navigation traces.

The rest of the README explains how this maps to the three tracks in more detail.

---

## 1. How This Repo Maps to the Three Tracks

### 1.1 Track 1 – Building Environments

**Goal:** Wrap a real website into a Gym-style environment with a `step()` function.

In this repo, Track 1 is implemented by:

- **`hover_crawler.py`**  
  Crawls the live Windsurf site with Selenium and generates a **static HTML mirror** and crawl graph.

- **`site_mirror/`**  
  The static website snapshot: `.html` files referencing each other by local links.

- **Mirror parsing in `new_rl_agent.py`**  
  The `parse_mirror_directory` logic scans `site_mirror/`, extracts canonical URLs and outbound links, and builds:
  - `adj[src_url] -> List[neighbor_urls]` navigation graph
  - `file_to_url` and `url_to_file` mappings

- **`WebsiteNavEnv` (in `new_rl_agent.py`)**  
  Wraps the mirror graph + Selenium into a Gym-like environment:
  - **State**: current URL (internally indexed)
  - **Actions**: choose an outbound link (edge in `adj`)
  - **Transition**: `driver.get()` the next HTML file
  - **Termination**: goal reached or `max_steps` exceeded

This is your **Environment Track artifact**: a reusable website navigation environment built from real HTML.

---

### 1.2 Track 2 – Building Task Curricula

**Goal:** Turn the raw environment into a set of interesting tasks with reward functions.

In this repo, Track 2 is implemented by:

- **`curriculum_builder.py`**  
  Consumes a crawl graph (e.g., `crawl_result.json`) and produces `curriculum_tailored.json` with tasks:
  - `start_url`: where the agent begins
  - `goal_url`: where the agent should end up
  - `user_goal`: natural language goal description
  - `path`: expert/demo navigation trace

- **`curriculum_tailored.json`**  
  The concrete curriculum used for training and evaluation.

- **Curriculum remapping in `new_rl_agent.py`**  
  Bridges between **human-friendly goals** and **exact mirror URLs**:
  - Some `goal_url` values are textual (e.g., link text instead of URL).
  - The remapper searches the mirror HTML to find the canonical destination URL.
  - Tasks whose start/goal URLs do not exist in the current mirror are filtered out.

- **Reward shaping in `WebsiteNavEnv`**  
  Uses curriculum demo paths to shape rewards:
  - `forgiving_progressive`: rewards getting closer along the demo path.
  - `strict_sequence`: rewards following the expert path exactly.
  - `dense_similarity`: rewards similarity of visited sequence vs. expert path.
  - `D_potential`: potential-based shaping over distance / path index.

This gives you a **Task Track artifact**: a progressively structured set of navigation tasks, plus reward functions that are grounded in expert traces.

---

### 1.3 Track 3 – Training Agents

**Goal:** Take the environment + curriculum and train RL agents efficiently.

In this repo, Track 3 is implemented by:

- **`QLearningAgent` (in `new_rl_agent.py`)**  
  A tabular Q-learning agent with epsilon-greedy exploration, learning rate, and discount factor.

- **Training loop `train_agent` (in `new_rl_agent.py`)**  
  Samples tasks from the curriculum, runs episodes in `WebsiteNavEnv`, and updates Q-values.

- **Reward/success logging and plotting**  
  Episode rewards and success rates are tracked and plotted to visualize learning progress.

- **Optuna integration**  
  `new_rl_agent.py` defines an Optuna objective for automated hyperparameter tuning (especially `D_potential` shaping):
  - Samples environment + agent hyperparameters per trial.
  - Trains for a fixed number of episodes.
  - Uses a success metric (e.g., average success rate) as objective.
  - Re-trains the best configuration and saves its artifacts.

- **`outputs/` directory**  
  Stores:
  - Reward curves (`reward_curve.png`)
  - Success rate plots (`success_rate.png`)
  - Serialized Q-tables
  - Demo rollouts (`demo_path.json`)

This is your **Training Track artifact**: a pipeline for training, tuning, and evaluating RL agents on a real website environment.

---

## 2. Environment Setup (Shared Across All Tracks)

These steps support all three tracks.

### 2.1. Python & Virtual Environment

- **Python version**: Use Python 3.10+ (3.11 recommended).
- From the project root:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2.2. Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies:

- `selenium`
- `webdriver-manager`
- `selenium-wire`
- `matplotlib`
- `optuna`
- `python-dotenv`

If you see `ModuleNotFoundError: No module named 'webdriver_manager'`, run:

```bash
pip install webdriver-manager
```

---

## 3. Track 1 – Building Environments

### 3.1. From Live Website to Static Mirror

If you already have a valid `site_mirror/` for Windsurf that matches your curriculum, you can skip this subsection.

Otherwise, configure and run the crawler:

1. **Optional `.env` configuration**  
   If `hover_crawler.py` relies on environment variables (auth tokens, base URL, API keys), create `.env` in the project root and fill in the values indicated by `hover_crawler.py` comments.

2. **Run the crawler**

```bash
# Example; check hover_crawler.py --help for exact flags
python hover_crawler.py \
  --start-url https://windsurf.com \
  --out-dir site_mirror
```

Output of this step (Track 1 artifact):

- `site_mirror/` populated with `.html` files.
- A crawl graph JSON (e.g., `crawl_result.json`) if configured.

### 3.2. Turning the Mirror into an Environment

`new_rl_agent.py` automatically performs the rest of Track 1 when you run it:

- Parses `site_mirror/`
- Builds the URL graph (`adj`)
- Instantiates `WebsiteNavEnv` with a Selenium driver

You generally do **not** need to call this manually; it happens inside the training CLI.

---

## 4. Track 2 – Building Task Curricula

### 4.1. From Crawl Graph to Curriculum

Running `curriculum_builder.py` turns your crawl graph into structured tasks:

```bash
python curriculum_builder.py \
  --crawl-json crawl_result.json \
  --out curriculum_tailored.json
```

Each task includes:

- `start_url`: starting page in the mirror.
- `goal_url`: target page (may be textual and later remapped).
- `user_goal`: natural language description of the user’s intent.
- `path`: expert/demo path (sequence of URLs) that solves the task.

This file is your **Curriculum artifact** used by Track 3.

### 4.2. Remapping and Validating Tasks Against the Mirror

When you run `new_rl_agent.py`, it performs several Track 2 operations:

- Remaps textual `goal_url` entries to canonical URLs by scanning the HTML mirror for matching link text.
- Discards tasks whose `start_url` or `goal_url` does not exist in the current mirror graph.
- Optionally filters tasks for a specific `--demo-goal` when running final policy rollouts.

This ensures that the curriculum is always **executable** against the current environment.

### 4.3. Reward Shaping from Expert Traces

`WebsiteNavEnv` uses the expert `path` field from each task to provide richer rewards than just success/failure:

- Reward modes like `forgiving_progressive`, `strict_sequence`, `dense_similarity`, and `D_potential` are all different ways of turning **"follow the expert path"** into a scalar reward at each `step()`.
- This makes the environment **training-friendly** for small agents and sample-limited regimes.

---

## 5. Track 3 – Training Agents

### 5.1. Basic Single-Config Training Run

To run a single training configuration end-to-end (no Optuna tuning yet):

```bash
python new_rl_agent.py \
  --mirror-root ./site_mirror \
  --curriculum curriculum_tailored.json \
  --episodes 5000 \
  --out-dir ./outputs
```

Important arguments (see `python new_rl_agent.py --help`):

- `--mirror-root`: Path to `site_mirror/` (Track 1 artifact).
- `--curriculum`: Path to `curriculum_tailored.json` (Track 2 artifact).
- `--episodes`: Number of training episodes.
- `--out-dir`: Base directory where training artifacts are written.
- `--demo-goal` (optional): Filter tasks for a specific goal and run a final demo rollout.

A typical run will:

1. Parse the mirror and build the nav graph (Track 1).
2. Load and remap curriculum tasks (Track 2).
3. Initialize `WebsiteNavEnv` + `QLearningAgent` (Track 1 + 3).
4. Train for the specified number of episodes.
5. Save plots, Q-tables, and demo rollouts into `--out-dir`.

### 5.2. Hyperparameter Tuning with Optuna

For the full Training Track experience, enable Optuna-based tuning (especially for `D_potential` shaping):

```bash
python new_rl_agent.py \
  --mirror-root ./site_mirror \
  --curriculum curriculum_tailored.json \
  --episodes 3000 \
  --trials 20 \
  --out-dir ./outputs
```

Per trial, the Optuna objective will:

1. Sample environment + agent hyperparameters.
2. Train for a fixed number of episodes.
3. Compute an objective (e.g., mean success rate over the last N episodes).
4. Report it back to Optuna.

After all trials, `new_rl_agent.py` re-trains the best configuration and saves its artifacts in a dedicated subdirectory under `./outputs`.

### 5.3. Outputs and Artifacts

Within your chosen `--out-dir` (e.g., `./outputs`), expect subdirectories per run or per trial that may contain:

- `reward_curve.png` – Episode reward plot.
- `success_rate.png` – Success rate over episodes.
- `q_table.pkl` / `q_table.json` – Serialized Q-table.
- `demo_path.json` – Agent rollout for a selected task after training.
- Optuna study-related logs or artifacts.

These artifacts are what you would use to **compare methods, debug reward shaping, and evaluate different agents or environments**.

---

## 7. Using This Pipeline on Hugging Face Infrastructure

You can treat this repository as a **backend training pipeline** and layer Hugging Face tooling on top.

### 7.1. Hugging Face Spaces (Interactive Frontend + Backend)

- Create a new **Space** in your org (Python / Gradio / Streamlit).
- Copy this repo into the Space.
- Ensure `requirements.txt` is present.
- In `app.py`, call into `new_rl_agent.py` (directly as Python functions or via subprocess) to:
  - Launch new training or tuning runs.
  - Visualize reward and success curves.
  - Show navigation rollouts for a chosen task.

You can configure the Space to use a GPU if you later add larger models (e.g., for policy networks or reward models). For pure tabular Q-learning, CPU is sufficient.

### 7.2. API / Endpoint Style

If you want **testing-friendly automation** from other services:

- Wrap parts of `new_rl_agent.py` in a small FastAPI/Flask app.
- Expose endpoints such as:
  - `/train` – trigger a training/tuning run.
  - `/run_task` – run the current best policy on a given task and return the path.
  - `/metrics` – fetch reward/success metrics.
- Deploy this app as a Space or a custom Hugging Face Inference Endpoint.

Your CI / evaluation harness can then call those endpoints to:

- Spin up new experiments.
- Evaluate agents on standardized tasks.
- Log metrics to your own dashboards.

---

## 8. Troubleshooting

- **`ModuleNotFoundError` for Selenium or other packages**  
  Ensure your virtualenv is active and `pip install -r requirements.txt` has run.

- **Chromedriver / browser issues**  
  `webdriver-manager` should automatically download a compatible ChromeDriver, but a matching Chrome/Chromium installation may still be required.

- **Curriculum goals not found in mirror**  
  The code remaps textual goals via mirror HTML. If many tasks are discarded, ensure your mirror is up-to-date and the curriculum was built from the same crawl.

- **Reward too low / agent not converging**  
  Increase episodes, tweak shaping mode, or rely on Optuna tuning for `D_potential` and other parameters.

If you run into issues not covered here, inspect `new_rl_agent.py` comments for design intent and debug hints.


