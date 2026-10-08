# Testing and running Ghost Ledger in VS Code

This guide takes you from a fresh clone to a green test run and running
dashboards, using VS Code only. The files it uses are already in the repo:
`.vscode/settings.json` (pytest discovery) and `.vscode/launch.json` (run
configurations).

## 1. Prerequisites

| Need | Version | Check |
|---|---|---|
| Python | 3.12 (the pinned `xgboost==3.4.1` needs it) | `python3.12 --version` |
| VS Code | recent | — |
| VS Code extension | **Python** (Microsoft) and **Python Debugger** (installed with it) | Extensions panel |
| Git | any | `git --version` |

Python 3.11 can run the test suite if you install `xgboost` 3.2.0 instead. The
build sandbox did this and got 292 passing tests. Use 3.12 if you can.

## 2. Open the project and create the environment

1. **File → Open Folder…** and choose the `Razorpay_Track_03_Ghost_Ledger` folder.
2. Open a terminal (**Terminal → New Terminal**) and run:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Pick the interpreter: press **Ctrl+Shift+P** (Cmd+Shift+P on macOS), run
   **Python: Select Interpreter**, and choose `.venv`. The status bar then shows
   `.venv`.

The `.venv` folder is git-ignored, so it never reaches a commit.

## 3. Create your `.env` (optional for tests)

Tests run without `.env`. Create one only when you want the API, the webhook,
or live Test Mode:

```bash
cp .env.example .env
```

Fill in what you need. Never commit `.env`, and never paste its secrets into
chat or into a file in the repo. The `.gitignore` already blocks it.

## 4. Run the tests

### Option A: the Testing panel (recommended)

1. Click the **flask-beaker icon** (Testing) in the Activity Bar. If it asks
   which framework to use, choose **pytest**, then the `tests` folder. The
   settings file already sets this, so it should not ask.
2. Click the **▶ Run All Tests** button. Expect **292 passed**.
3. Open any test file, for example `tests/test_v3_strategies.py`. Click the
   green ▶ beside a test function to run just that test. Right-click it and
   choose **Debug Test** to step through it.

### Option B: the Run and Debug panel

1. Click the **Run and Debug** icon (Ctrl+Shift+D).
2. In the dropdown, choose **Pytest: all tests**. Press **F5**.
3. Choose **Pytest: current file** to run whatever file is open.

### Option C: the terminal

```bash
pytest tests -q                       # all tests, about one minute
pytest tests/test_v3_strategies.py -q # one file
pytest -k "playbook" -q               # any test whose name matches
```

### Turning on Track B for a test run

The strategy engine is off by default. To test it with the switch on:

```bash
ENABLE_ADVANCED_STRATEGIES=1 pytest tests -q
```

With the switch on, one scheduler test is expected to fail. That test asserts
exactly four jobs, and the switch adds a fifth (dunning). This is a known,
expected result (291 passed, 1 expected failure).

## 5. Run the pipeline and dashboards

Use the **Run and Debug** dropdown. Each entry has a short name:

| Dropdown entry | What it starts | Open in browser |
|---|---|---|
| **Pipeline: v2 batch (no autopsy)** | Generates data and the model, runs the recovery pipeline once. Run it first. | — |
| **API server (webhooks, approvals, merchant)** | `uvicorn` on port 8000 | http://localhost:8000/docs and http://localhost:8000/merchant |
| **Streamlit: v2 dashboard (8501)** | The main recovery dashboard | http://localhost:8501 |
| **Streamlit: live progress (8502, needs API)** | The live progress page. It calls the API, so start the API first. | http://localhost:8502 |

Order for the full set:

1. Run **Pipeline: v2 batch (no autopsy)** once (about a minute or two).
2. Start **API server**.
3. Start **Streamlit: v2 dashboard** and **Streamlit: live progress**.

Each one runs in its own terminal tab. Stop one with the red square in the
terminal, or with Ctrl+C.

### Setting the operator key for the live progress page

The live progress page needs `OPERATOR_API_KEY`. Put it in `.env` (any long
random string) before starting the API and the page. The launch configurations
read `.env` automatically (`envFile`). Do not put the key in `launch.json`.

## 6. Debugging a failing test

1. Put a red breakpoint in the source, by clicking in the left gutter.
2. Right-click the failing test in the Testing panel and choose **Debug Test**.
3. Use the Variables panel and the Debug Console to inspect values.

If a test seems to change the demo data, check `tests/conftest.py` (or the
top of the test file). The suite copies the demo database to
`data/test_ghost_ledger.db` and works on that copy.

## 7. Checking the strategy engine by hand

With the switch on (`ENABLE_ADVANCED_STRATEGIES=1` in `.env`), run the
pipeline entry. The output starts with the playbook summary. The headline
should match the switch-off run: the recovery figures are identical with the
switch on (this was verified in the build).

To check the retry timing comparison:

```bash
python scripts/retry_timing_holdout.py
```

It writes `reports/retry_timing_holdout.json`. The result is a simulation, not
live data (see `docs/STRATEGIES.md`).

## 8. Common problems

| Problem | Fix |
|---|---|
| The Testing panel shows **no tests** | Select the interpreter (`.venv`), then click the refresh icon in the Testing panel. Check the **Python Test Log** in the Output panel. |
| `ModuleNotFoundError` in tests | The terminal or interpreter is not `.venv`. Re-select the interpreter and reopen the terminal. |
| `xgboost` install fails | Use Python 3.12. Python 3.11 needs `xgboost==3.2.0` (edit the install command, not `requirements.txt`). |
| `Address already in use` on 8000, 8501, or 8502 | Another process has the port. Stop the old terminal, or change the port in the launch entry. |
| The live progress page says it cannot reach the API | Start **API server** first. The page reads `API_BASE_URL` (default `http://127.0.0.1:8000`). |
| The live progress page shows 503 | `OPERATOR_API_KEY` is not set. Add it to `.env` and restart the API and the page. |
| Tests pass but the dashboard shows no numbers | Run **Pipeline: v2 batch** once. |
| Git shows `data/` or `models/` changes after a run | Those are regenerated by the pipeline. Do not commit them unless you intend to change the reference reports. Restore with `git checkout -- <file>`. |

## 9. Before you commit

```bash
pytest tests -q          # must be green
git status               # check that .env and data/*.db are not listed
```

`.env`, `.venv/`, and the databases are git-ignored, so they should not appear.
