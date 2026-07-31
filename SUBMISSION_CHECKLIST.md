# Submission checklist — Week 2 AI Email Assistant

Seven required items. Status and where each one lives.

| # | Required | Status | What / where |
|---|---|---|---|
| 1 | Source code | Done | All `.py` files + `demo.html` in `email_assistant/` |
| 2 | Documentation | Done | `README.md`, `SETUP.md` |
| 3 | Progress report | Done | `PROGRESS_REPORT.md` |
| 4 | Working demo | Done | `run_demo.py`, the FastAPI app, the notebook |
| 5 | Screenshots | **You** | Capture 6 — see `screenshots/SCREENSHOT_GUIDE.md` |
| 6 | GitHub repository | **You** | Push the repo — steps below |
| 7 | Explanation video | **You** | Record 4–6 min — outline in `screenshots/SCREENSHOT_GUIDE.md` |

---

## Do these three, in order

### A. Screenshots (~10 min)

1. Follow Steps 3–7 of `SETUP.md` to get the demo running.
2. Capture the six shots listed in `screenshots/SCREENSHOT_GUIDE.md`, saving them
   into the `screenshots/` folder with the suggested filenames.
3. The single most important one is the web UI after clicking **Process whole
   inbox** — it shows the whole thing working at a glance.

Windows shortcut: **Win + Shift + S** to snip, then paste into Paint and save.

### B. GitHub repository (~10 min)

From the project root (`SafeX-AI-Chatbot`):

```powershell
cd "$env:USERPROFILE\OneDrive\Desktop\Internship\SafeX-AI-Chatbot"
git init
git add .
git commit -m "Week 2: AI Email Assistant prototype"
```

Then create an empty repo on github.com (no README/gitignore — you already have
them), copy its URL, and:

```powershell
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

A `.gitignore` is already in place, so `venv/`, `__pycache__/` and the `.zip` are
excluded automatically. Take the screenshots **before** this step so the images get
committed too.

> **Check before pushing:** open `config.py` and confirm there is no real API key
> in it (there should not be — the key comes from an environment variable). Never
> commit a Groq key.

### C. Explanation video (~15 min incl. a retake)

Use the outline in `screenshots/SCREENSHOT_GUIDE.md`. Record your screen with the
demo running and talk through: the problem, the pipeline, a live draft, the
"LLM rewrites, never authors" design decision, and the honest accuracy numbers.

Free recorders: **Windows Game Bar** (Win + G), OBS Studio, or Loom.

Keep the honesty point in — showing the 64.3% holdout result and explaining it
reads as stronger engineering than claiming 100%.

---

## What to actually hand in

Most internship programs want either a GitHub link or a zipped folder (often both).

- **GitHub link** — the repo URL from step B. This covers source code,
  documentation, progress report, working demo, and screenshots in one place.
- **Video** — upload to Google Drive / YouTube (unlisted) / Loom and submit the
  link, or attach the file if the form allows. Video files are usually too big to
  put in the Git repo.
- **Zip (if requested)** — right-click the `email_assistant` folder →
  *Send to → Compressed (zipped) folder*. Delete any `venv/` or `__pycache__/`
  inside first so it stays small.

### One-line summary for the submission form

> Week 2 individual contribution: an AI Email Assistant that classifies inbound
> customer emails into 13 intents (hybrid rules + TF-IDF/Logistic Regression) and
> drafts reply suggestions, with priority-based triage and human-review routing.
> Built with Python, pandas, scikit-learn and FastAPI, with an optional Groq LLM
> polish layer. Repo: <your link> · Demo video: <your link>
