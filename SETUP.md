# Setup & run guide — AI Email Assistant

Step by step, from a fresh terminal to a working demo. Windows instructions first
(that is what this project sits on); macOS/Linux equivalents are at the bottom.

**Time:** about 5 minutes, most of it waiting on `pip`.

---

## Step 0 — Check Python is installed

Open **PowerShell** (Start menu → type "PowerShell" → Enter) and run:

```powershell
python --version
```

You should see `Python 3.10.x` or newer. If you get "not recognized", install
Python from <https://www.python.org/downloads/> and tick **"Add Python to PATH"**
during the install, then reopen PowerShell.

---

## Step 1 — Go to the project folder

```powershell
cd "$env:USERPROFILE\OneDrive\Desktop\Internship\SafeX-AI-Chatbot"
```

Confirm you are in the right place — this should list `app.py`, `email_assistant`
and `venv`:

```powershell
dir
```

---

## Step 2 — The virtual environment

A virtual environment ("venv") is a private folder of Python packages for this
project, so installing things here cannot break anything else on your machine.

**This project already has one at `venv\`, from the Week 1 chatbot.** Reuse it —
both modules then share a single environment.

### If `venv\` exists (the normal case)

Nothing to create. Skip to Step 3.

### If you want a fresh one instead, or `venv\` is missing

```powershell
python -m venv venv
```

That creates the `venv` folder. It takes a few seconds.

---

## Step 3 — Activate the venv

```powershell
.\venv\Scripts\Activate.ps1
```

Your prompt should now start with `(venv)`:

```
(venv) PS C:\Users\eesaa\OneDrive\Desktop\Internship\SafeX-AI-Chatbot>
```

That `(venv)` prefix is how you know it worked. **You need this on every new
terminal you open** — it is not permanent.

> **If PowerShell blocks the script** with *"running scripts is disabled on this
> system"*, run this once, then retry the activate command:
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
>
> Answer `Y` when prompted. This only affects your own user account.

> **Using Command Prompt (cmd) instead of PowerShell?** The command is:
> ```cmd
> venv\Scripts\activate.bat
> ```

---

## Step 4 — Install the requirements

With `(venv)` showing in your prompt:

```powershell
python -m pip install --upgrade pip
python -m pip install -r email_assistant\requirements.txt
```

This installs pandas, NumPy, scikit-learn, FastAPI, uvicorn, requests, Jupyter and
matplotlib. Expect 1–3 minutes on a first run. Some are already present from Week 1
and will be skipped.

Check it worked:

```powershell
python -c "import pandas, sklearn, fastapi, uvicorn; print('all packages OK')"
```

You want to see `all packages OK`.

---

## Step 5 — Move into the module folder

```powershell
cd email_assistant
```

Every command from here assumes you are inside `email_assistant\` with the venv
active. The module reads its data with paths relative to itself, so running from
the wrong folder is the most common cause of a `FileNotFoundError`.

---

## Step 6 — Run the console demo

This is the fastest proof that everything works, and the best thing to screen-record:

```powershell
python run_demo.py
```

You should see four sections scroll past:

1. Information extraction on a single email
2. Intent classification accuracy (100% labelled corpus, 100% holdout)
3. Three drafted replies
4. The 40-email inbox sorted by priority

To also save the output to a file for your submission:

```powershell
python run_demo.py --save
```

That writes `demo_output.txt` next to the script.

---

## Step 7 — Run the web UI

```powershell
uvicorn api:app --port 8000
```

Leave that terminal running — it is the server. You will see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Now open your browser to:

- **<http://localhost:8000/>** — the demo interface
- **<http://localhost:8000/docs>** — interactive API documentation

In the UI: pick a sample email from the dropdown, click **Generate draft**. Then
click **Process whole inbox** to see all 40 emails triaged by priority, and click
any row to view its draft.

Press **Ctrl+C** in the terminal to stop the server.

> If port 8000 is already in use, pick another: `uvicorn api:app --port 8080`
> (and browse to `localhost:8080`).

---

## Step 8 — Run the notebook

Open a **new** terminal, activate the venv again (Step 3), then:

```powershell
cd email_assistant
jupyter notebook notebooks\email_assistant_demo.ipynb
```

Your browser opens the notebook. Use **Cell → Run All** (or **Kernel → Restart &
Run All**) to execute everything fresh. The charts and confusion matrix are already
saved in the file, so it also reads fine without running anything.

---

## Step 9 (optional) — Turn on the LLM polish

Without this the assistant uses its built-in templates, which is fully functional.
With it, drafts get rewritten into more natural prose by Llama 3.3 70B.

1. Get a free API key at <https://console.groq.com> (no credit card needed).
2. In your terminal, before running anything:

```powershell
$env:GROQ_API_KEY = "gsk_your_key_here"
```

(Command Prompt: `set GROQ_API_KEY=gsk_your_key_here`)

3. Run `python run_demo.py` again. The header should now read
   `LLM polish: ON (Groq)` and drafts will come back with `via=llm`.

The key lasts only for that terminal session. To make it permanent:

```powershell
[Environment]::SetEnvironmentVariable("GROQ_API_KEY", "gsk_your_key_here", "User")
```

Then close and reopen the terminal.

> **Never commit your API key to GitHub.** Set it as an environment variable as
> shown above — do not paste it into `config.py`.

---

## Quick reference — the whole thing in one block

```powershell
cd "$env:USERPROFILE\OneDrive\Desktop\Internship\SafeX-AI-Chatbot"
.\venv\Scripts\Activate.ps1
python -m pip install -r email_assistant\requirements.txt
cd email_assistant
python run_demo.py            # console demo
uvicorn api:app --port 8000   # web UI at http://localhost:8000/
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `'python' is not recognized` | Python not installed, or not on PATH | Reinstall Python with "Add to PATH" ticked |
| `running scripts is disabled on this system` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `ModuleNotFoundError: No module named 'pandas'` | venv not activated, or requirements not installed | Check for `(venv)` in your prompt; rerun Step 4 |
| `FileNotFoundError: ...sample_emails.csv` | Running from the wrong folder | `cd email_assistant` first |
| `'uvicorn' is not recognized` | venv not active, or uvicorn not installed | Activate venv, then `python -m uvicorn api:app --port 8000` |
| `[Errno 10048] address already in use` | Port 8000 taken | Use `--port 8080` and browse to `localhost:8080` |
| Browser shows "can't reach this page" | Server not running | The uvicorn terminal must stay open while you use the UI |
| UI header says "API offline" | Opened `demo.html` directly as a file | Go through `http://localhost:8000/`, not by double-clicking the HTML |
| Drafts say `via=template` after setting the key | Key set in a different terminal | Set `GROQ_API_KEY` in the same terminal you run from |

---

## macOS / Linux

Same steps, different commands:

```bash
cd ~/Desktop/Internship/SafeX-AI-Chatbot
python3 -m venv venv                    # only if venv/ does not exist
source venv/bin/activate                # activate
python -m pip install -r email_assistant/requirements.txt
cd email_assistant
python run_demo.py
uvicorn api:app --port 8000
export GROQ_API_KEY="gsk_your_key_here" # optional LLM polish
```

---

## Optional — better name extraction with spaCy

The extractor works on regex alone. Installing spaCy switches on proper
person/organisation/date recognition automatically, with no code change:

```powershell
python -m pip install spacy
python -m spacy download en_core_web_sm
```

Roughly 50 MB of download. Rerun `python run_demo.py` — the `spacy_used` field in
the extraction output will flip to `True`.
