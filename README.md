# Analytics-and-Business-Intelligence-QHO539-
Year 2 Study PLR
# Weekly Learning (Weeks 1–10)

Run your Week 1–10 Python code from one place with a clean GUI (Streamlit).
Open this folder in **VS Code**, install the extensions below, and press **Run App**.

## Recommended VS Code extensions
- Python (Microsoft)
- Pylance (Microsoft)

## Quick start
```bash
# 1) Create & activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the GUI
# Option A: VS Code > Terminal > Run Task > Run Streamlit App
# Option B: from terminal
streamlit run app.py
```

## Where to paste your weekly code
Put the code for each week into `weeks/weekX.py` inside the `run(config)` function.
If you already have plotting/regression code, just drop it in there and keep the function signature.

## Data
Place CSVs in `data/` and reference them by relative path (e.g., `data/myfile.csv`).

## Structure
```
weekly_learning_vscode/
├─ app.py
├─ requirements.txt
├─ README.md
├─ .vscode/
│  ├─ settings.json
│  └─ tasks.json
├─ assets/
│  └─ logo.txt
├─ data/
│  └─ sample.csv
├─ utils/
│  └─ data_loader.py
└─ weeks/
   ├─ __init__.py
   ├─ week1.py
   ├─ week2.py
   ├─ week3.py
   ├─ week4.py
   ├─ week5.py
   ├─ week6.py
   ├─ week7.py
   ├─ week8.py
   ├─ week9.py
   └─ week10.py
```


