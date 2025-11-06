# Analytics and Business Intelligence - Weekly Learning Project

A PyQt6-based GUI application for running and visualizing Python analytics code from Weeks 1-10. This project provides a clean interface to execute weekly exercises and view results with data analysis, machine learning, and visualization capabilities.

## Prerequisites

- Python 3.10 or higher
- Git (for cloning the repository)
- pip (Python package manager)

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd weekly_learning_vscode
```

### 2. Create a Virtual Environment

The virtual environment (`.venv`) is **not included** in the repository, so you need to create it:

**Windows (PowerShell):**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

With the virtual environment activated, install all required packages:

```bash
pip install -r requirements.txt
```

This will install:
- PyQt6/PySide6 (GUI framework)
- pandas, numpy (data processing)
- matplotlib, seaborn, plotly (visualization)
- scikit-learn, statsmodels (machine learning & statistics)
- prophet (time series forecasting)
- And other dependencies listed in [requirements.txt](requirements.txt)

### 4. Verify Installation

Test that the installation was successful:

```bash
python test_week.py
```

## Running the Application

After activating your virtual environment, run:

```bash
python qt_app.py
```

This will launch the GUI application where you can:
- Select which week's code to run (Week 1-10)
- Load and visualize data
- View analysis results and charts
- Export outputs

## Project Structure

```
weekly_learning_vscode/
├── qt_app.py              # Main PyQt6 GUI application
├── test_week.py           # Test runner for individual weeks
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .vscode/              # VS Code configuration
│   ├── settings.json
│   └── tasks.json
├── assets/               # Application assets
│   └── logo.txt
├── data/                 # CSV data files
│   └── Gross disposable household income...csv
├── outputs/              # Generated charts and analysis results
│   ├── week1_*.png
│   ├── week2_*.png
│   └── ...
├── utils/                # Utility modules
│   ├── data_loader.py   # CSV loading utilities
│   └── qt_mpl.py        # Matplotlib integration for Qt
└── weeks/                # Weekly exercise modules
    ├── __init__.py
    ├── week1.py         # Week 1 exercises
    ├── week2.py         # Week 2 exercises
    ├── ...
    └── week10.py        # Week 10 exercises
```

## Adding Your Own Code

### For Weekly Exercises

1. Navigate to `weeks/weekX.py` (where X is the week number)
2. Add your code inside the `run(config)` function
3. Use the `config` parameter to access:
   - Data file paths
   - Output directory
   - Any configuration options

Example:
```python
def run(config):
    # Your code here
    df = pd.read_csv(config.get('data_file'))
    # Process data, create visualizations, etc.
    return results
```

### Adding Data Files

1. Place your CSV files in the `data/` folder
2. Reference them in your code using relative paths: `data/myfile.csv`
3. Use the `utils.data_loader.load_csv()` function for consistent data loading

## Recommended VS Code Extensions

- **Python** (Microsoft) - Python language support
- **Pylance** (Microsoft) - Fast Python language server
- **Python Debugger** (Microsoft) - Debugging support

## Troubleshooting

### Virtual Environment Issues

If you have trouble activating the virtual environment:
- Ensure you're using the correct command for your OS/shell
- On Windows, you may need to enable script execution: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Import Errors

If you get import errors:
- Make sure your virtual environment is activated (you should see `(.venv)` in your terminal prompt)
- Reinstall dependencies: `pip install -r requirements.txt`

### Qt Platform Plugin Issues

If you get "Could not find the Qt platform plugin" errors:
- Try reinstalling PyQt6: `pip uninstall PyQt6 && pip install PyQt6`
- Ensure you're running from the activated virtual environment

## Contributing

When adding new weekly exercises:
1. Create or modify the corresponding `weeks/weekX.py` file
2. Follow the existing structure with a `run(config)` function
3. Save outputs to the `outputs/` directory
4. Update this README if adding new dependencies

## License

This is an educational project for Analytics and Business Intelligence (QHO539).

## Notes

- The `.venv` directory is excluded from git (via `.gitignore`) to keep the repository clean
- Generated outputs in `outputs/` and data files in `data/` are tracked to preserve analysis results
- Cache files (`__pycache__/`) are also excluded from version control
