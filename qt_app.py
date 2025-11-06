from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from PySide6 import QtCore, QtWidgets, QtGui  # type: ignore
else:  # pragma: no cover - runtime compatibility shim
    try:
        from PySide6 import QtCore, QtWidgets, QtGui  # type: ignore
    except Exception:
        from PyQt6 import QtCore, QtWidgets, QtGui  # type: ignore
try:
    import qdarktheme  # type: ignore
except Exception:
    qdarktheme = None  # type: ignore
try:
    import qdarkstyle  # type: ignore
except Exception:
    qdarkstyle = None  # type: ignore
from matplotlib.figure import Figure

from utils.data_loader import load_csv
from utils.qt_mpl import MplWidget


PROJECT_ROOT = Path(__file__).resolve().parent


def _find_default_csv() -> str:
    data_dir = PROJECT_ROOT / "data"
    if data_dir.exists():
        csvs = sorted(data_dir.glob("*.csv"))
        if csvs:
            return str(csvs[0])
    return ""


def _resolve_attr(root: Any, path: str) -> Any:
    node = root
    for part in path.split("."):
        node = getattr(node, part)
    return node


def _choose_attr(root: Any, *paths: str) -> Any:
    for path in paths:
        try:
            return _resolve_attr(root, path)
        except AttributeError:
            continue
    raise AttributeError(f"None of the attribute paths {paths!r} found on {root!r}")


try:
    Signal = QtCore.Signal  # type: ignore[attr-defined]
except AttributeError:
    Signal = QtCore.pyqtSignal  # type: ignore[attr-defined]

ALIGN_CENTER = _choose_attr(QtCore, "Qt.AlignCenter", "Qt.AlignmentFlag.AlignCenter")
SCROLLBAR_AS_NEEDED = _choose_attr(
    QtCore,
    "Qt.ScrollBarPolicy.AsNeeded",
    "Qt.ScrollBarPolicy.ScrollBarAsNeeded",
    "Qt.ScrollBarAsNeeded",
)


class Sidebar(QtWidgets.QWidget):
    run_requested = Signal(str, str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        # Week selection
        self.week_combo = QtWidgets.QComboBox()
        self.week_combo.addItems([f"Week {i}" for i in range(1, 11)])
        self.week_combo.setToolTip("Select which week's analysis to run (1-10)")

        # Data path input
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("Optional CSV path…")
        self.path_edit.setToolTip("Path to GDHI CSV file (leave empty for default)")
        default_path = _find_default_csv()
        self.path_edit.setText(default_path)

        # Browse button
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.setToolTip("Select a CSV file from your computer")
        browse_btn.clicked.connect(self._browse)

        # Run button - STYLED AND PROMINENT
        self.run_btn = QtWidgets.QPushButton("▶ Run Analysis")
        self.run_btn.setDefault(True)
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d7377;
                color: white;
                font-size: 13px;
                font-weight: bold;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #14a59e;
            }
            QPushButton:pressed {
                background-color: #0a5c5f;
            }
            QPushButton:disabled {
                background-color: #4a4a4a;
                color: #888888;
            }
        """)
        self.run_btn.setToolTip("Run the selected week's analysis (Ctrl+R)")
        self.run_btn.clicked.connect(self._emit_run)

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow("Week:", self.week_combo)
        form.setSpacing(10)

        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(self.path_edit, 3)
        path_row.addWidget(browse_btn, 1)

        outer = QtWidgets.QVBoxLayout(self)

        # Title with styling
        title = QtWidgets.QLabel("📊 Weekly Learning")
        font = title.font()
        font.setPointSize(14)
        font.setBold(True)
        title.setFont(font)
        align_center = ALIGN_CENTER
        title.setAlignment(align_center)
        title.setStyleSheet("padding: 10px; color: #14a59e;")

        # Subtitle
        subtitle = QtWidgets.QLabel("GDHI Data Analysis")
        subtitle.setAlignment(align_center)
        subtitle.setStyleSheet("color: #888888; font-size: 10px; padding-bottom: 15px;")

        # Separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: #444444;")

        outer.addWidget(title)
        outer.addWidget(subtitle)
        outer.addWidget(separator)
        outer.addSpacing(10)
        outer.addLayout(form)
        outer.addSpacing(5)
        outer.addWidget(QtWidgets.QLabel("Data Source:"))
        outer.addLayout(path_row)
        outer.addSpacing(15)
        outer.addWidget(self.run_btn)
        outer.addStretch(1)

        # Footer info
        info_label = QtWidgets.QLabel("💡 Tip: Use Ctrl+R to run")
        info_label.setStyleSheet("color: #666666; font-size: 9px; padding: 5px;")
        info_label.setAlignment(align_center)
        info_label.setWordWrap(True)
        outer.addWidget(info_label)

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select CSV", str(Path.cwd()), "CSV Files (*.csv)")
        if path:
            self.path_edit.setText(path)

    def _emit_run(self):
        self.run_requested.emit(self.week_combo.currentText(), self.path_edit.text())


class CenterPane(QtWidgets.QStackedWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        align_center = ALIGN_CENTER

        # Create styled welcome screen
        welcome_widget = QtWidgets.QWidget()
        welcome_layout = QtWidgets.QVBoxLayout(welcome_widget)
        welcome_layout.setAlignment(align_center)

        # Welcome icon/title
        welcome_title = QtWidgets.QLabel("📈 Welcome to Weekly Learning")
        title_font = welcome_title.font()
        title_font.setPointSize(20)
        title_font.setBold(True)
        welcome_title.setFont(title_font)
        welcome_title.setAlignment(align_center)
        welcome_title.setStyleSheet("color: #14a59e; padding: 20px;")

        # Subtitle
        subtitle = QtWidgets.QLabel("UK GDHI Data Analysis • 10-Week Educational Curriculum")
        subtitle.setAlignment(align_center)
        subtitle.setStyleSheet("color: #888888; font-size: 12px; padding-bottom: 30px;")

        # Instructions card
        instructions = QtWidgets.QLabel(
            "🔹 Select a week (1-10) from the sidebar\n"
            "🔹 Choose your CSV data file\n"
            "🔹 Click 'Run Analysis' or press Ctrl+R\n"
            "🔹 View results in interactive tabs"
        )
        instructions.setAlignment(align_center)
        instructions.setStyleSheet("""
            QLabel {
                background-color: #2a2a2a;
                border: 1px solid #444444;
                border-radius: 10px;
                padding: 30px;
                font-size: 13px;
                line-height: 1.8;
                color: #cccccc;
            }
        """)
        instructions.setWordWrap(True)

        # Dataset info
        dataset_info = QtWidgets.QLabel(
            "📊 Dataset: Gross Disposable Household Income (GDHI)\n"
            "📍 Coverage: UK NUTS3 Local Areas (170 regions)\n"
            "📅 Period: 1997-2016 (20 years)"
        )
        dataset_info.setAlignment(align_center)
        dataset_info.setStyleSheet("color: #666666; font-size: 11px; padding-top: 30px;")

        welcome_layout.addStretch()
        welcome_layout.addWidget(welcome_title)
        welcome_layout.addWidget(subtitle)
        welcome_layout.addSpacing(20)
        welcome_layout.addWidget(instructions)
        welcome_layout.addWidget(dataset_info)
        welcome_layout.addStretch()

        self.placeholder = welcome_widget
        self.addWidget(self.placeholder)

    def show_widget(self, widget: QtWidgets.QWidget):
        self.addWidget(widget)
        self.setCurrentWidget(widget)


class WorkerThread(QtCore.QThread):
    """Worker thread for running heavy computation off the main GUI thread."""
    finished = Signal(object)  # Emits the result from build_widget
    error = Signal(str)  # Emits error message
    progress = Signal(str)  # Emits progress messages

    def __init__(self, module_name: str, config: Dict[str, Any]):
        super().__init__()
        self.module_name = module_name
        self.config = config

    def run(self):
        try:
            # Set matplotlib to use non-interactive backend to avoid threading issues
            import matplotlib
            matplotlib.use('Agg')  # Thread-safe backend

            self.progress.emit(f"Importing {self.module_name}...")
            mod = importlib.import_module(self.module_name)

            if hasattr(mod, "build_widget"):
                self.progress.emit(f"Building widget for {self.module_name}...")
                output = mod.build_widget(self.config)  # type: ignore
                self.progress.emit("Creating display...")
                self.finished.emit(output)
            else:
                self.error.emit(f"{self.module_name} does not have build_widget function")
        except Exception as e:
            import traceback
            error_msg = f"Error in {self.module_name}:\n{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📊 Weekly Learning - GDHI Data Analysis")
        self.resize(1200, 800)

        # Create menu bar
        self._create_menu_bar()

        # Main content
        self.sidebar = Sidebar()
        self.center = CenterPane()

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self.sidebar)
        splitter.addWidget(self.center)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 920])
        self.setCentralWidget(splitter)

        self.sidebar.run_requested.connect(self.on_run)

        # Status bar with progress bar
        self.status = self.statusBar()
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMaximum(0)  # Indeterminate mode
        self.progress_bar.hide()
        self.status.addPermanentWidget(self.progress_bar)

        # Add keyboard shortcuts
        self._create_shortcuts()

        # Worker thread for background processing
        self.worker: Optional[WorkerThread] = None

    def _create_menu_bar(self):
        """Create the menu bar with File, View, and Help menus."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = file_menu.addAction("📂 Open CSV...")
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open a CSV data file")
        open_action.triggered.connect(self._open_file)

        file_menu.addSeparator()

        exit_action = file_menu.addAction("❌ Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)

        # View menu
        view_menu = menubar.addMenu("&View")

        reset_action = view_menu.addAction("🔄 Reset View")
        reset_action.setStatusTip("Return to welcome screen")
        reset_action.triggered.connect(self._reset_view)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = help_menu.addAction("ℹ️ About")
        about_action.setStatusTip("About this application")
        about_action.triggered.connect(self._show_about)

        shortcuts_action = help_menu.addAction("⌨️ Keyboard Shortcuts")
        shortcuts_action.setStatusTip("View keyboard shortcuts")
        shortcuts_action.triggered.connect(self._show_shortcuts)

    def _create_shortcuts(self):
        """Create keyboard shortcuts."""
        # Ctrl+R to run analysis
        run_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self)
        run_shortcut.activated.connect(self.sidebar._emit_run)

    def _open_file(self):
        """Open file dialog and set path in sidebar."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select CSV File", str(Path.cwd()), "CSV Files (*.csv);;All Files (*.*)"
        )
        if path:
            self.sidebar.path_edit.setText(path)
            self.status.showMessage(f"Loaded: {Path(path).name}", 3000)

    def _reset_view(self):
        """Reset to welcome screen."""
        self.center.setCurrentWidget(self.center.placeholder)
        self.status.showMessage("View reset", 2000)

    def _show_about(self):
        """Show about dialog."""
        QtWidgets.QMessageBox.about(
            self,
            "About Weekly Learning",
            "<h3>📊 Weekly Learning - GDHI Analysis</h3>"
            "<p><b>Version:</b> 1.0</p>"
            "<p><b>Description:</b> Educational data science curriculum analyzing UK Gross "
            "Disposable Household Income (GDHI) data across 10 weeks.</p>"
            "<p><b>Dataset:</b> UK NUTS3 Local Areas (1997-2016)</p>"
            "<p><b>Regions:</b> 170 areas</p>"
            "<p><b>Technologies:</b> Python, PyQt6, Matplotlib, Pandas, NumPy, Scikit-learn</p>"
        )

    def _show_shortcuts(self):
        """Show keyboard shortcuts dialog."""
        QtWidgets.QMessageBox.information(
            self,
            "Keyboard Shortcuts",
            "<h3>⌨️ Keyboard Shortcuts</h3>"
            "<table cellpadding='5'>"
            "<tr><td><b>Ctrl+R</b></td><td>Run selected week</td></tr>"
            "<tr><td><b>Ctrl+O</b></td><td>Open CSV file</td></tr>"
            "<tr><td><b>Ctrl+Q</b></td><td>Exit application</td></tr>"
            "</table>"
        )

    def on_run(self, week_label: str, data_path: str):
        # Prevent starting a new task if one is already running
        if self.worker is not None and self.worker.isRunning():
            self.status.showMessage("⚠️ Please wait for the current task to complete...", 3000)
            return

        # Show progress bar and update UI
        self.progress_bar.show()
        self.status.showMessage(f"🔄 Running {week_label} - Please wait...", 0)
        self.sidebar.run_btn.setEnabled(False)
        self.sidebar.run_btn.setText("⏳ Processing...")

        config: Dict[str, Any] = {"data_path": data_path, "debug": True}
        module_name = f"weeks.week{week_label.split()[-1]}"

        # Create and configure worker thread
        self.worker = WorkerThread(module_name, config)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(lambda output: self._on_success(week_label, output, config))
        self.worker.error.connect(lambda msg: self._on_error(week_label, msg))
        self.worker.start()

    def _on_progress(self, message: str):
        """Update status bar with progress messages from worker thread."""
        self.status.showMessage(f"🔄 {message}", 0)

    def _on_success(self, week_label: str, output: Any, config: Dict[str, Any]):
        """Handle successful completion of worker thread."""
        widget = self._create_widget_from_output(output)
        if widget is None:
            widget = self._default_preview_widget(config)

        self.center.show_widget(widget)
        self.progress_bar.hide()
        self.status.showMessage(f"✅ {week_label} completed successfully!", 5000)
        self.sidebar.run_btn.setEnabled(True)
        self.sidebar.run_btn.setText("▶ Run Analysis")

    def _on_error(self, week_label: str, error_msg: str):
        """Handle error from worker thread."""
        widget = self._error_widget(error_msg)
        self.center.show_widget(widget)
        self.progress_bar.hide()
        self.status.showMessage(f"❌ Error in {week_label}", 5000)
        self.sidebar.run_btn.setEnabled(True)
        self.sidebar.run_btn.setText("▶ Run Analysis")

    def _create_widget_from_output(self, output: Any) -> Optional[QtWidgets.QWidget]:
        if isinstance(output, QtWidgets.QWidget):
            tabs = self._extract_tabs_from_widget(output)
            return tabs or output

        tab_widget = self._build_tab_widget(output)
        if tab_widget is not None:
            return tab_widget

        if isinstance(output, Figure):
            return self._wrap_content("Figure", output)

        return None

    def _build_tab_widget(self, output: Any) -> Optional[QtWidgets.QWidget]:
        items: list[Tuple[str, Any]] = []

        if isinstance(output, dict):
            for title, content in output.get("figures", []):
                items.append((str(title), content))
            for title, content in output.get("widgets", []):
                items.append((str(title), content))
            for title, content in output.get("text", []):
                items.append((str(title), content))
        elif isinstance(output, (list, tuple)):
            for item in output:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    title, content = item
                    items.append((str(title), content))
        elif isinstance(output, Figure):
            items.append(("Figure", output))

        if not items:
            return None

        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444444;
                background-color: #2a2a2a;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #3a3a3a;
                color: #cccccc;
                padding: 8px 20px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #0d7377;
                color: white;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #4a4a4a;
            }
        """)
        added = False
        for idx, (title, content) in enumerate(items):
            wrapped = self._wrap_content(title, content)
            if wrapped is None:
                continue

            # Add icon based on content type
            if "Figure" in title or "Chart" in title or "Plot" in title:
                icon_prefix = "📊 "
            elif "Summary" in title or "Text" in title:
                icon_prefix = "📝 "
            else:
                icon_prefix = "📄 "

            tabs.addTab(wrapped, icon_prefix + title)
            added = True

        return tabs if added else None

    def _wrap_content(self, title: str, content: Any) -> Optional[QtWidgets.QWidget]:
        if isinstance(content, QtWidgets.QWidget):
            return content

        if isinstance(content, Figure):
            # Wrap matplotlib figure in scrollable area
            mpl_widget = MplWidget(figure=content)
            mpl_widget.draw()

            # Get figure size in inches and convert to pixels (assuming 100 DPI)
            fig_width, fig_height = content.get_size_inches()
            width_px = int(fig_width * 100)
            height_px = int(fig_height * 100)

            # Set minimum size for the widget to ensure full figure is visible
            mpl_widget.setMinimumSize(width_px, height_px)

            # Create scroll area
            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidget(mpl_widget)
            scroll_area.setWidgetResizable(False)  # Don't resize - allow scrolling

            # Set scroll bar policy (compatible with PyQt6 and PySide6)
            as_needed = SCROLLBAR_AS_NEEDED
            scroll_area.setHorizontalScrollBarPolicy(as_needed)
            scroll_area.setVerticalScrollBarPolicy(as_needed)
            scroll_area.setStyleSheet("""
                QScrollArea {
                    border: none;
                    background-color: #1e1e1e;
                }
            """)

            return scroll_area

        if isinstance(content, str):
            text = QtWidgets.QPlainTextEdit()
            text.setPlainText(content)
            text.setReadOnly(True)
            return text

        return None

    def _extract_tabs_from_widget(self, widget: QtWidgets.QWidget) -> Optional[QtWidgets.QWidget]:
        layout = widget.layout()
        if layout is None:
            return None

        children: List[QtWidgets.QWidget] = []
        for idx in range(layout.count()):
            item = layout.itemAt(idx)
            child = item.widget() if item is not None else None
            if child is not None:
                children.append(child)

        if len(children) <= 1:
            return None

        tabs = QtWidgets.QTabWidget()
        added = False
        for idx, child in enumerate(children, start=1):
            title = child.windowTitle() if hasattr(child, "windowTitle") else ""
            if not title:
                title = f"View {idx}"

            child.setParent(None)

            if isinstance(child, MplWidget):
                # Wrap MplWidget in scroll area
                scroll_area = QtWidgets.QScrollArea()
                scroll_area.setWidget(child)
                scroll_area.setWidgetResizable(False)

                # Set scroll bar policy (compatible with PyQt6 and PySide6)
                as_needed = SCROLLBAR_AS_NEEDED
                scroll_area.setHorizontalScrollBarPolicy(as_needed)
                scroll_area.setVerticalScrollBarPolicy(as_needed)
                scroll_area.setStyleSheet("""
                    QScrollArea {
                        border: none;
                        background-color: #1e1e1e;
                    }
                """)
                tabs.addTab(scroll_area, f"Figure {idx}")
                added = True
            elif isinstance(child, QtWidgets.QLabel):
                text_widget = QtWidgets.QPlainTextEdit()
                text_widget.setPlainText(child.text())
                text_widget.setReadOnly(True)
                tabs.addTab(text_widget, title if title else f"Text {idx}")
                added = True
            else:
                tabs.addTab(child, title)
                added = True

        return tabs if added else None

    def _error_widget(self, msg: str) -> QtWidgets.QWidget:
        """Create a styled error display widget."""
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(40, 40, 40, 40)

        # Error icon/title
        error_title = QtWidgets.QLabel("❌ Error Occurred")
        error_font = error_title.font()
        error_font.setPointSize(16)
        error_font.setBold(True)
        error_title.setFont(error_font)
        error_title.setStyleSheet("color: #e74c3c; padding: 10px;")
        align_center = ALIGN_CENTER
        error_title.setAlignment(align_center)

        # Error message in styled box
        error_text = QtWidgets.QPlainTextEdit()
        error_text.setPlainText(msg)
        error_text.setReadOnly(True)
        error_text.setStyleSheet("""
            QPlainTextEdit {
                background-color: #2a1515;
                border: 2px solid #e74c3c;
                border-radius: 5px;
                padding: 15px;
                font-family: 'Consolas', 'Courier New', monospace;
                color: #ffcccc;
            }
        """)

        layout.addWidget(error_title)
        layout.addWidget(error_text)
        return w

    def _default_preview_widget(self, config: Dict[str, Any]) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        # Table preview
        table = QtWidgets.QTableWidget()
        layout.addWidget(QtWidgets.QLabel("Preview (first 20 rows)"))
        layout.addWidget(table)

        # Matplotlib area
        mpl = MplWidget()
        layout.addWidget(mpl)

        # Load data and populate
        path = config.get("data_path")
        try:
            df = load_csv(path)
            head = df.head(20)
            table.setRowCount(len(head))
            table.setColumnCount(len(head.columns))
            table.setHorizontalHeaderLabels([str(c) for c in head.columns])
            for r in range(len(head)):
                for c in range(len(head.columns)):
                    val = head.iat[r, c]
                    item = QtWidgets.QTableWidgetItem(str(val))
                    table.setItem(r, c, item)

            # Simple demo plot for first two numeric columns
            ax = mpl.figure.add_subplot(111)
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if len(num_cols) >= 2:
                ax.plot(df[num_cols[0]], df[num_cols[1]], color="#4e79a7")
                ax.set_xlabel(num_cols[0])
                ax.set_ylabel(num_cols[1])
                ax.set_title("Demo plot (numeric columns)")
            else:
                ax.text(0.5, 0.5, "Not enough numeric columns to plot", ha="center", va="center")
            mpl.draw()
        except Exception as e:
            layout.insertWidget(0, QtWidgets.QLabel(f"Couldn't load data: {e}"))

        return w


def main() -> int:
    app = QtWidgets.QApplication([])
    if qdarktheme is not None:
        qdarktheme.setup_theme("dark")
    elif qdarkstyle is not None:
        qt_api = "pyqt6"
        try:
            from PySide6 import QtCore as _tmp  # type: ignore
            qt_api = "pyside6"
        except Exception:
            qt_api = "pyqt6"
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api=qt_api))
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
