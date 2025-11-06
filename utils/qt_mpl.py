from __future__ import annotations

from typing import Optional

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
try:  # Prefer PySide6 if available
    from PySide6 import QtWidgets  # type: ignore
except Exception:  # Fallback to PyQt6
    from PyQt6 import QtWidgets  # type: ignore


class MplWidget(QtWidgets.QWidget):
    """A QWidget that contains a Matplotlib Figure and toolbar."""

    def __init__(self, figure: Optional[Figure] = None, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.figure = figure or Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def draw(self):
        self.canvas.draw_idle()
