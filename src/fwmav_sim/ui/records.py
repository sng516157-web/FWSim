"""Records browser (M6 Task 5): the outputs/ folder as a first-class view.

Newest-first list of run records with kind/status; double-click opens the
record (time series + 3D + validity via the main window callback);
right-click "compare with…" overlays the selected record's channels on the
scope (dashed) against the currently loaded one.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMenu,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
    _HAVE_QT = True
except ImportError:
    _HAVE_QT = False


def scan_records(out_root: Path) -> list[dict]:
    """Newest-first record summaries under out_root (dirs with metadata.json)."""
    out_root = Path(out_root)
    records = []
    if not out_root.is_dir():
        return records
    for meta_path in out_root.glob("*/metadata.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        ts_meta = meta.get("time_series_meta", {})
        records.append({
            "dir": meta_path.parent,
            "name": meta_path.parent.name,
            "kind": ts_meta.get("kind", "?"),
            "status": meta.get("status", ts_meta.get("status", "OK")),
            "mtime": meta_path.stat().st_mtime,
        })
    return sorted(records, key=lambda r: -r["mtime"])


class RecordsPanel(QWidget if _HAVE_QT else object):
    """outputs/ record list with open/compare actions."""

    def __init__(self, out_root: Path, on_open, on_compare, parent=None):
        if not _HAVE_QT:
            raise ImportError("PySide6 required for the records panel")
        super().__init__(parent)
        self._out_root = Path(out_root)
        self._on_open = on_open
        self._on_compare = on_compare
        self._records: list[dict] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QLabel("run records (double-click to open)")
        layout.addWidget(header)
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(self._open_item)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._context_menu)
        layout.addWidget(self.list)
        refresh = QPushButton("refresh")
        refresh.clicked.connect(self.refresh)
        layout.addWidget(refresh)
        self.refresh()

    def refresh(self) -> None:
        self._records = scan_records(self._out_root)
        self.list.clear()
        for rec in self._records:
            icon = "✓" if rec["status"] == "OK" else "✗"
            item = QListWidgetItem(f"{icon} {rec['name']}  [{rec['kind']}]")
            if rec["status"] != "OK":
                item.setForeground(Qt.GlobalColor.red)
            self.list.addItem(item)

    def _open_item(self, item):
        idx = self.list.row(item)
        if 0 <= idx < len(self._records):
            self._on_open(self._records[idx]["dir"])

    def _context_menu(self, pos):
        item = self.list.itemAt(pos)
        if item is None:
            return
        idx = self.list.row(item)
        menu = QMenu(self)
        open_act = menu.addAction("open")
        cmp_act = menu.addAction("compare with current record…")
        chosen = menu.exec(self.list.viewport().mapToGlobal(pos))
        if chosen is open_act:
            self._on_open(self._records[idx]["dir"])
        elif chosen is cmp_act:
            self._on_compare(self._records[idx]["dir"])
