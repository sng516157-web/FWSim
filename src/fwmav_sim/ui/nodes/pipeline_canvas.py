"""Pipeline DAG editor canvas (M8 Task 2).

A QGraphicsScene of block nodes with typed ports. Edges are created by
dragging output -> input; the connection is checked against the
control/pipeline BLOCK_REGISTRY (signal type + shape) BEFORE the edge is
accepted — the editor cannot produce a graph the executor would reject.
"Compile" produces the pipeline YAML dict; "run" hands it to the
experiment runner via the experiment's `pipeline:` field.

Rendering/interaction is the human check; graph ops are methods on
PipelineScene so they can be tested offscreen.
"""

from __future__ import annotations

from typing import Any

from fwmav_sim.control.pipeline import BLOCK_REGISTRY, Pipeline, PipelineError, BlockInstance

try:
    from PySide6.QtCore import Qt, QRectF
    from PySide6.QtGui import QColor, QPen, QBrush, QPainter
    from PySide6.QtWidgets import (
        QGraphicsEllipseItem,
        QGraphicsItem,
        QGraphicsLineItem,
        QGraphicsRectItem,
        QGraphicsScene,
        QGraphicsSimpleTextItem,
        QGraphicsView,
        QWidget,
    )
    _HAVE_QT = True
except ImportError:  # pragma: no cover
    _HAVE_QT = False


# ---------------------------------------------------------------------------
# Qt-free graph model (the testable core)
# ---------------------------------------------------------------------------


class PipelineGraph:
    """A pipeline DAG under construction: blocks + wirings, validated against
    the same registry as the executor."""

    def __init__(self):
        self.blocks: dict[str, dict[str, Any]] = {}  # id -> {type, params, pos}

    def add_block(self, block_id: str, block_type: str,
                  params: dict | None = None, pos=(0.0, 0.0)) -> None:
        if block_id in self.blocks:
            raise PipelineError(f"duplicate block id {block_id!r}")
        if block_type not in BLOCK_REGISTRY:
            raise PipelineError(f"unknown block type {block_type!r}")
        self.blocks[block_id] = {"type": block_type, "params": params or {},
                                 "pos": tuple(pos), "wirings": {}}

    def remove_block(self, block_id: str) -> None:
        if block_id not in self.blocks:
            raise PipelineError(f"no block {block_id!r}")
        del self.blocks[block_id]
        for b in self.blocks.values():
            b["wirings"] = {p: r for p, r in b["wirings"].items()
                            if not r.startswith(f"{block_id}.")}

    def connect(self, dst_id: str, dst_port: str, src_id: str, src_port: str) -> None:
        """Wire src.src_port -> dst.dst_port after registry type checks."""
        if dst_id not in self.blocks or src_id not in self.blocks:
            raise PipelineError("connect: unknown block id")
        dst_spec = BLOCK_REGISTRY[self.blocks[dst_id]["type"]]
        src_spec = BLOCK_REGISTRY[self.blocks[src_id]["type"]]
        if dst_port not in dst_spec.inputs:
            raise PipelineError(f"{dst_id}: no input port {dst_port!r}")
        if src_port not in src_spec.outputs:
            raise PipelineError(f"{src_id}: no output port {src_port!r}")
        want, got = dst_spec.inputs[dst_port], src_spec.outputs[src_port]
        if want.signal != got.signal:
            raise PipelineError(
                f"type mismatch: {dst_port} expects {want.signal!r}, "
                f"{src_id}.{src_port} produces {got.signal!r}")
        if want.shape is not None and got.shape is not None and want.shape != got.shape:
            raise PipelineError(
                f"shape mismatch: {dst_port} wants {want.shape}, got {got.shape}")
        self.blocks[dst_id]["wirings"][dst_port] = f"{src_id}.{src_port}"

    def disconnect(self, dst_id: str, dst_port: str) -> None:
        self.blocks[dst_id]["wirings"].pop(dst_port, None)

    def compile(self) -> list[BlockInstance]:
        """Compile to BlockInstance list; Pipeline() runs full validation
        (cycles, unwired ports)."""
        blocks = [
            BlockInstance.model_validate({
                "id": bid, "type": b["type"], "params": b["params"],
                "inputs": [{"port": p, "from": r} for p, r in b["wirings"].items()]})
            for bid, b in self.blocks.items()
        ]
        Pipeline(blocks)  # raises loudly on any residual problem
        return blocks

    def to_yaml_dict(self) -> dict:
        return {"pipeline": [
            {"id": bid, "type": b["type"], "params": b["params"],
             "inputs": [{"port": p, "from": r} for p, r in b["wirings"].items()]}
            for bid, b in self.blocks.items()]}

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "PipelineGraph":
        g = cls()
        for b in raw.get("pipeline", []):
            g.add_block(b["id"], b["type"], b.get("params", {}))
        for b in raw.get("pipeline", []):
            for inp in b.get("inputs", []):
                src, sport = inp["from"].split(".", 1)
                g.connect(b["id"], inp["port"], src, sport)
        return g


# ---------------------------------------------------------------------------
# Qt scene (thin rendering layer over PipelineGraph)
# ---------------------------------------------------------------------------

if _HAVE_QT:

    _PORT_R = 5.0

    class _Port(QGraphicsEllipseItem):
        def __init__(self, parent_node, name: str, signal: str, is_input: bool):
            super().__init__(-_PORT_R, -_PORT_R, 2 * _PORT_R, 2 * _PORT_R, parent_node)
            self.node, self.name, self.signal, self.is_input = parent_node, name, signal, is_input
            self.setBrush(QBrush(QColor("#e0a030" if is_input else "#30a0e0")))
            self.setToolTip(f"{name} [{signal}]")

    class _Node(QGraphicsRectItem):
        W, H = 150.0, 30.0

        def __init__(self, block_id: str, block_type: str):
            super().__init__(0, 0, self.W, self.H)
            self.block_id = block_id
            self.setFlag(QGraphicsItem.ItemIsMovable)
            self.setFlag(QGraphicsItem.ItemIsSelectable)
            self.setBrush(QBrush(QColor("#2d2d30")))
            self.setPen(QPen(QColor("#888"), 1))
            label = QGraphicsSimpleTextItem(f"{block_id}\n{block_type}", self)
            label.setBrush(QBrush(QColor("#ddd")))
            spec = BLOCK_REGISTRY[block_type]
            self.inputs: dict[str, _Port] = {}
            self.outputs: dict[str, _Port] = {}
            n_in, n_out = max(len(spec.inputs), 1), max(len(spec.outputs), 1)
            for i, (pname, port) in enumerate(spec.inputs.items()):
                p = _Port(self, pname, port.signal, True)
                p.setPos(0, self.H * (i + 1) / (n_in + 1))
                self.inputs[pname] = p
            for i, (pname, port) in enumerate(spec.outputs.items()):
                p = _Port(self, pname, port.signal, False)
                p.setPos(self.W, self.H * (i + 1) / (n_out + 1))
                self.outputs[pname] = p

    class _Edge(QGraphicsLineItem):
        def __init__(self, src_port: _Port, dst_port: _Port):
            super().__init__()
            self.src, self.dst = src_port, dst_port
            self.setPen(QPen(QColor("#7ec97e"), 2))
            self.refresh()

        def refresh(self):
            a = self.src.sceneBoundingRect().center()
            b = self.dst.sceneBoundingRect().center()
            self.setLine(a.x(), a.y(), b.x(), b.y())

    class PipelineScene(QGraphicsScene):
        """Graphics scene bound to a PipelineGraph (the model)."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.graph = PipelineGraph()
            self._nodes: dict[str, _Node] = {}
            self._edges: list[_Edge] = []
            self._drag_src: _Port | None = None
            self._drag_line: QGraphicsLineItem | None = None

        # graph ops (offscreen-testable) ---------------------------------

        def add_block(self, block_id: str, block_type: str, params=None, pos=(0, 0)):
            self.graph.add_block(block_id, block_type, params, pos)
            node = _Node(block_id, block_type)
            node.setPos(*pos)
            self.addItem(node)
            self._nodes[block_id] = node

        def connect(self, dst_id, dst_port, src_id, src_port):
            self.graph.connect(dst_id, dst_port, src_id, src_port)
            e = _Edge(self._nodes[src_id].outputs[src_port],
                      self._nodes[dst_id].inputs[dst_port])
            self.addItem(e)
            self._edges.append(e)

        def compile(self):
            return self.graph.compile()

        # interaction -----------------------------------------------------

        def mousePressEvent(self, ev):
            item = self.itemAt(ev.scenePos(), self.views()[0].transform())
            if isinstance(item, _Port) and not item.is_input:
                self._drag_src = item
                self._drag_line = QGraphicsLineItem()
                self._drag_line.setPen(QPen(QColor("#7ec97e"), 2, Qt.DashLine))
                self.addItem(self._drag_line)
                p = item.sceneBoundingRect().center()
                self._drag_line.setLine(p.x(), p.y(), p.x(), p.y())
                ev.accept()
                return
            super().mousePressEvent(ev)

        def mouseMoveEvent(self, ev):
            if self._drag_line is not None:
                p = self._drag_src.sceneBoundingRect().center()
                self._drag_line.setLine(p.x(), p.y(), ev.scenePos().x(), ev.scenePos().y())
                return
            super().mouseMoveEvent(ev)
            for e in self._edges:
                e.refresh()

        def mouseReleaseEvent(self, ev):
            if self._drag_src is not None:
                item = self.itemAt(ev.scenePos(), self.views()[0].transform())
                if isinstance(item, _Port) and item.is_input:
                    try:
                        self.connect(item.node.block_id, item.name,
                                     self._drag_src.node.block_id, self._drag_src.name)
                    except PipelineError as exc:
                        # rejected connections are visible, not silent
                        from PySide6.QtWidgets import QMessageBox
                        QMessageBox.warning(None, "connection rejected", str(exc))
                self.removeItem(self._drag_line)
                self._drag_line = None
                self._drag_src = None
                return
            super().mouseReleaseEvent(ev)

    class PipelineCanvas(QGraphicsView):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setScene(PipelineScene(self))
            self.setRenderHint(QPainter.Antialiasing)
            self.setDragMode(QGraphicsView.RubberBandDrag)

        def save_yaml(self, path) -> None:
            """Compile + validate, then write the pipeline YAML (fails loudly
            on invalid graphs — nothing half-built is saved)."""
            import yaml as _yaml

            self.scene().compile()
            with open(path, "w", encoding="utf-8") as fh:
                _yaml.safe_dump(self.scene().graph.to_yaml_dict(), fh,
                                sort_keys=False)

    class PipelineEditorWidget(QWidget):
        """Canvas + a minimal toolbar: add block, save YAML."""

        def __init__(self, parent=None):
            super().__init__(parent)
            from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLineEdit,
                                           QPushButton, QVBoxLayout)
            lay = QVBoxLayout(self)
            bar = QHBoxLayout()
            self.block_type = QComboBox()
            self.block_type.addItems(sorted(BLOCK_REGISTRY))
            self.block_id = QLineEdit()
            self.block_id.setPlaceholderText("block id")
            add = QPushButton("add block")
            save = QPushButton("save pipeline YAML…")
            bar.addWidget(self.block_type)
            bar.addWidget(self.block_id)
            bar.addWidget(add)
            bar.addWidget(save)
            lay.addLayout(bar)
            self.canvas = PipelineCanvas()
            lay.addWidget(self.canvas)
            add.clicked.connect(self._add)
            save.clicked.connect(self._save)

        def _add(self):
            from PySide6.QtWidgets import QMessageBox

            bid = self.block_id.text().strip()
            if not bid:
                QMessageBox.warning(self, "add block", "give the block an id first")
                return
            try:
                self.canvas.scene().add_block(bid, self.block_type.currentText())
            except PipelineError as exc:
                QMessageBox.warning(self, "add block", str(exc))

        def _save(self):
            from PySide6.QtWidgets import QFileDialog, QMessageBox

            path, _ = QFileDialog.getSaveFileName(self, "save pipeline YAML",
                                                  "experiments", "YAML (*.yaml)")
            if not path:
                return
            try:
                self.canvas.save_yaml(path)
            except PipelineError as exc:
                QMessageBox.warning(self, "pipeline invalid", str(exc))
