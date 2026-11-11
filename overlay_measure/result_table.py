from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QMenu,
    QScrollBar,
    QTableWidget,
)


class ResultTableWidget(QTableWidget):
    """QTableWidget with consistent copy, scrolling and result deletion UX."""

    deleteRequested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._row_payloads: list[dict] = []
        self._delete_allowed = False
        self.setObjectName("resultTable")
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.horizontalHeader().setStretchLastSection(False)
        self.external_horizontal_bar = QScrollBar(Qt.Horizontal)
        self.external_horizontal_bar.setObjectName("resultHorizontalBar")
        self.external_horizontal_bar.setFixedHeight(17)
        self.external_horizontal_bar.setStyleSheet(
            "QScrollBar:horizontal {height: 14px; margin: 1px 0 0 0; background: #EEF1F4; "
            "border: 1px solid #D8DEE6; border-radius: 6px;}"
            "QScrollBar::handle:horizontal {min-width: 44px; background: #7F8B99; border-radius: 5px;}"
            "QScrollBar::handle:horizontal:hover {background: #526173;}"
            "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {width: 0px;}"
        )
        internal_bar = self.horizontalScrollBar()
        internal_bar.rangeChanged.connect(self._sync_external_scroll_range)
        internal_bar.valueChanged.connect(self.external_horizontal_bar.setValue)
        self.external_horizontal_bar.valueChanged.connect(internal_bar.setValue)
        self._sync_external_scroll_range(internal_bar.minimum(), internal_bar.maximum())

    def _sync_external_scroll_range(self, minimum: int, maximum: int):
        bar = self.external_horizontal_bar
        bar.blockSignals(True)
        bar.setRange(minimum, maximum)
        bar.setPageStep(self.horizontalScrollBar().pageStep())
        bar.setValue(self.horizontalScrollBar().value())
        bar.blockSignals(False)
        bar.setEnabled(maximum > minimum)
        # A parentless bar would pop up as a top-level window when shown.
        bar.setVisible(maximum > minimum and bar.parentWidget() is not None)

    def fit_result_columns(self):
        """Size columns from complete cell text so overflow remains scrollable."""
        self.resizeColumnsToContents()
        metrics = self.fontMetrics()
        for column in range(self.columnCount()):
            header_item = self.horizontalHeaderItem(column)
            texts = [header_item.text()] if header_item is not None else []
            texts.extend(
                self.item(row, column).text()
                for row in range(self.rowCount())
                if self.item(row, column) is not None
            )
            desired = max((metrics.horizontalAdvance(text) for text in texts), default=48) + 28
            self.setColumnWidth(column, max(self.columnWidth(column), min(desired, 1600)))
            for row in range(self.rowCount()):
                item = self.item(row, column)
                if item is not None:
                    item.setToolTip(item.text())
        self._sync_external_scroll_range(
            self.horizontalScrollBar().minimum(), self.horizontalScrollBar().maximum()
        )

    def set_row_payloads(self, payloads: Iterable[dict] | None):
        self._row_payloads = [dict(item or {}) for item in (payloads or [])]

    def set_delete_allowed(self, allowed: bool):
        self._delete_allowed = bool(allowed)

    def row_payload(self, row: int) -> dict:
        if 0 <= row < len(self._row_payloads):
            return dict(self._row_payloads[row])
        return {}

    def selected_row_payloads(self) -> list[dict]:
        rows = sorted({index.row() for index in self.selectedIndexes()})
        payloads = []
        seen = set()
        for row in rows:
            payload = self.row_payload(row)
            if not payload:
                continue
            identity = tuple(sorted((str(key), repr(value)) for key, value in payload.items()))
            if identity in seen:
                continue
            seen.add(identity)
            payloads.append(payload)
        return payloads

    def copy_selection(self, include_headers: bool = False):
        indexes = self.selectedIndexes()
        if not indexes:
            return
        selected = {(index.row(), index.column()) for index in indexes}
        rows = range(min(row for row, _ in selected), max(row for row, _ in selected) + 1)
        columns = range(min(column for _, column in selected), max(column for _, column in selected) + 1)
        lines = []
        if include_headers:
            lines.append("\t".join(
                self.horizontalHeaderItem(column).text() if self.horizontalHeaderItem(column) else ""
                for column in columns
            ))
        for row in rows:
            values = []
            for column in columns:
                item = self.item(row, column)
                if (row, column) in selected and item is not None:
                    values.append(item.text().replace("\r", " ").replace("\n", " ").replace("\t", " "))
                else:
                    values.append("")
            lines.append("\t".join(values))
        QApplication.clipboard().setText("\n".join(lines))

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            self.copy_selection(False)
            event.accept()
            return
        if event.matches(QKeySequence.SelectAll):
            self.selectAll()
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ShiftModifier:
            delta = event.angleDelta().y() or event.angleDelta().x()
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() - delta)
            event.accept()
            return
        super().wheelEvent(event)

    def contextMenuEvent(self, event):
        index = self.indexAt(event.pos())
        if index.isValid() and index not in self.selectedIndexes():
            self.clearSelection()
            self.selectRow(index.row())
            self.setCurrentCell(index.row(), index.column())

        menu = QMenu(self)
        copy_action = menu.addAction("复制选中内容")
        copy_headers_action = menu.addAction("复制选中内容（含表头）")
        menu.addSeparator()
        select_all_action = menu.addAction("全选")
        menu.addSeparator()

        payloads = self.selected_row_payloads()
        deletable = [item for item in payloads if item.get("deletable", True)]
        if len(deletable) > 1:
            delete_action = menu.addAction(f"删除选中结果（{len(deletable)}项）")
        else:
            delete_action = menu.addAction("删除当前结果")
        delete_action.setEnabled(self._delete_allowed and bool(deletable))
        if not self._delete_allowed:
            delete_action.setToolTip("仅工程模式允许删除结果")
            delete_action.setStatusTip("仅工程模式允许删除结果")
        elif payloads and not deletable:
            delete_action.setToolTip("统计或派生结果不能直接删除")

        chosen = menu.exec(event.globalPos())
        if chosen == copy_action:
            self.copy_selection(False)
        elif chosen == copy_headers_action:
            self.copy_selection(True)
        elif chosen == select_all_action:
            self.selectAll()
        elif chosen == delete_action and deletable:
            self.deleteRequested.emit(deletable)
