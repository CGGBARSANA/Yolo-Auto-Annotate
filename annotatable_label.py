import sys
import os
import json
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QFileDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QMouseEvent
from PyQt5.QtCore import Qt, QRect


class AnnotatableLabel(QLabel):
    def __init__(self):
        super().__init__()
        # self.setPixmap()
        # self.image = None
        self.drawing_enabled = False
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.boxes = []
        self.setMouseTracking(True)

    def enable_drawing(self):
        self.drawing_enabled = True

    def disable_drawing(self):
        self.drawing_enabled = False

    def clear_boxes(self):
        self.boxes = []
        self.update()

    def mousePressEvents(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.drawing_enabled:
            self.drawing = True
            self.start_point = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing_enabled and self.drawing:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        try:
            if event.button() == Qt.LeftButton and self.drawing_enabled and self.drawing:
                self.drawing = False
                rect = QRect(self.start_point, self.end_point).normalized()
                self.boxes.append(rect)
                self.start_point = self.end_point = None
                self.update()
        except Exception as e:
            print("MouseReleaseEvent: ", e)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.pixmap() is None:
            return

        painter = QPainter(self)
        painter.setPen(QPen(Qt.green, 2))

        for rect in self.boxes:
            painter.drawRect(rect)

        if self.drawing_enabled and self.drawing and self.start_point and self.end_point:
            temp_rect = QRect(self.start_point, self.end_point).normalized()
            painter.drawRect(temp_rect)
