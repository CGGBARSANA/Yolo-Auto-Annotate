import os
import cv2
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, pyqtSignal


class ImageThumbnail(QWidget):
    clicked = pyqtSignal(int)

    def __init__(self, image_path, index, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.index = index
        self.is_annotated = False

        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

        # Create image label
        self.image_label = QLabel()
        self.image_label.setFixedSize(140, 140)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                border-radius: 3px;
                background-color: white;
            }
        """)

        # Create filename label - show full filename
        self.filename_label = QLabel()
        filename = os.path.basename(self.image_path)
        self.filename_label.setText(filename)
        self.filename_label.setAlignment(Qt.AlignCenter)
        self.filename_label.setWordWrap(True)

        # Calculate dynamic height based on filename length
        # Estimate 2 lines for every 25 characters
        lines_needed = max(1, (len(filename) + 24) // 25)
        filename_height = lines_needed * 16  # 16px per line approximately
        total_height = 140 + filename_height + 20  # image + filename + margins

        self.setFixedSize(150, total_height)

        # Set font for filename
        font = QFont()
        font.setPointSize(8)
        self.filename_label.setFont(font)
        self.filename_label.setStyleSheet("""
            QLabel {
                color: #333;
                background-color: transparent;
                border: none;
                padding: 2px;
            }
        """)

        layout.addWidget(self.image_label)
        layout.addWidget(self.filename_label)
        self.setLayout(layout)

        # Set overall widget style
        self.setStyleSheet("""
            ImageThumbnail {
                border: 2px solid #ccc;
                border-radius: 5px;
                background-color: white;
            }
            ImageThumbnail:hover {
                border-color: #0078d4;
            }
        """)

        self.load_thumbnail()

        # Keep full filename as tooltip
        self.setToolTip(os.path.basename(self.image_path))

    def load_thumbnail(self):
        img = cv2.imread(self.image_path)
        if img is not None:
            # Resize image for thumbnail
            h, w = img.shape[:2]
            aspect = w / h
            if aspect > 1:
                new_w, new_h = 130, int(130 / aspect)
            else:
                new_w, new_h = int(130 * aspect), 130

            img = cv2.resize(img, (new_w, new_h))
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.index)

    def set_current(self, is_current):
        self._is_current = is_current
        if is_current:
            self.setStyleSheet("""
                ImageThumbnail {
                    border: 3px solid #0078d4;
                    border-radius: 5px;
                    background-color: #e6f3ff;
                }
            """)
            self.filename_label.setStyleSheet("""
                QLabel {
                    color: #0078d4;
                    background-color: transparent;
                    border: none;
                    padding: 2px;
                    font-weight: bold;
                }
            """)
        else:
            color = "#00ff04" if self.is_annotated else "#ccc"
            self.setStyleSheet(f"""
                ImageThumbnail {{
                    border: 2px solid {color};
                    border-radius: 5px;
                    background-color: white;
                }}
                ImageThumbnail:hover {{
                    border-color: #0078d4;
                }}
            """)
            filename_color = "#00aa00" if self.is_annotated else "#333"
            font_weight = "bold" if self.is_annotated else "normal"
            self.filename_label.setStyleSheet(f"""
                QLabel {{
                    color: {filename_color};
                    background-color: transparent;
                    border: none;
                    padding: 2px;
                    font-weight: {font_weight};
                }}
            """)

    def set_annotated(self, is_annotated):
        self.is_annotated = is_annotated
        if not hasattr(self, '_is_current') or not self._is_current:
            self.set_current(False)