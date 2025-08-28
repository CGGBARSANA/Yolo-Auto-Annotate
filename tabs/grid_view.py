from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea, QGridLayout
from PyQt5.QtCore import Qt


class GridView(QWidget):
    def __init__(self, parent_annotator=None):
        super().__init__()
        self.parent_annotator = parent_annotator
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        # Tab 1: Grid View
        grid_layout = QVBoxLayout()

        # Grid controls
        grid_controls = QHBoxLayout()
        self.add_images_btn = QPushButton("Import images")
        grid_controls.addWidget(self.add_images_btn)
        grid_controls.addStretch()
        grid_layout.addLayout(grid_controls)

        self.remove_all_btn = QPushButton("Remove all images")
        grid_controls.addWidget(self.remove_all_btn)

        self.export_annotations_btn = QPushButton("Export all annotations")
        grid_controls.addWidget(self.export_annotations_btn)
        separator = QLabel("|")
        separator.setStyleSheet("color: gray; font-weight: bold;")
        grid_controls.addWidget(separator)

        # Thumbnail selection controls
        self.thumbnail_selection_label = QLabel("Selected: 0 / 0")
        grid_controls.addWidget(self.thumbnail_selection_label)

        self.select_all_thumbnails_btn = QPushButton("Select All")
        grid_controls.addWidget(self.select_all_thumbnails_btn)

        self.deselect_all_thumbnails_btn = QPushButton("Deselect All")
        grid_controls.addWidget(self.deselect_all_thumbnails_btn)

        self.remove_selected_btn = QPushButton("Remove Selected")
        self.remove_selected_btn.setStyleSheet("QPushButton { background-color: #ffcccc; }")
        grid_controls.addWidget(self.remove_selected_btn)

        grid_controls.addStretch()
        grid_layout.addLayout(grid_controls)

        # Scrollable grid area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(10)
        scroll_area.setWidget(self.grid_widget)

        grid_layout.addWidget(scroll_area)
        self.setLayout(grid_layout)
        # tabWidget.addTab(grid_tab, "Grid View")