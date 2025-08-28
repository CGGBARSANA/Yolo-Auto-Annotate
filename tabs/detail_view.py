from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea, QGridLayout, QGroupBox, \
    QButtonGroup, QRadioButton, QComboBox
from PyQt5.QtCore import Qt
from annotatable_label import AnnotatableLabel


class DetailView(QWidget):
    def __init__(self, parent_annotator=None):
        super().__init__()
        self.parent_annotator = parent_annotator
        self.worker = None
        self.image_label = AnnotatableLabel()
        self.setup_ui()


    def setup_ui(self):
        # Tab 2: Detail View
        detail_layout = QHBoxLayout()

        # Left panel for controls
        left_panel = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setMaximumWidth(300)
        left_widget.setLayout(left_panel)

        # Image info section
        info_group = QGroupBox("Image Info")
        info_layout = QVBoxLayout()
        self.status_label = QLabel("Load images to start")
        info_layout.addWidget(self.status_label)
        info_group.setLayout(info_layout)
        left_panel.addWidget(info_group)

        # Class Selection Mode section
        class_mode_group = QGroupBox("Class Selection Mode")
        class_mode_layout = QVBoxLayout()

        self.class_mode_group = QButtonGroup()
        self.model_only_radio = QRadioButton("Model Classes Only")
        self.custom_only_radio = QRadioButton("Custom Labels Only")
        self.both_radio = QRadioButton("Both Model & Custom")



        self.class_mode_group.addButton(self.model_only_radio, 0)
        self.class_mode_group.addButton(self.custom_only_radio, 1)
        self.class_mode_group.addButton(self.both_radio, 2)

        class_mode_layout.addWidget(self.model_only_radio)
        class_mode_layout.addWidget(self.custom_only_radio)
        class_mode_layout.addWidget(self.both_radio)
        class_mode_group.setLayout(class_mode_layout)
        left_panel.addWidget(class_mode_group)

        # Label assignment section
        self.label_combo = QComboBox()
        self.label_combo.setEditable(True)
        self.label_combo.setPlaceholderText("Select or enter label")
        self.assign_label_btn = QPushButton("Assign Label to Selected")

        # Custom label management buttons
        label_management_layout = QHBoxLayout()
        self.manage_labels_btn = QPushButton("Manage Labels")
        self.refresh_labels_btn = QPushButton("Refresh")
        label_management_layout.addWidget(self.manage_labels_btn)
        label_management_layout.addWidget(self.refresh_labels_btn)

        label_group = QGroupBox("Label Assignment")
        label_layout = QVBoxLayout()
        label_layout.addWidget(QLabel("Assign label to selected boxes:"))
        label_layout.addWidget(self.label_combo)
        label_layout.addWidget(self.assign_label_btn)
        label_layout.addLayout(label_management_layout)
        label_group.setLayout(label_layout)
        left_panel.addWidget(label_group)


        # Manual Selection
        self.manual_annotation_btn = QPushButton("Manual Annotation: OFF")
        self.manual_annotation_btn.setCheckable(True)
        self.manual_annotation_btn.setChecked(False)
        self.manual_annotation_enabled = False

        # self.remove_manual_annotation_btn = QPushButton("Remove Manual Annotation")
        self.reset_manual_annotation_btn = QPushButton("Reset Manual Annotation")
        self.save_manual_annotation_btn = QPushButton("Save Manual Annotation")
        self.save_manual_annotation_btn.setDisabled(True)

        manual_selection_group = QGroupBox("Manual Annotate Selection Controls")
        manual_selection_layout = QVBoxLayout()
        manual_selection_layout.addWidget(self.manual_annotation_btn)
        # manual_selection_layout.addWidget(self.remove_manual_annotation_btn)
        manual_selection_layout.addWidget(self.reset_manual_annotation_btn)
        manual_selection_layout.addWidget(self.save_manual_annotation_btn)
        manual_selection_group.setLayout(manual_selection_layout)
        left_panel.addWidget(manual_selection_group)


        # Auto Selection controls
        self.selection_count_label = QLabel("Selected: 0")
        self.select_all_btn = QPushButton("Select All")
        self.deselect_all_btn = QPushButton("Deselect All")
        self.remove_selected_annotation_btn = QPushButton("Remove Annotation")



        selection_group = QGroupBox("Selection Controls")
        selection_layout = QVBoxLayout()

        selection_layout.addWidget(self.selection_count_label)
        selection_layout.addWidget(self.select_all_btn)
        selection_layout.addWidget(self.deselect_all_btn)
        selection_layout.addWidget(self.remove_selected_annotation_btn)
        selection_group.setLayout(selection_layout)
        left_panel.addWidget(selection_group)

        # Navigation controls
        self.next_btn = QPushButton("Next Image")
        self.prev_btn = QPushButton("Previous Image")
        self.save_btn = QPushButton("Save Annotation")
        self.auto_save_btn = QPushButton("Auto Save: ON")
        self.auto_save_btn.setCheckable(True)
        self.auto_save_btn.setChecked(True)
        self.auto_save_enabled = True

        nav_group = QGroupBox("Navigation & Save")
        nav_layout = QVBoxLayout()
        nav_btn_layout = QHBoxLayout()
        nav_btn_layout.addWidget(self.prev_btn)
        nav_btn_layout.addWidget(self.next_btn)
        nav_layout.addLayout(nav_btn_layout)
        nav_layout.addWidget(self.save_btn)
        nav_layout.addWidget(self.auto_save_btn)
        nav_group.setLayout(nav_layout)
        left_panel.addWidget(nav_group)

        # Session management
        session_group = QGroupBox("Session Management")
        session_layout = QVBoxLayout()

        self.clear_session_btn = QPushButton("Clear All Annotations")

        session_layout.addWidget(self.clear_session_btn)
        # session_layout.addWidget(self.export_annotations_btn)
        session_group.setLayout(session_layout)
        left_panel.addWidget(session_group)

        left_panel.addStretch()

        # Right panel for image display
        # self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.image_label)

        detail_layout.addWidget(left_widget)
        detail_layout.addLayout(right_layout)
        self.setLayout(detail_layout)
