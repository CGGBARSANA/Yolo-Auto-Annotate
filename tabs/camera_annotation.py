import cv2
import math
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QComboBox, QGridLayout, QFrame, QMessageBox
)


class CameraAnnotation(QWidget):
    def __init__(self, parent_annotator=None):
        super().__init__()
        self.parent_annotator = parent_annotator
        self.available_cameras = self.detect_cameras()
        self.camera_feeds = {}  # {camera_id: (cap, QLabel)}
        self.timers = {}
        self.active_cameras = []  # Track order of added cameras
        self.setup_ui()

    def setup_ui(self):
        camera_layout = QHBoxLayout()

        # Left panel
        left_panel = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setMaximumWidth(300)
        left_widget.setLayout(left_panel)

        # GroupBox for camera selection
        camera_selection = QGroupBox("Add Cameras")
        selection_layout = QVBoxLayout()

        # Combo box for camera selection
        combo_layout = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.populate_camera_combo()

        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add_selected_camera)

        combo_layout.addWidget(self.camera_combo)
        combo_layout.addWidget(self.add_button)
        selection_layout.addLayout(combo_layout)

        # Refresh button to detect cameras again
        self.refresh_button = QPushButton("Refresh Cameras")
        self.refresh_button.clicked.connect(self.refresh_cameras)
        selection_layout.addWidget(self.refresh_button)

        camera_selection.setLayout(selection_layout)
        left_panel.addWidget(camera_selection)

        # GroupBox for active cameras
        active_cameras = QGroupBox("Active Cameras")
        active_layout = QVBoxLayout()

        self.active_cameras_list = QLabel("No cameras active")
        active_layout.addWidget(self.active_cameras_list)

        self.remove_all_button = QPushButton("Remove All Cameras")
        self.remove_all_button.clicked.connect(self.remove_all_cameras)
        self.remove_all_button.setEnabled(False)
        active_layout.addWidget(self.remove_all_button)

        active_cameras.setLayout(active_layout)
        left_panel.addWidget(active_cameras)

        self.camera_status = QLabel("Select a camera and click Add to start")
        left_panel.addWidget(self.camera_status)

        # Add stretch to push everything to top
        left_panel.addStretch()

        # Right panel for feeds
        self.feed_layout = QGridLayout()
        self.feed_layout.setSpacing(10)
        self.feed_container = QWidget()
        self.feed_container.setLayout(self.feed_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.feed_container)

        camera_layout.addWidget(left_widget)
        camera_layout.addWidget(scroll_area)

        self.setLayout(camera_layout)

    def detect_cameras(self, max_tested=10):
        """Detect available cameras"""
        available = []
        for i in range(max_tested):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def populate_camera_combo(self):
        try:
            """Populate combo box with available cameras"""
            self.camera_combo.clear()
            if not self.available_cameras:
                self.camera_combo.addItem("No cameras detected")
                self.add_button.setEnabled(False)
            else:
                for cam_id in self.available_cameras:
                    if cam_id not in self.active_cameras:
                        self.camera_combo.addItem(f"Camera {cam_id}", cam_id)

                if self.camera_combo.count() == 0:
                    self.camera_combo.addItem("All cameras in use")
                    self.add_button.setEnabled(False)
                else:
                    self.add_button.setEnabled(True)
        except Exception as e:
            pass


    def refresh_cameras(self):
        """Refresh the list of available cameras"""
        self.available_cameras = self.detect_cameras()
        self.populate_camera_combo()
        self.camera_status.setText(f"Found {len(self.available_cameras)} cameras")

    def add_selected_camera(self):
        """Add the selected camera from combo box"""
        if self.camera_combo.count() == 0 or not self.add_button.isEnabled():
            return

        cam_id = self.camera_combo.currentData()
        if cam_id is None:
            return

        if cam_id in self.active_cameras:
            QMessageBox.warning(self, "Warning", f"Camera {cam_id} is already active!")
            return

        self.start_camera(cam_id)

    def start_camera(self, cam_id):
        """Start a camera feed"""
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            self.camera_status.setText(f"Failed to open camera {cam_id}")
            QMessageBox.error(self, "Error", f"Failed to open camera {cam_id}")
            return

        # Create QLabel to display feed with title
        feed_widget = QFrame()
        feed_widget.setFrameStyle(QFrame.Box)
        feed_layout = QVBoxLayout()

        title_label = QLabel(f"Camera {cam_id}")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; padding: 5px;")

        video_label = QLabel()
        video_label.setMinimumSize(320, 240)
        video_label.setStyleSheet("border: 1px solid gray;")
        video_label.setAlignment(Qt.AlignCenter)
        video_label.setText("Loading...")

        # Remove button for individual camera
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(lambda: self.remove_camera(cam_id))

        feed_layout.addWidget(title_label)
        feed_layout.addWidget(video_label)
        feed_layout.addWidget(remove_button)
        feed_widget.setLayout(feed_layout)

        # Calculate grid position
        num_cameras = len(self.active_cameras)
        cols = math.ceil(math.sqrt(num_cameras + 1))  # +1 for the new camera
        rows = math.ceil((num_cameras + 1) / cols)

        # Reorganize all cameras in grid
        self.reorganize_grid()

        # Add new camera to grid
        row = num_cameras // cols
        col = num_cameras % cols
        self.feed_layout.addWidget(feed_widget, row, col)

        # Store camera data
        self.camera_feeds[cam_id] = (cap, video_label, feed_widget)
        self.active_cameras.append(cam_id)

        # Setup timer for this camera
        timer = QTimer(self)
        timer.timeout.connect(lambda: self.update_frame(cam_id))
        timer.start(33)  # ~30 FPS

        self.timers[cam_id] = timer
        self.camera_status.setText(f"Camera {cam_id} started ({len(self.active_cameras)} total)")

        # Update UI
        self.update_active_cameras_display()
        self.populate_camera_combo()  # Refresh combo box
        self.remove_all_button.setEnabled(True)

    def remove_camera(self, cam_id):
        """Remove a specific camera"""
        if cam_id in self.camera_feeds:
            cap, video_label, feed_widget = self.camera_feeds.pop(cam_id)
            cap.release()
            feed_widget.deleteLater()

        if cam_id in self.timers:
            self.timers[cam_id].stop()
            del self.timers[cam_id]

        if cam_id in self.active_cameras:
            self.active_cameras.remove(cam_id)

        self.camera_status.setText(f"Camera {cam_id} removed ({len(self.active_cameras)} remaining)")

        # Reorganize remaining cameras
        self.reorganize_grid()

        # Update UI
        self.update_active_cameras_display()
        self.populate_camera_combo()  # Refresh combo box

        if len(self.active_cameras) == 0:
            self.remove_all_button.setEnabled(False)

    def remove_all_cameras(self):
        """Remove all active cameras"""
        cameras_to_remove = self.active_cameras.copy()
        for cam_id in cameras_to_remove:
            self.remove_camera(cam_id)

    def reorganize_grid(self):
        """Reorganize cameras in optimal grid layout"""
        if not self.active_cameras:
            return

        # Clear current layout
        for i in reversed(range(self.feed_layout.count())):
            self.feed_layout.itemAt(i).widget().setParent(None)

        # Calculate optimal grid dimensions
        num_cameras = len(self.active_cameras)
        cols = math.ceil(math.sqrt(num_cameras))

        # Redistribute cameras
        for i, cam_id in enumerate(self.active_cameras):
            if cam_id in self.camera_feeds:
                _, _, feed_widget = self.camera_feeds[cam_id]
                row = i // cols
                col = i % cols
                self.feed_layout.addWidget(feed_widget, row, col)

    def update_active_cameras_display(self):
        """Update the active cameras list display"""
        if not self.active_cameras:
            self.active_cameras_list.setText("No cameras active")
        else:
            camera_list = ", ".join([f"Camera {cam_id}" for cam_id in self.active_cameras])
            self.active_cameras_list.setText(f"Active: {camera_list}")

    def update_frame(self, cam_id):
        """Update frame for a specific camera"""
        if cam_id in self.camera_feeds:
            cap, video_label, _ = self.camera_feeds[cam_id]
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                video_label.setPixmap(pixmap.scaled(video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                video_label.setText(f"Camera {cam_id} disconnected")

    def closeEvent(self, event):
        """Clean up when closing the application"""
        self.remove_all_cameras()
        event.accept()


# if __name__ == "__main__":
#     from PyQt5.QtWidgets import QApplication
#     import sys
#
#     app = QApplication(sys.argv)
#     win = CameraAnnotation()
#     win.setWindowTitle("Multi-Camera Viewer")
#     win.resize(1200, 800)
#     win.show()
#     sys.exit(app.exec_())