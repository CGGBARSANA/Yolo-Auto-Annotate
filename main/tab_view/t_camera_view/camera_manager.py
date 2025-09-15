import cv2


def detect_cameras_safe(available_cameras):
    """Safely detect cameras without crashing"""
    i = 0
    while True:
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Windows
        # cap = cv2.VideoCapture(i, cv2.CAP_V4L2)  # Linux
        # cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)  # macOS

        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
            i += 1
        else:
            break  # stop scanning as soon as a camera is missing

    return available_cameras


def populate_camera_combo(camera_combo, available_cameras, add_button, active_cameras):
    """Populate combo box with available cameras."""
    try:
        NO_CAMERAS = "No cameras detected"
        ALL_USED = "All cameras in use"

        camera_combo.clear()

        if not available_cameras:
            camera_combo.addItem(NO_CAMERAS)
            add_button.setEnabled(False)
            return

        available_to_add = [cam_id for cam_id in available_cameras if cam_id not in active_cameras]

        if not available_to_add:
            camera_combo.addItem(ALL_USED)
            add_button.setEnabled(False)
            return

        for cam_id in available_to_add:
            camera_combo.addItem(f"Camera {cam_id}", cam_id)
        add_button.setEnabled(True)

    except Exception as e:
        print("Camera Handle populate_camera_combo:",e)
