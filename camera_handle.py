import cv2


def detect_cameras_safe(available_cameras):
    """Safely detect cameras without crashing"""
    try:
        for i in range(5):  # Reduced range to prevent crashes
            try:
                cap = cv2.VideoCapture(i)
                if cap is not None and cap.isOpened():
                    # Test if we can read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available_cameras.append(i)
                cap.release()
            except Exception as e:
                print(f"Error testing camera {i}: {e}")
                continue
        return available_cameras
    except Exception as e:
        print(f"Camera detection error: {e}")


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
