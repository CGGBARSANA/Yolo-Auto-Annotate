import os
import numpy as np
import json


def save_yolo_format(
        detections,
        image_shape,
        directory,
        filename_base,
        sub_dir="",
        selected_indices=None,
        box_labels=None,
        custom_label_manager=None,
        class_names=None,
        class_selection_mode=None
):
    """
    Save YOLO format annotations.

    Parameters:
        detections (list[dict]): List of detections, each with "box" and "cls".
        image_shape (tuple): (height, width) or (height, width, channels).
        directory (str): Base directory to save the manage_label_btn file.
        filename_base (str): File name without extension.
        sub_dir (str): Optional subdirectory.
        selected_indices (list[int]): Indices of detections to save (if filtering).
        box_labels (dict): Optional {index: custom_label} mapping.
        custom_label_manager: Optional object with get_class_id() method.
        class_names (dict): Mapping of class name to ID.
        class_selection_mode: Mode for custom manage_label_btn manager.

    Returns:
        bool: True if saved successfully, False otherwise.
    """
    try:
        h, w = image_shape[:2]
        label_file = os.path.join(directory, sub_dir, f"{filename_base}.txt")
        os.makedirs(os.path.dirname(label_file), exist_ok=True)

        # If filtering by selected indices
        if selected_indices is not None:
            det_list = [detections[i] for i in selected_indices]
        else:
            det_list = detections

        with open(label_file, "w") as f:
            for i, det in enumerate(det_list):
                x1, y1, x2, y2 = det["box"]

                # Determine class ID
                if box_labels and (i in box_labels) and custom_label_manager:
                    custom_label = box_labels[i]
                    cls = custom_label_manager.get_class_id(
                        custom_label,
                        class_names,
                        class_selection_mode
                    )
                else:
                    cls = det["cls"]

                # Convert to YOLO format (normalized)
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        return True
    except Exception as e:
        print(f"Error saving YOLO format: {e}")
        return False


def save_current_annotation(image_path, detections, selected_boxes, box_labels, original_shape, directory, file_name, class_selection_mode="both", sub_folder=""):
    """Save current annotation state to JSON file"""
    try:
        annotation_data = {
            "image_path": image_path,
            "detections": detections,
            "selected_boxes": selected_boxes,
            "box_labels": box_labels,
            "original_shape": original_shape,
            "timestamp": str(np.datetime64('now')),
            "class_selection_mode": class_selection_mode  # Save current mode
        }
        annotation_file = os.path.join(directory, sub_folder, f"{file_name}")
        with open(annotation_file, 'w') as f:
            json.dump(annotation_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving annotation: {e}")
        return False


def create_save_directory(save_directory):
    """Create directory structure for saving annotations"""
    try:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Create subdirectories
        subdirs = ["continuous_capture", "wrong_detections", "images", "labels", "yolo_labels"]
        for subdir in subdirs:
            path = os.path.join(save_directory, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
    except Exception as e:
        print(f"Error creating save directory: {e}")