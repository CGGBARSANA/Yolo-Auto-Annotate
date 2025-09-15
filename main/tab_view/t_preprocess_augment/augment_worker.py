import cv2
import os
from PyQt5.QtCore import QThread, pyqtSignal


class AugmentationWorker(QThread):
    """Worker thread for processing augmentations"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(int, int)  # success_count, total_count
    error = pyqtSignal(str)

    def __init__(self, images_data, augmentation_pipeline, preprocess_pipeline, output_dir, augment_count):
        super().__init__()
        self.images_data = images_data
        self.augmentation_pipeline = augmentation_pipeline
        self.preprocess_pipeline = preprocess_pipeline
        self.output_dir = output_dir
        self.augment_count = augment_count

    def run(self):
        try:
            success_count = 0
            total_operations = len(self.images_data) * (1 + self.augment_count)  # original + augmented
            current_operation = 0

            for img_data in self.images_data:
                try:
                    # Process original image with preprocessing only
                    processed_img, processed_annotations = self.process_image(
                        img_data, self.preprocess_pipeline, "preprocessed"
                    )
                    if processed_img is not None:
                        success_count += 1

                    current_operation += 1
                    self.progress.emit(int((current_operation / total_operations) * 100))

                    # Generate augmented versions
                    for aug_idx in range(self.augment_count):
                        augmented_img, augmented_annotations = self.process_image(
                            img_data, self.augmentation_pipeline, f"aug_{aug_idx}"
                        )
                        if augmented_img is not None:
                            success_count += 1

                        current_operation += 1
                        self.progress.emit(int((current_operation / total_operations) * 100))

                except Exception as e:
                    print(f"Error processing image {img_data['path']}: {e}")
                    current_operation += (1 + self.augment_count)
                    self.progress.emit(int((current_operation / total_operations) * 100))

            self.finished.emit(success_count, len(self.images_data) * (1 + self.augment_count))

        except Exception as e:
            self.error.emit(str(e))

    def process_image(self, img_data, pipeline, suffix):
        """Process a single image with the given pipeline"""
        try:
            image = cv2.imread(img_data['path'])
            if image is None:
                return None, None

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Prepare bounding boxes for albumentations format
            bboxes = []
            class_labels = []

            for bbox_data in img_data['annotations']:
                # Convert YOLO format to albumentations format (x_min, y_min, x_max, y_max, normalized)
                x_center, y_center, width, height = bbox_data['bbox']
                x_min = max(0, x_center - width / 2)
                y_min = max(0, y_center - height / 2)
                x_max = min(1, x_center + width / 2)
                y_max = min(1, y_center + height / 2)

                bboxes.append([x_min, y_min, x_max, y_max])
                class_labels.append(bbox_data['class_id'])

            # Apply transformations
            if bboxes:
                transformed = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
            else:
                transformed = pipeline(image=image)

            transformed_image = transformed['image']
            transformed_bboxes = transformed.get('bboxes', [])
            transformed_labels = transformed.get('class_labels', [])

            # Save transformed image
            base_name = os.path.splitext(os.path.basename(img_data['path']))[0]
            output_filename = f"{base_name}_{suffix}.jpg"
            output_path = os.path.join(self.output_dir, '../images', output_filename)

            # Convert back to BGR for saving
            bgr_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, bgr_image)

            # Save annotations in YOLO format
            if transformed_bboxes:
                label_filename = f"{base_name}_{suffix}.txt"
                label_path = os.path.join(self.output_dir, '../labels', label_filename)

                with open(label_path, 'w') as f:
                    for bbox, class_id in zip(transformed_bboxes, transformed_labels):
                        x_min, y_min, x_max, y_max = bbox
                        # Convert back to YOLO format
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        width = x_max - x_min
                        height = y_max - y_min

                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            return transformed_image, transformed_bboxes

        except Exception as e:
            print(f"Error in process_image: {e}")
            return None, None

