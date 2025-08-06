import os
import json


class CustomLabelManager:
    """Manages custom labels with consistent class IDs"""

    def __init__(self, settings_manager):
        self.settings_manager = settings_manager
        self.custom_labels = {}  # {label_name: class_id}
        self.custom_only_labels = {}  # Separate storage for custom-only mode
        self.max_model_class_id = 0
        self.next_custom_id = 0
        self.model_classes = {}
        self.current_mode = "both"  # Track current class selection mode
        self.load_custom_labels()

    def set_model_classes(self, model_class_names):
        """Set the model class names and update max class ID"""
        self.model_classes = model_class_names or {}
        if model_class_names:
            self.max_model_class_id = max(model_class_names.keys()) if model_class_names else 0
            # For "both" mode, custom labels start after the highest model ID
            self.next_custom_id = self.max_model_class_id + 1

            # Update next_custom_id to avoid conflicts with existing custom labels in "both" mode
            if self.custom_labels:
                max_custom_id = max(self.custom_labels.values())
                self.next_custom_id = max(self.next_custom_id, max_custom_id + 1)

    def set_mode(self, class_selection_mode):
        """Set the current class selection mode"""
        self.current_mode = class_selection_mode

    def add_custom_label(self, label_name):
        """Add a new custom label with consistent class ID based on current mode"""
        if self.current_mode == "custom_only":
            # In custom_only mode, use separate storage and start from 0
            if label_name in self.custom_only_labels:
                return self.custom_only_labels[label_name]

            class_id = len(self.custom_only_labels)
            self.custom_only_labels[label_name] = class_id
        else:
            # In "both" mode, use regular storage and start after model IDs
            if label_name in self.custom_labels:
                return self.custom_labels[label_name]

            class_id = self.next_custom_id
            self.custom_labels[label_name] = class_id
            self.next_custom_id += 1

        self.save_custom_labels()
        return class_id

    def remove_custom_label(self, label_name):
        """Remove a custom label"""
        removed = False

        if self.current_mode == "custom_only":
            if label_name in self.custom_only_labels:
                del self.custom_only_labels[label_name]
                # Reassign IDs to maintain sequential order starting from 0
                self._reassign_custom_only_ids()
                removed = True
        else:
            if label_name in self.custom_labels:
                del self.custom_labels[label_name]
                removed = True

        if removed:
            self.save_custom_labels()
        return removed

    def edit_custom_label(self, old_name, new_name):
        """Edit a custom label name"""
        if self.current_mode == "custom_only":
            if old_name in self.custom_only_labels and new_name not in self.custom_only_labels:
                class_id = self.custom_only_labels[old_name]
                del self.custom_only_labels[old_name]
                self.custom_only_labels[new_name] = class_id
                self.save_custom_labels()
                return True
        else:
            if old_name in self.custom_labels and new_name not in self.custom_labels:
                class_id = self.custom_labels[old_name]
                del self.custom_labels[old_name]
                self.custom_labels[new_name] = class_id
                self.save_custom_labels()
                return True
        return False

    def _reassign_custom_only_ids(self):
        """Reassign custom-only label IDs to be sequential starting from 0"""
        if self.custom_only_labels:
            old_labels = list(self.custom_only_labels.keys())
            self.custom_only_labels.clear()
            for i, label_name in enumerate(old_labels):
                self.custom_only_labels[label_name] = i

    def get_class_id(self, label_name, model_class_names=None, class_selection_mode=None):
        """Get class ID for a label (model or custom) respecting the current mode"""
        if model_class_names is None:
            model_class_names = self.model_classes
        if class_selection_mode is None:
            class_selection_mode = self.current_mode

        # Handle different modes
        if class_selection_mode == "model_only":
            # Only allow model classes
            for class_id, class_name in model_class_names.items():
                if class_name == label_name:
                    return class_id
            raise ValueError(f"Label '{label_name}' is not a model class")

        elif class_selection_mode == "custom_only":
            # In custom_only mode, all labels are custom and start from 0
            if label_name in self.custom_only_labels:
                return self.custom_only_labels[label_name]
            else:
                # Create new custom label starting from current count
                new_id = len(self.custom_only_labels)
                self.custom_only_labels[label_name] = new_id
                self.save_custom_labels()
                return new_id

        else:  # "both" mode
            # Check if it's a model class first - model classes keep their original IDs
            for class_id, class_name in model_class_names.items():
                if class_name == label_name:
                    return class_id

            # Check if it's an existing custom class
            if label_name in self.custom_labels:
                return self.custom_labels[label_name]

            # If not found, add as new custom label starting after highest model ID
            class_id = self.next_custom_id
            self.custom_labels[label_name] = class_id
            self.next_custom_id += 1
            self.save_custom_labels()
            return class_id

    def get_all_labels(self, model_class_names=None, class_selection_mode=None):
        """Get all labels (model + custom) with their class IDs based on mode"""
        if model_class_names is None:
            model_class_names = self.model_classes
        if class_selection_mode is None:
            class_selection_mode = self.current_mode

        all_labels = {}

        if class_selection_mode == "model_only":
            # Only model classes with their original IDs
            if model_class_names:
                all_labels.update(model_class_names)

        elif class_selection_mode == "custom_only":
            # Only custom labels starting from ID 0
            for label_name, class_id in self.custom_only_labels.items():
                all_labels[class_id] = label_name

        else:  # "both" mode
            # Model classes keep their original IDs (0, 1, 2, ..., max_model_id)
            if model_class_names:
                all_labels.update(model_class_names)
            # Custom classes start after model classes (max_model_id + 1, max_model_id + 2, ...)
            for label_name, class_id in self.custom_labels.items():
                all_labels[class_id] = label_name

        return all_labels

    def get_custom_labels_list(self):
        """Get list of custom label names based on current mode"""
        if self.current_mode == "custom_only":
            return list(self.custom_only_labels.keys())
        else:
            return list(self.custom_labels.keys())

    def save_custom_labels(self):
        """Save custom labels to file"""
        try:
            settings = self.settings_manager.load_settings()
            if settings and 'annotation_save_dir' in settings:
                custom_labels_file = os.path.join(settings['annotation_save_dir'], 'custom_labels.json')

                # Save both storage dictionaries
                data_to_save = {
                    'both_mode_labels': self.custom_labels,  # For "both" mode
                    'custom_only_labels': self.custom_only_labels,  # For "custom_only" mode
                    'next_custom_id': self.next_custom_id
                }

                with open(custom_labels_file, 'w') as f:
                    json.dump(data_to_save, f, indent=2)
                return True
        except Exception as e:
            print(f"Error saving custom labels: {e}")
        return False

    def load_custom_labels(self):
        """Load custom labels from file"""
        try:
            settings = self.settings_manager.load_settings()
            if settings and 'annotation_save_dir' in settings:
                custom_labels_file = os.path.join(settings['annotation_save_dir'], 'custom_labels.json')
                if os.path.exists(custom_labels_file):
                    with open(custom_labels_file, 'r') as f:
                        data = json.load(f)

                    # Handle both old format (direct dictionary) and new format
                    if isinstance(data, dict) and 'both_mode_labels' in data:
                        # New format
                        self.custom_labels = data.get('both_mode_labels', {})
                        self.custom_only_labels = data.get('custom_only_labels', {})
                        self.next_custom_id = data.get('next_custom_id', 0)
                    else:
                        # Old format - treat as "both" mode labels
                        self.custom_labels = data
                        self.custom_only_labels = {}
                        if self.custom_labels:
                            self.next_custom_id = max(self.custom_labels.values()) + 1

                    return True
        except Exception as e:
            print(f"Error loading custom labels: {e}")
        return False

    def get_label_info_for_display(self):
        """Get formatted label information for display in dialogs"""
        info = []

        if self.current_mode == "model_only":
            info.append("Mode: Model Classes Only")
            if self.model_classes:
                for class_id, class_name in sorted(self.model_classes.items()):
                    info.append(f"  {class_name} (ID: {class_id})")

        elif self.current_mode == "custom_only":
            info.append("Mode: Custom Labels Only")
            if self.custom_only_labels:
                for label_name, class_id in sorted(self.custom_only_labels.items(), key=lambda x: x[1]):
                    info.append(f"  {label_name} (ID: {class_id})")
            else:
                info.append("  No custom labels defined")

        else:  # "both" mode
            info.append("Mode: Both Model Classes and Custom Labels")
            info.append("Model Classes:")
            if self.model_classes:
                for class_id, class_name in sorted(self.model_classes.items()):
                    info.append(f"  {class_name} (ID: {class_id})")

            info.append("Custom Labels:")
            if self.custom_labels:
                for label_name, class_id in sorted(self.custom_labels.items(), key=lambda x: x[1]):
                    info.append(f"  {label_name} (ID: {class_id})")
            else:
                info.append("  No custom labels defined")

        return "\n".join(info)