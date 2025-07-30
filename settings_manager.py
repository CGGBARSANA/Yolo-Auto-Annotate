import os
import json


class SettingsManager:
    """Manages application settings"""

    def __init__(self):
        self.settings_file = os.path.join(os.path.expanduser("~"), ".yolo_annotator_settings.json")
        self.default_settings = {
            'model_path': '',
            'label_dir': '',
            'annotation_save_dir': ''
        }

    def load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    self.default_settings.update(settings)
                    return self.default_settings
        except Exception as e:
            print(f"Error loading settings: {e}")

        return self.default_settings.copy()

    def save_settings(self, settings):
        """Save settings to file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False

    def settings_exist(self):
        """Check if settings file exists and has required settings"""
        settings = self.load_settings()
        required_keys = ['model_path', 'label_dir', 'annotation_save_dir']

        for key in required_keys:
            if not settings.get(key) or not os.path.exists(settings[key]):
                return False
        return True