import os
import time


class OutputManager:
    def __init__(self, base_path):
        """
        Creates a structured output directory for the experiment.
        Structure:
          base_path/
            ├── images/         (Decoded X-rays, Filmstrips)
            ├── plots/
            │   ├── validity/   (Is the image valid?)
            │   ├── dynamics/   (How did it evolve?)
            │   └── geometry/   (Where did it end up?)
        """
        self.base_dir = base_path.replace(".png", "")  # Strip extension if passed a file path
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Define Subdirectories
        self.dirs = {
            "root": self.base_dir,
            "images": os.path.join(self.base_dir, "images"),
            "validity": os.path.join(self.base_dir, "plots", "validity"),
            "dynamics": os.path.join(self.base_dir, "plots", "dynamics"),
            "geometry": os.path.join(self.base_dir, "plots", "geometry"),
        }

        # Create directories
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        print(f" Experiment Output Initialized at: {self.base_dir}")

    def get_path(self, category, filename):
        """Returns the full path for a file in a specific category."""
        if category not in self.dirs:
            raise ValueError(f"Unknown category '{category}'. Options: {list(self.dirs.keys())}")
        return os.path.join(self.dirs[category], filename)