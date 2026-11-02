import os
import json
import shutil
import glob

class ContinuousLearningPipeline:
    def __init__(self, events_dir="events_output", repository_dir="training_repository"):
        self.events_dir = events_dir
        self.repository_dir = repository_dir
        
        # Create structured subdirectories for active learning
        self.true_positives_dir = os.path.join(self.repository_dir, "true_positives")
        self.false_positives_dir = os.path.join(self.repository_dir, "false_positives")
        self.edge_cases_dir = os.path.join(self.repository_dir, "edge_cases")
        
        for d in [self.true_positives_dir, self.false_positives_dir, self.edge_cases_dir]:
            os.makedirs(d, exist_ok=True)

    def review_and_categorize_event(self, json_filename, label="true_positive", notes=""):
        """
        Moves an event's JSON metadata and paired MP4 clip into the appropriate 
        training repository category based on human review or automated filtering.
        
        Labels: 'true_positive', 'false_positive', 'edge_case'
        """
        base_name = os.path.splitext(json_filename)[0]
        json_path = os.path.join(self.events_dir, f"{base_name}.json")
        video_path = os.path.join(self.events_dir, f"{base_name}.mp4")
        
        if not os.path.exists(json_path):
            print(f"[ERROR] Event metadata not found: {json_path}")
            return False

        # Load metadata, append review notes & label
        with open(json_path, "r") as f:
            metadata = json.load(f)
        
        metadata["review_status"] = label
        metadata["reviewer_notes"] = notes

        # Determine target repository folder
        if label == "true_positive":
            target_folder = self.true_positives_dir
        elif label == "false_positive":
            target_folder = self.false_positives_dir
        else:
            target_folder = self.edge_cases_dir

        # Copy files to training repository
        new_json_path = os.path.join(target_folder, f"{base_name}.json")
        with open(new_json_path, "w") as f:
            json.dump(metadata, f, indent=4)

        if os.path.exists(video_path):
            shutil.copy(video_path, os.path.join(target_folder, f"{base_name}.mp4"))

        print(f"[SUCCESS] Event '{base_name}' categorized as [{label}] and stored in training repo.")
        return True

    def export_dataset_manifest(self):
        """
        Generates a summary JSON manifest of all validated training data 
        ready for model retraining pipelines.
        """
        manifest = {
            "true_positives_count": len(glob.glob(os.path.join(self.true_positives_dir, "*.json"))),
            "false_positives_count": len(glob.glob(os.path.join(self.false_positives_dir, "*.json"))),
            "edge_cases_count": len(glob.glob(os.path.join(self.edge_cases_dir, "*.json"))),
            "dataset_path": os.path.abspath(self.repository_dir)
        }
        
        manifest_path = os.path.join(self.repository_dir, "dataset_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)
            
        print(f"[INFO] Dataset manifest exported to {manifest_path}")
        return manifest