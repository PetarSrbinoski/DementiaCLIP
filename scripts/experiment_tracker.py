# scripts/experiment_tracker.py
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil
import config


class ExperimentTracker:
    """Track experiments, save model weights, and log results for reporting"""

    def __init__(self, experiment_name, mode):
        self.experiment_name = experiment_name
        self.mode = mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{mode}_{self.timestamp}"

        # Create experiment directory
        self.exp_dir = config.OUTPUTS_DIR / "experiments" / self.experiment_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.exp_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.results_file = self.exp_dir / "results.json"
        self.cv_results_file = self.exp_dir / "cv_results.csv"
        self.config_file = self.exp_dir / "config.txt"
        self.log_file = self.exp_dir / "training_log.txt"

        self.results = {
            "experiment_id": self.experiment_id,
            "experiment_name": experiment_name,
            "mode": mode,
            "timestamp": self.timestamp,
            "model_config": {},
            "training_config": {},
            "cv_metrics": [],
            "average_metrics": {},
            "fold_models": []
        }

        self.cv_results = []

    def log_config(self, model_name, pretrained, clip_model_name, other_config=None):
        """Log model and training configuration"""
        self.results["model_config"] = {
            "model_name": model_name,
            "clip_model_name": clip_model_name,
            "pretrained": pretrained,
            "mode": self.mode
        }

        self.results["training_config"] = {
            "batch_size": config.BATCH_SIZE,
            "epochs": config.EPOCHS_VISION if self.mode == 'vision_only' else config.EPOCHS_MULTIMODAL,
            "lr_vision": float(config.LR_VISION),
            "lr_multimodal": float(config.LR_MULTIMODAL),
            "n_splits": config.N_SPLITS,
            "random_state": config.RANDOM_STATE,
            "device": config.DEVICE,
            "total_samples": int(other_config.get("total_samples", 0)) if other_config else 0,
            "num_control": int(other_config.get("num_control", 0)) if other_config else 0,
            "num_dementia": int(other_config.get("num_dementia", 0)) if other_config else 0,
        }

        if other_config:
            self.results["training_config"].update({
                k: v for k, v in other_config.items()
                if k not in ["total_samples", "num_control", "num_dementia"]
            })

        self._save_config_txt()

    def log_fold_results(self, fold, metrics, model_path=None):
        """Log results for a single fold"""
        fold_result = {
            "fold": fold + 1,
            "accuracy": float(metrics['acc']),
            "f1_score": float(metrics['f1']),
            "roc_auc": float(metrics['roc_auc'])
        }

        if model_path:
            fold_result["model_path"] = str(model_path)
            self.results["fold_models"].append(str(model_path))

        self.results["cv_metrics"].append(fold_result)
        self.cv_results.append(fold_result)

    def log_average_metrics(self, avg_metrics_df):
        """Log cross-validation averages"""
        self.results["average_metrics"] = {
            "avg_accuracy": float(avg_metrics_df['acc']),
            "avg_f1_score": float(avg_metrics_df['f1']),
            "avg_roc_auc": float(avg_metrics_df['roc_auc']),
            "std_accuracy": float(self.cv_results_df()['accuracy'].std()) if len(self.cv_results) > 0 else 0.0,
            "std_f1_score": float(self.cv_results_df()['f1_score'].std()) if len(self.cv_results) > 0 else 0.0,
            "std_roc_auc": float(self.cv_results_df()['roc_auc'].std()) if len(self.cv_results) > 0 else 0.0,
        }

    def cv_results_df(self):
        """Return CV results as DataFrame"""
        return pd.DataFrame(self.cv_results)

    def save_model_weights(self, model, fold):
        """Save model weights for a fold"""
        model_path = self.models_dir / f"model_fold_{fold + 1}.pt"
        torch.save(model.state_dict(), model_path)
        return model_path

    def save_checkpoint(self, model, optimizer, epoch, fold):
        """Save checkpoint during training"""
        checkpoint_path = self.models_dir / f"checkpoint_fold_{fold + 1}_epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        return checkpoint_path

    def save_results(self):
        """Save all results to JSON and CSV"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=4)

        if self.cv_results:
            cv_df = pd.DataFrame(self.cv_results)
            cv_df.to_csv(self.cv_results_file, index=False)

    def _save_config_txt(self):
        """Save configuration as readable text file"""
        with open(self.config_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("EXPERIMENT CONFIGURATION\n")
            f.write("=" * 60 + "\n\n")

            f.write("EXPERIMENT DETAILS:\n")
            f.write(f"  Experiment ID: {self.experiment_id}\n")
            f.write(f"  Experiment Name: {self.experiment_name}\n")
            f.write(f"  Mode: {self.mode}\n")
            f.write(f"  Timestamp: {self.timestamp}\n\n")

            f.write("MODEL CONFIGURATION:\n")
            for key, value in self.results["model_config"].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            f.write("TRAINING CONFIGURATION:\n")
            for key, value in self.results["training_config"].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            f.write("OUTPUT PATHS:\n")
            f.write(f"  Experiment Directory: {self.exp_dir}\n")
            f.write(f"  Models Directory: {self.models_dir}\n")
            f.write(f"  Results JSON: {self.results_file}\n")
            f.write(f"  Results CSV: {self.cv_results_file}\n")

    def log_message(self, message):
        """Log training messages to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")

    def get_summary(self):
        """Return a summary of the experiment"""
        summary = f"""
{'=' * 60}
EXPERIMENT SUMMARY
{'=' * 60}
Experiment ID: {self.experiment_id}
Name: {self.experiment_name}
Mode: {self.mode}
Timestamp: {self.timestamp}

MODEL: {self.results['model_config'].get('clip_model_name')}
Training Mode: {self.mode}

RESULTS:
  Average Accuracy:  {self.results['average_metrics'].get('avg_accuracy', 0):.4f} ± {self.results['average_metrics'].get('std_accuracy', 0):.4f}
  Average F1-Score:  {self.results['average_metrics'].get('avg_f1_score', 0):.4f} ± {self.results['average_metrics'].get('std_f1_score', 0):.4f}
  Average ROC-AUC:   {self.results['average_metrics'].get('avg_roc_auc', 0):.4f} ± {self.results['average_metrics'].get('std_roc_auc', 0):.4f}

FOLD DETAILS:
"""
        for fold_result in self.cv_results:
            summary += f"  Fold {fold_result['fold']}: Acc={fold_result['accuracy']:.4f}, F1={fold_result['f1_score']:.4f}, ROC-AUC={fold_result['roc_auc']:.4f}\n"

        summary += f"\nAll results saved to: {self.exp_dir}\n"
        summary += f"{'=' * 60}\n"
        return summary

    def copy_metadata(self):
        """Copy metadata file for reference"""
        if config.METADATA_FILE.exists():
            shutil.copy(config.METADATA_FILE, self.exp_dir / "metadata_used.csv")


# Utility function to create a report
def generate_experiment_report(output_dir=None):
    """Generate a report comparing all experiments"""
    if output_dir is None:
        output_dir = config.OUTPUTS_DIR / "experiments"

    if not output_dir.exists():
        print(f"No experiments directory found at {output_dir}")
        return

    report_data = []

    for exp_dir in sorted(output_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        results_file = exp_dir / "results.json"
        if not results_file.exists():
            continue

        with open(results_file, 'r') as f:
            results = json.load(f)

        report_data.append({
            "Experiment ID": results.get("experiment_id"),
            "Name": results.get("experiment_name"),
            "Mode": results.get("mode"),
            "Model": results.get("model_config", {}).get("clip_model_name"),
            "Accuracy": results.get("average_metrics", {}).get("avg_accuracy"),
            "Accuracy±": results.get("average_metrics", {}).get("std_accuracy"),
            "F1-Score": results.get("average_metrics", {}).get("avg_f1_score"),
            "F1±": results.get("average_metrics", {}).get("std_f1_score"),
            "ROC-AUC": results.get("average_metrics", {}).get("avg_roc_auc"),
            "ROC-AUC±": results.get("average_metrics", {}).get("std_roc_auc"),
            "Timestamp": results.get("timestamp"),
        })

    if not report_data:
        print("No experiment results found.")
        return None

    report_df = pd.DataFrame(report_data)
    report_file = output_dir / "experiment_report.csv"
    report_df.to_csv(report_file, index=False)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON REPORT")
    print("=" * 80)
    print(report_df.to_string(index=False))
    print("=" * 80)
    print(f"\nReport saved to: {report_file}\n")

    return report_df


if __name__ == "__main__":
    # Example: Generate report from all experiments
    generate_experiment_report()