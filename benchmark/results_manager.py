import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd


class ResultsManager:
    """Manages benchmark results storage, reporting, and comparison."""

    def __init__(self, results_dir: Path = Path("./results")):
        self.results_dir = results_dir
        self.results_dir.mkdir(exist_ok=True)

    def save_results(self, model_name: str, results: List[Dict]) -> Path:
        """Saves the complete benchmark results for a model to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}_results.json"
        filepath = self.results_dir / filename

        save_data = {
            "model_name": model_name,
            "timestamp": timestamp,
            "results": results,
            "summary": self._generate_summary(results),
        }

        save_data = self._convert_for_json(save_data)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2)

        print(f"  > Full results saved to: {filepath}")
        return filepath

    def save_report(self, model_name: str, results: List[Dict]) -> Path:
        """Generates and saves a human-readable Markdown report."""
        report = self._generate_markdown_report(model_name, results)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}_report.md"
        filepath = self.results_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"  > Markdown report saved to: {filepath}")
        return filepath

    def compare_models(self, all_results: Dict[str, List[Dict]]):
        """Prints a comparison table of multiple models to the console."""
        comparison_data = []
        for model_name, results in all_results.items():
            summary = self._generate_summary(results)
            comparison_data.append(
                {
                    "Model": model_name,
                    "Avg Score": summary["overall_mean"],
                    "Std Dev": summary["overall_std"],
                    "Best Case": f"{summary['best_score']:,.0f} ({summary['best_case']})",
                    "Worst Case": f"{summary['worst_score']:,.0f} ({summary['worst_case']})",
                    "Time (s)": summary["total_time"],
                }
            )

        if not comparison_data:
            print("No results to compare.")
            return

        df = pd.DataFrame(comparison_data)
        df = df.sort_values("Avg Score", ascending=False).reset_index(drop=True)
        df["Rank"] = df.index + 1
        df = df.set_index("Rank")

        print("\n" + "=" * 80)
        print("MODEL BENCHMARK COMPARISON")
        print("=" * 80)
        df["Avg Score"] = df["Avg Score"].map("{:,.0f}".format)
        df["Std Dev"] = df["Std Dev"].map("{:,.0f}".format)
        df["Time (s)"] = df["Time (s)"].map("{:.2f}".format)
        print(df.to_string())
        print("=" * 80 + "\n")

    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Helper to generate summary statistics from a list of results."""
        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            return {
                "overall_mean": 0,
                "overall_std": 0,
                "best_score": 0,
                "best_case": "N/A",
                "worst_score": 0,
                "worst_case": "N/A",
                "total_time": 0,
            }

        mean_scores = [r["mean_score"] for r in valid_results]
        best_idx = np.argmax(mean_scores)
        worst_idx = np.argmin(mean_scores)

        return {
            "overall_mean": np.mean(mean_scores),
            "overall_std": np.std(mean_scores),
            "best_score": mean_scores[best_idx],
            "best_case": valid_results[best_idx]["case_name"],
            "worst_score": mean_scores[worst_idx],
            "worst_case": valid_results[worst_idx]["case_name"],
            "total_time": sum(r["evaluation_time"] for r in valid_results),
        }

    def _generate_markdown_report(self, model_name: str, results: List[Dict]) -> str:
        """Helper to format the detailed Markdown report."""
        summary = self._generate_summary(results)
        report = f"# Benchmark Report: {model_name}\n\n"
        report += "## Overall Summary\n"
        report += f"- **Average Score**: {summary['overall_mean']:,.0f} (±{summary['overall_std']:,.0f})\n"
        report += f"- **Best Performance**: {summary['best_score']:,.0f} on `{summary['best_case']}`\n"
        report += f"- **Worst Performance**: {summary['worst_score']:,.0f} on `{summary['worst_case']}`\n\n"

        for result in results:
            report += f"## Benchmark Case: `{result['case_name']}`\n"
            if "error" in result:
                report += f"**Status:** FAILED\n**Reason:** {result['error']}\n\n"
                continue

            report += f"- **Mean Score**: {result['mean_score']:,.0f} (±{result['std_score']:,.0f})\n"
            team = result["predicted_team"]
            stats = team["total_stats"]
            report += f"- **Team Stats (S/P/C)**: {stats['smile']:,}/{stats['pure']:,}/{stats['cool']:,}\n"
            report += "| Slot | Card | Accessory | SIS |\n"
            report += "|:----:|:-----|:----------|:----|\n"
            for slot in team["slots"]:
                card = slot["card"]["name"]
                acc = slot["accessory"]["name"] if slot["accessory"] else " "
                sis_count = len(slot["sis"])
                report += f"| {slot['position']} | {card} | {acc} | {sis_count} |\n"
            report += "\n"
        return report

    def _convert_for_json(self, obj: Any) -> Any:
        """Recursively converts numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        return obj
