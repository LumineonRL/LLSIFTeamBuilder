import sys
import json
import warnings
from pathlib import Path
from typing import List, Dict

from benchmark.benchmark_case import BenchmarkCase
from benchmark.model_evaluator import ModelEvaluator
from benchmark.results_manager import ResultsManager


class BenchmarkRunner:
    """Main class for discovering benchmark cases and orchestrating model benchmarks."""

    def __init__(
        self,
        suite_dir: Path = Path("./data/model_eval"),
        master_data_path: Path = Path("./data"),
        results_dir: Path = Path("./results"),
        num_simulations: int = 1000,
    ):
        self.suite_dir = suite_dir
        self.master_data_path = master_data_path
        self.num_simulations = num_simulations
        self.results_manager = ResultsManager(results_dir)
        self.benchmark_cases = self._load_benchmark_cases()

    def _load_benchmark_cases(self) -> List[BenchmarkCase]:
        """Load all valid benchmark cases from the suite directory."""
        print(f"Loading benchmark cases from: {self.suite_dir}")
        if not self.suite_dir.exists():
            print(
                f"Error: Benchmark suite directory not found at '{self.suite_dir}'",
                file=sys.stderr,
            )
            return []

        loaded_cases = []
        for item in sorted(self.suite_dir.iterdir()):
            if item.is_dir() and (item / "config.json").exists():
                try:
                    case = BenchmarkCase(item, self.master_data_path)
                    loaded_cases.append(case)
                    print(f"  > Loaded case: {item.name}")
                except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                    print(
                        f"  > Failed to load case '{item.name}': Invalid or missing file/key. Error: {e}",
                        file=sys.stderr,
                    )

        if not loaded_cases:
            print("Warning: No valid benchmark cases were found.", file=sys.stderr)

        return loaded_cases

    def run(
        self,
        models_to_test: Dict[str, Path],
        save_json: bool = True,
        save_report: bool = True,
    ):
        """
        Run the full benchmark suite for one or more models.

        Args:
            models_to_test: A dictionary mapping a friendly model name to its .zip file path.
            save_json: If True, saves detailed results to a JSON file.
            save_report: If True, saves a report to a Markdown file.
        """
        if not self.benchmark_cases:
            print(
                "Cannot run benchmark: No benchmark cases were loaded.", file=sys.stderr
            )
            return

        all_model_results = {}
        for model_name, model_path in models_to_test.items():
            print("\n" + "-" * 80)
            print(f"BENCHMARKING MODEL: {model_name}")
            print(f"Path: {model_path}")
            print("-" * 80)

            if not model_path.exists():
                print(
                    f"Error: Model file not found for '{model_name}'. Skipping.",
                    file=sys.stderr,
                )
                continue

            evaluator = ModelEvaluator(model_path, num_simulations=self.num_simulations)

            model_results = []
            for case in self.benchmark_cases:
                try:
                    result = evaluator.evaluate_on_case(case)
                    model_results.append(result)
                except (RuntimeError, ValueError, KeyError, FileNotFoundError) as e:
                    print(
                        f"  > CRITICAL ERROR during evaluation of '{case.name}': {e}",
                        file=sys.stderr,
                    )
                    model_results.append({"case_name": case.name, "error": str(e)})

            all_model_results[model_name] = model_results

            if save_json:
                self.results_manager.save_results(model_name, model_results)
            if save_report:
                self.results_manager.save_report(model_name, model_results)

        if len(all_model_results) > 1:
            self.results_manager.compare_models(all_model_results)

        print("\nBenchmark suite finished.")


def main():
    """Defines the models to test and runs the suite."""

    warnings.filterwarnings("ignore", category=UserWarning)
    
    models_to_benchmark = {
        "llsif_ppo_agent_final": Path("./models/llsif_ppo/llsif_ppo_agent_final.zip"),
        "best_model": Path("./models/llsif_ppo/best_model.zip"),
        "750k": Path("./models/llsif_ppo/llsif_ppo_agent_checkpoint_750000_steps.zip"),
    }

    runner = BenchmarkRunner(num_simulations=1000)
    runner.run(models_to_benchmark)


if __name__ == "__main__":
    main()
