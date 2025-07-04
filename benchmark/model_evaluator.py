import time
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import numpy as np
from sb3_contrib import MaskablePPO

from src.simulator.simulation.play import Play
from src.simulator.simulation.play_config import PlayConfig
from src.simulator.team.team import Team
from src.simulator.team.guest import Guest
from src.team_builder.env.env import LLSIFTeamBuildingEnv
from benchmark.benchmark_case import BenchmarkCase


class ModelEvaluator:
    """Evaluates a trained model on a given BenchmarkCase."""

    def __init__(
        self, model_path: Path, num_simulations: int = 1000, deterministic: bool = True
    ):
        self.model_path = model_path
        self.num_simulations = num_simulations
        self.deterministic = deterministic
        self.model = MaskablePPO.load(str(model_path))
        print(f"Loaded model from: {self.model_path}")

    def evaluate_on_case(self, case: BenchmarkCase) -> Dict:
        """
        Evaluate the model on a single benchmark case.

        Returns a dictionary containing detailed results and statistics.
        """
        start_time = time.time()

        env = case.get_env()

        print(f"  > Predicting team for case: {case.name}...")
        team_data, approach_rate = self._get_model_prediction(env)

        if not team_data.get("slots"):
            print("  > Agent failed to build a team. Score is 0.")
            return self._create_error_result(case, "Agent did not build a team.")

        predicted_team = self._reconstruct_team(team_data, case)

        print(f"  > Running {self.num_simulations} simulations...")
        simulation_scores = self._run_simulations(predicted_team, case, approach_rate)

        evaluation_time = time.time() - start_time

        mean_score = np.mean(simulation_scores) if simulation_scores else 0

        print(
            f"  > Evaluation complete in {evaluation_time:.2f}s. "
            f"Mean Score: {mean_score:,.0f}"
        )

        return {
            "case_name": case.name,
            "predicted_team": self._extract_team_info(predicted_team),
            "predicted_approach_rate": approach_rate,
            "simulation_scores": simulation_scores,
            "mean_score": mean_score,
            "std_score": np.std(simulation_scores) if simulation_scores else 0,
            "min_score": np.min(simulation_scores) if simulation_scores else 0,
            "max_score": np.max(simulation_scores) if simulation_scores else 0,
            "evaluation_time": evaluation_time,
        }

    def _get_model_prediction(
        self, env: LLSIFTeamBuildingEnv
    ) -> Tuple[Dict[str, Any], int]:
        """Get the model's predicted team and approach rate."""
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            action, _ = self.model.predict(
                obs, deterministic=self.deterministic, action_masks=env.action_masks()
            )
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        return info.get("final_team_data", {}), info.get("final_approach_rate", 1)

    def _reconstruct_team(self, team_data: Dict[str, Any], case: BenchmarkCase) -> Team:
        """Reconstructs a Team object from the logged data for a stateless, robust evaluation."""
        guest_manager = Guest(str(case.master_data_path / "unique_leader_skills.json"))
        team = Team(case.deck, case.accessory_manager, case.sis_manager, guest_manager)

        guest_str = team_data.get("guest", "None")
        if guest_str != "None" and team.guest_manager:
            match = re.search(r"leader_skill_id=(\d+)", guest_str)
            if match:
                team.guest_manager.set_guest(int(match.group(1)))

        unassigned_sis = defaultdict(list)
        for ps in case.sis_manager.skills.values():
            unassigned_sis[ps.sis.name].append(ps.manager_internal_id)

        for slot_data in team_data.get("slots", []):
            slot_num = slot_data["slot_number"]
            if slot_data.get("deck_id"):
                team.equip_card_in_slot(slot_num, slot_data["deck_id"])
            if slot_data.get("accessory_id"):
                team.equip_accessory_in_slot(slot_num, slot_data["accessory_id"])
            for sis_name in slot_data.get("sis_list", []):
                if unassigned_sis.get(sis_name):
                    sis_id = unassigned_sis[sis_name].pop(0)
                    team.equip_sis_in_slot(slot_num, sis_id)

        team.calculate_team_stats()
        return team

    def _run_simulations(
        self, team: Team, case: BenchmarkCase, approach_rate: int
    ) -> List[int]:
        """Run multiple simulations with the predicted team."""
        if not case.song:
            print(
                f"  > ERROR: Song could not be loaded for case '{case.name}'. Skipping simulations."
            )
            return []

        play_config = PlayConfig(
            accuracy=case.config.accuracy,
            approach_rate=approach_rate,
            seed=None,
            enable_logging=False,
        )
        play = Play(team, case.song, play_config, case.game_data)
        return play.simulate(n_trials=self.num_simulations)

    def _extract_team_info(self, team: Team) -> Dict:
        """Extract serializable team information for logging."""
        info = {
            "total_stats": {
                "smile": team.total_team_smile,
                "pure": team.total_team_pure,
                "cool": team.total_team_cool,
            },
            "slots": [],
        }
        if team.guest_manager and team.guest_manager.current_guest:
            guest = team.guest_manager.current_guest
            info["guest"] = {
                "leader_skill_id": guest.leader_skill_id,
                "leader_attribute": guest.leader_attribute,
                "leader_secondary_attribute": guest.leader_secondary_attribute,
                "leader_value": guest.leader_value,
                "leader_extra_attribute": guest.leader_extra_attribute,
                "leader_extra_target": guest.leader_extra_target,
                "leader_extra_value": guest.leader_extra_value,
            }

        for i, slot in enumerate(team.slots):
            if slot.card:
                slot_info = {
                    "position": i + 1,
                    "card": {
                        "name": slot.card.display_name,
                        "deck_id": slot.card_entry.deck_id if slot.card_entry else None,
                    },
                    "accessory": (
                        {"name": slot.accessory.name} if slot.accessory else None
                    ),
                    "sis": [s.sis.name for s in slot.sis_entries],
                }
                info["slots"].append(slot_info)
        return info

    def _create_error_result(self, case: BenchmarkCase, error_msg: str) -> Dict:
        """Creates a result dictionary for a failed evaluation."""
        return {
            "case_name": case.name,
            "mean_score": 0,
            "std_score": 0,
            "error": error_msg,
            "evaluation_time": 0.0,
        }
