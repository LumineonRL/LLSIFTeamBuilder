import json
from typing import Dict, Any


import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class TeamLogCallback(BaseCallback):
    """
    Logs the best team composition to TensorBoard's text section when a new
    best reward is found during training.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.best_reward = -np.inf
        self.training_env: VecEnv
        self.num_calls: int = 0

    def _on_step(self) -> bool:
        """Checks for finished episodes and logs info if a new best reward is found."""
        dones = self.locals.get("dones", [])
        if not np.any(dones):
            return True

        for idx, done in enumerate(dones):
            if done:
                info = self.locals["infos"][idx]
                if "episode" in info:
                    episode_reward = info["episode"]["r"]
                    if episode_reward > self.best_reward:
                        self.best_reward = episode_reward
                        if self.verbose > 0:
                            print(
                                f"\nNew best reward found: {self.best_reward:.2f} at step {self.num_calls}"
                            )
                        env = self.training_env.get_attr("unwrapped", indices=[idx])[0]
                        render_data = env.render(mode="agent")

                        if render_data:
                            log_text = self._format_render_data(
                                render_data, episode_reward
                            )

                            if self.verbose > 0:
                                self._print_team_summary(render_data)

                            writer = getattr(self.logger, "writer", None)

                            if writer is not None:
                                writer.add_text(
                                    "best_team_composition",
                                    log_text,
                                    global_step=self.num_calls,
                                )

                            self.logger.record("custom/best_reward", self.best_reward)
                            self.logger.dump(step=self.num_calls)

        return True

    def _format_render_data(self, data: Dict[str, Any], score: float) -> str:
        """Formats the agent render data into a readable Markdown string."""
        try:
            team_details = data.get("team", {})
            guest = data.get("guest", "None")
            approach_rate = data.get("final_approach_rate", "N/A")
            team_str = json.dumps(team_details, indent=2)

            return (
                f"### New Best Score: {score:.2f}\n\n"
                f"**Global Step:** `{self.num_calls}`\n\n"
                f"**Approach Rate:** `{approach_rate}`\n\n"
                f"**Guest:** `{guest}`\n\n"
                f"**Team Details:**\n\n"
                f"```json\n{team_str}\n```"
            )

        except (TypeError, AttributeError, KeyError) as e:
            return f"Error formatting render data: {e}"

    def _print_team_summary(self, data: Dict[str, Any]) -> None:
        """Prints a concise team summary to console."""

        _ = data
        try:
            print("\n--- Best Team Composition ---")
            env = self.training_env.get_attr("unwrapped", indices=[0])[0]
            if hasattr(env, "state") and env.state.team:
                team = env.state.team
                approach_rate = env.state.final_approach_rate
                if approach_rate:
                    print(f"Approach Rate: {approach_rate}")

                if team.guest_manager and team.guest_manager.current_guest:
                    guest = team.guest_manager.current_guest
                    print(f"Guest: Leader Skill ID {guest.leader_skill_id}")
                else:
                    print("Guest: None")

                print("\nTeam Cards:")

                for i, slot in enumerate(team.slots):
                    if slot.card:
                        print(
                            f"  Slot {i+1}: {slot.card.display_name} (Lv.{slot.card.level})"
                        )
                        if slot.accessory:
                            print(f"    - Accessory: {slot.accessory.name}")
                        if slot.sis_list:
                            print(
                                f"    - SIS: {', '.join([sis.name for sis in slot.sis_list])}"
                            )
                    else:
                        print(f"  Slot {i+1}: [Empty]")

                print(
                    f"\nTeam Stats: S/P/C {team.total_team_smile}/{team.total_team_pure}/{team.total_team_cool}"
                )

            print("-----------------------------\n")

        except (AttributeError, IndexError, KeyError) as e:
            if self.verbose > 0:
                print(f"Error printing team summary: {e}")
