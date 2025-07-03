from typing import Dict, Any, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


class TeamLogCallback(BaseCallback):
    """
    Callback that tracks the best reward, logs detailed team data,
    and all other standard Stable Baselines3 metrics.
    """

    def __init__(self, verbose: int = 0):
        """
        Initializes the callback.
        Args:
            verbose: The verbosity level: 0 for no output, 1 for info messages, 2 for debug messages.
        """
        super().__init__(verbose)
        self.best_reward = -np.inf
        self.best_team_data: Optional[Dict[str, Any]] = None
        self.tb_writer = None

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        It's a good place to get the TensorBoard writer instance.
        """
        for output_format in self.logger.output_formats:
            if isinstance(output_format, TensorBoardOutputFormat):
                self.tb_writer = output_format.writer
                break
        if not self.tb_writer and self.verbose > 1:
            print("TensorBoard writer not found. Skipping text log.")

    def _on_step(self) -> bool:
        """
        This method is called after each step in the environment.
        It checks for completed episodes and logs team data if a new best
        reward is found.
        """
        if "dones" in self.locals and np.any(self.locals["dones"]):
            for i, done in enumerate(self.locals["dones"]):
                if not done:
                    continue

                info = self.locals["infos"][i]
                if "episode" in info:
                    episode_reward = info["episode"]["r"]

                    if episode_reward > self.best_reward:
                        self.best_reward = episode_reward

                        self.logger.record("custom/best_reward", self.best_reward)

                        if "final_team_data" in info:
                            self.best_team_data = info["final_team_data"]
                            if self.verbose > 0:
                                print(
                                    f"\n>>> Step {self.num_timesteps}: New Best Score Found: {self.best_reward:,.2f} <<<"
                                )
                            self._log_best_team_composition()
                        elif self.verbose > 0:
                            print(
                                f"Warning: New best score {self.best_reward} found at step {self.num_timesteps}, "
                                "but 'final_team_data' was not in the info dict."
                            )
        self.logger.record("custom/best_reward", self.best_reward)
        return True

    def _log_best_team_composition(self):
        """
        Logs the best team composition found so far to the console and
        to TensorBoard as richly formatted Markdown text.
        """
        if not self.best_team_data:
            if self.verbose > 0:
                print("No best team data available to log.")
            return

        if self.verbose > 0:
            console_text = self._format_for_console(self.best_team_data)
            print(console_text)

        if self.tb_writer:
            markdown_text = self._format_for_tensorboard(self.best_team_data)
            self.tb_writer.add_text(
                "Best Team Composition", markdown_text, global_step=self.num_timesteps
            )

    def _format_for_tensorboard(self, team_data: Dict[str, Any]) -> str:
        """
        Formats team details as rich Markdown for an excellent display
        in TensorBoard's text panel.
        """
        stats = team_data.get("team_stats", {})
        guest = team_data.get("guest", "None")
        final_rate = team_data.get("final_approach_rate", "N/A")

        markdown_parts = [
            f"# Best Team (Score: {self.best_reward:,.2f})",
            "---",
            f"**Guest:** `{guest}`",
            f"**Total Stats (S/P/C):** `{stats.get('smile', 0)}/{stats.get('pure', 0)}/{stats.get('cool', 0)}`",
            f"**Final Approach Rate:** `{final_rate}`",
            "---",
            "## Team Details",
        ]

        for slot in team_data.get("slots", []):
            slot_stats = slot.get("stats", {})
            markdown_parts.extend(
                [
                    f"### Slot {slot.get('slot_number', 'N/A')}",
                    f"- **Card:** {slot.get('card_name', 'Unknown')} (Deck ID: {slot.get('deck_id', 'N/A')})",
                    f"- **Stats (S/P/C):** {slot_stats.get('smile', 0)}/{slot_stats.get('pure', 0)}/{slot_stats.get('cool', 0)}",
                ]
            )
            acc_name = slot.get("accessory", "None")
            acc_id = slot.get("accessory_id")
            if acc_id is not None:
                markdown_parts.append(
                    f"- **Accessory:** {acc_name} (Manager ID: {acc_id})"
                )
            else:
                markdown_parts.append(f"- **Accessory:** {acc_name}")

            sis_used = slot.get("sis_slots_used", 0)
            sis_max = slot.get("sis_slots_max", 0)
            markdown_parts.append(f"- **SIS ({sis_used}/{sis_max} slots):**")

            sis_list = slot.get("sis_list", [])
            if sis_list:
                for sis in sis_list:
                    markdown_parts.append(f"  - `{sis}`")
            else:
                markdown_parts.append("  - *None*")

        return "\n".join(markdown_parts)

    def _format_for_console(self, team_data: Dict[str, Any]) -> str:
        """
        Formats team details from a dictionary for clear, multi-line printing
        to the standard console.
        """
        lines = ["--- Team Configuration ---"]
        lines.append(f"Guest: {team_data.get('guest', 'None')}")
        stats = team_data.get("team_stats", {})
        lines.append(
            f"Total Stats: S/P/C {stats.get('smile', 0)}/{stats.get('pure', 0)}/{stats.get('cool', 0)}"
        )
        for slot in team_data.get("slots", []):
            lines.append(f"\n[ Slot {slot.get('slot_number', 'N/A')} ]")
            lines.append(
                f"  Card: {slot.get('card_name', 'Unknown')} (Deck ID: {slot.get('deck_id', 'N/A')})"
            )
            slot_stats = slot.get("stats", {})
            lines.append(
                f"  Stats: S/P/C {slot_stats.get('smile', 0)}/{slot_stats.get('pure', 0)}/{slot_stats.get('cool', 0)}"
            )
            acc_name = slot.get("accessory", "None")
            acc_id = slot.get("accessory_id")
            accessory_str = f"  Accessory: {acc_name}"
            if acc_id is not None:
                accessory_str += f" (Manager ID: {acc_id})"
            lines.append(accessory_str)
            sis_used = slot.get("sis_slots_used", 0)
            sis_max = slot.get("sis_slots_max", 0)
            lines.append(f"  SIS ({sis_used}/{sis_max} slots used):")
            sis_list = slot.get("sis_list", [])
            if sis_list:
                for sis_name in sis_list:
                    lines.append(f"    - {sis_name}")
        lines.append("--------------------------")
        final_rate = team_data.get("final_approach_rate")
        if final_rate is not None:
            lines.append(f"Final Approach Rate: {final_rate}")
        return "\n".join(lines)

    def on_training_end(self) -> None:
        """
        Logs the final best reward and total steps when training ends.
        """
        if self.verbose > 0:
            print(f"\nTraining ended. Final best reward: {self.best_reward:.2f}")
            print(f"Total steps completed: {self.num_timesteps}")

        self.logger.record("custom/final_best_reward", float(self.best_reward))
        self.logger.record("custom/total_steps", self.num_timesteps)
