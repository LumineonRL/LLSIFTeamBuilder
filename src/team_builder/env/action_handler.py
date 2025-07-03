import warnings
from typing import Dict, Callable, TYPE_CHECKING

import numpy as np
import gymnasium as gym

from src.simulator import Team
from src.team_builder.env.build_phase import BuildPhase

if TYPE_CHECKING:
    from src.team_builder.env.env import LLSIFTeamBuildingEnv


class ActionHandler:
    """
    Handles action masking and state transitions using an ID-based
    action mapping.

    - The integer `action` directly corresponds to an item's ID (e.g.,
      `deck_id`, `manager_internal_id`).
    - For phases that allow skipping (Accessory, SIS), the "pass" action is
      mapped to the highest possible action value (`action_space.n - 1`).
    """

    def __init__(self, env: "LLSIFTeamBuildingEnv"):
        """
        Initializes the ActionHandler.

        Args:
            env: The main environment instance.
        """
        self.env = env

        self._mask_handlers: Dict[BuildPhase, Callable[[np.ndarray], None]] = {
            BuildPhase.CARD_SELECTION: self._mask_for_card_placement,
            BuildPhase.ACCESSORY_ASSIGNMENT: self._mask_for_accessory_placement,
            BuildPhase.SIS_ASSIGNMENT: self._mask_for_sis_placement,
            BuildPhase.GUEST_SELECTION: self._mask_for_guest_selection,
            BuildPhase.SCORE_SIMULATION: self._mask_for_approach_rate,
        }

        self._action_handlers: Dict[BuildPhase, Callable[[int], None]] = {
            BuildPhase.CARD_SELECTION: self._handle_card_placement,
            BuildPhase.ACCESSORY_ASSIGNMENT: self._handle_accessory_placement,
            BuildPhase.SIS_ASSIGNMENT: self._handle_sis_placement,
            BuildPhase.GUEST_SELECTION: self._handle_guest_selection,
            BuildPhase.SCORE_SIMULATION: self._handle_approach_rate,
        }

    @property
    def _action_space_size(self) -> int:
        """Helper property to safely get the size of the discrete action space."""
        if not isinstance(self.env.action_space, gym.spaces.Discrete):
            raise TypeError(
                f"Expected a Discrete action space, but got {type(self.env.action_space)}."
            )
        return int(self.env.action_space.n)

    @property
    def pass_action(self) -> int:
        """
        The action value reserved for 'pass' actions.
        """
        return self._action_space_size - 1

    # --- Action Masking ---

    def get_action_mask(self) -> np.ndarray:
        """
        Returns a boolean mask indicating valid actions based on the build phase.
        """
        mask = np.zeros(self._action_space_size, dtype=bool)
        try:
            phase = self.env.state.build_phase
            handler = self._mask_handlers[phase]
            handler(mask)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Cannot get action mask for unknown phase '{self.env.state.build_phase}'"
            ) from exc
        return mask

    def _mask_for_card_placement(self, mask: np.ndarray) -> None:
        """Masks valid cards. Action is the card's `deck_id`."""
        if self.env.state.team is None:
            return
        all_deck_ids = set(self.env.deck.entries.keys())
        unassigned_ids = all_deck_ids - self.env.state.team.assigned_deck_ids
        for deck_id in unassigned_ids:
            if deck_id < self.pass_action:
                mask[deck_id] = True

    def _mask_for_accessory_placement(self, mask: np.ndarray) -> None:
        """Masks valid accessories. Action is `manager_internal_id`."""
        if self.env.state.team is None:
            return
        mask[self.pass_action] = True  # 'Pass' is always an option.
        current_slot = self.env.state.team.slots[self.env.state.current_slot_idx]
        if not current_slot.card:
            return

        unassigned_accessories = self.env.accessory_manager.get_unassigned_accessories(
            self.env.state.team.assigned_accessory_ids
        )
        for acc_entry in unassigned_accessories:
            action = acc_entry.manager_internal_id
            if action < self.pass_action:
                with warnings.catch_warnings():
                    # warnings.simplefilter("ignore", UserWarning)
                    is_valid = self.env.state.team.check_accessory_id_restriction(
                        current_slot.card, acc_entry.accessory
                    )
                if is_valid:
                    mask[action] = True

    def _mask_for_sis_placement(self, mask: np.ndarray) -> None:
        """Masks valid SIS. Action is `manager_internal_id`."""
        if self.env.state.team is None:
            return
        mask[self.pass_action] = True
        current_slot = self.env.state.team.slots[self.env.state.current_slot_idx]
        if not current_slot.card:
            return

        equipped_sis_ids = {s.sis.id for s in current_slot.sis_entries}
        unassigned_sis = self.env.sis_manager.get_unassigned_sis(
            self.env.state.team.assigned_sis_ids
        )
        for sis_entry in unassigned_sis:
            action = sis_entry.manager_internal_id
            if action < self.pass_action and sis_entry.sis.id not in equipped_sis_ids:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    is_valid = self.env.state.team.check_sis_equip_restriction(
                        current_slot, sis_entry.sis
                    )
                if is_valid:
                    mask[action] = True

    def _mask_for_guest_selection(self, mask: np.ndarray) -> None:
        """Masks valid guests. Action is the guest's ID."""
        if self.env.enable_guests and self.env.guest_manager:
            for guest_id in self.env.guest_manager.all_guests.keys():
                if guest_id < self.pass_action:
                    mask[guest_id] = True
        else:
            # If guests are disabled, 'pass' is the only option.
            mask[self.pass_action] = True

    def _mask_for_approach_rate(self, mask: np.ndarray) -> None:
        """Masks valid approach rates. Action is the rate's index (0-9)."""
        mask[:10] = True

    # --- State Transitions ---

    def handle_action(self, action: int) -> None:
        """Applies the given action to modify the environment's state."""
        # FIXED: Add explicit action validation
        action_mask = self.get_action_mask()
        if action >= len(action_mask) or not action_mask[action]:
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 20:  # Truncate for readability
                valid_str = (
                    f"{valid_actions[:10].tolist()}...{valid_actions[-10:].tolist()}"
                )
            else:
                valid_str = str(valid_actions.tolist())
            raise ValueError(
                f"Action {action} is not valid. Valid actions: {valid_str}. "
                f"Current phase: {self.env.state.build_phase.name}, "
                f"Current slot: {self.env.state.current_slot_idx + 1}, "
                f"Action space size: {self._action_space_size}"
            )

        try:
            phase = self.env.state.build_phase
            handler = self._action_handlers[phase]
            handler(action)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Invalid action '{action}' for unknown phase '{self.env.state.build_phase}'"
            ) from exc

    def _advance_slot_or_phase(self, next_phase: "BuildPhase") -> None:
        """Helper to advance the current slot or move to the next build phase."""
        self.env.state.current_slot_idx += 1
        if self.env.state.current_slot_idx >= Team.NUM_SLOTS:
            self.env.state.build_phase = next_phase
            self.env.state.current_slot_idx = 0

    def _handle_card_placement(self, action: int) -> None:
        """Handles card placement. Action is the card's `deck_id`."""
        if self.env.state.team is None:
            return
        self.env.state.team.equip_card_in_slot(
            self.env.state.current_slot_idx + 1, action
        )
        self._advance_slot_or_phase(next_phase=BuildPhase.ACCESSORY_ASSIGNMENT)

    def _handle_accessory_placement(self, action: int) -> None:
        """Handles accessory placement. Action is `manager_internal_id` or `pass`."""
        if self.env.state.team is None:
            return
        if action != self.pass_action:
            self.env.state.team.equip_accessory_in_slot(
                self.env.state.current_slot_idx + 1, action
            )
        self._advance_slot_or_phase(next_phase=BuildPhase.SIS_ASSIGNMENT)

    def _handle_sis_placement(self, action: int) -> None:
        """Handles SIS placement. Action is `manager_internal_id` or `pass`."""
        if self.env.state.team is None:
            return
        if action == self.pass_action:
            # If passing, advance to the next slot or phase.
            next_phase = (
                BuildPhase.GUEST_SELECTION
                if self.env.enable_guests
                else BuildPhase.SCORE_SIMULATION
            )
            self._advance_slot_or_phase(next_phase=next_phase)
        else:
            # If equipping, stay on the current slot for potentially more SIS.
            self.env.state.team.equip_sis_in_slot(
                self.env.state.current_slot_idx + 1, action
            )

    def _handle_guest_selection(self, action: int) -> None:
        """Handles guest selection. Action is the guest's ID or `pass`."""
        if (
            action != self.pass_action
            and self.env.enable_guests
            and self.env.guest_manager
        ):
            self.env.guest_manager.set_guest(action)
        # This phase always transitions to the final phase.
        self.env.state.build_phase = BuildPhase.SCORE_SIMULATION

    @staticmethod
    def _handle_approach_rate(action: int) -> None:
        """Terminal action for approach rate selection"""
        pass
