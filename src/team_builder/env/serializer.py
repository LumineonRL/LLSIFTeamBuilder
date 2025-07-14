from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional, Union, Sequence

import numpy as np

from src.simulator import Accessory, Card, SIS, GuestData, Song, Note

if TYPE_CHECKING:
    from .config import EnvConfig


class Serializer:
    """
    Handles the serialization of game objects into feature vectors for the
    observation space.
    """

    def __init__(self, config: "EnvConfig"):
        """
        Initializes the Serializer with the environment configuration.
        """
        self.config = config
        self.card_feature_size = self._calculate_card_feature_size()
        self.accessory_feature_size = self._calculate_accessory_feature_size()
        self.sis_feature_size = self._calculate_sis_feature_size()
        self.guest_feature_size = self._calculate_guest_feature_size()
        self.note_feature_size = self._calculate_note_feature_size()
        self.song_feature_size = self._calculate_song_feature_size()

    def _calculate_card_feature_size(self) -> int:
        """Calculates the size of the feature vector for a single card."""
        c = self.config
        parts = [
            1,  # Rarity
            3,  # Stats (Smile, Pure, Cool)
            1,  # SIS Slots
            1,  # LS Value
            1,  # LS Extra Value
            1,  # Skill level
            len(c.skill_type_map),  # Skill Type
            len(c.skill_activation_map),  # Skill Activation
        ]
        return sum(parts)

    def _calculate_accessory_feature_size(self) -> int:
        """Calculates the size of the feature vector for a single accessory."""
        c = self.config
        parts = [
            1,  # Has card_id
            3,  # Stats (Smile, Pure, Cool)
            1,  # Skill Level
            len(c.ACCESSORY_SKILL_TYPE_MAP),  # Skill Type
        ]
        return sum(parts)

    def _calculate_sis_feature_size(self) -> int:
        """Calculates the size of the feature vector for a single SIS."""
        c = self.config
        parts = [
            len(c.SIS_EFFECT_MAP),  # effect (one-hot)
            1,  # slots
            1,  # target (binary)
            1,  # value (normalized)
        ]
        return sum(parts)

    def _calculate_guest_feature_size(self) -> int:
        """Calculates the size of the feature vector for a single guest."""
        parts = [
            1,  # LS Value
            1,  # LS Extra Value
        ]
        return sum(parts)

    def _calculate_note_feature_size(self) -> int:
        """Calculates the size of the feature vector for a single note."""
        return 1 + 1 + 9 + 1 + 1  # time, end, pos, star, swing

    def _calculate_song_feature_size(self) -> int:
        """Calculates the size of the feature vector for a song's metadata."""
        parts = [
            1,  # Length
        ]
        return sum(parts)

    def serialize_attribute(self, attribute_value: Optional[str]) -> np.ndarray:
        """Serializes a specified attribute string value into a one-hot vector."""
        if attribute_value is None:
            return np.zeros(len(self.config.ATTRIBUTE_MAP), dtype=np.float32)
        return self._one_hot(attribute_value, self.config.ATTRIBUTE_MAP)

    def serialize_card(self, card: Card) -> np.ndarray:
        """Converts a Card object into a normalized, flat numpy array."""
        if card is None:
            raise ValueError("Input card cannot be None.")
        try:
            features = []
            c = self.config
            ls = card.leader_skill
            skill = card.skill
            features.append(c.RARITY_MAP.get(card.rarity, 0) / (len(c.RARITY_MAP) - 1))
            features.append(card.stats.smile / c.MAX_STAT_VALUE)
            features.append(card.stats.pure / c.MAX_STAT_VALUE)
            features.append(card.stats.cool / c.MAX_STAT_VALUE)
            features.append(card.current_sis_slots / c.MAX_SIS_SLOTS)
            features.append((ls.value or 0.0) / c.MAX_LS_VALUE)
            features.append((ls.extra_value or 0.0) / c.MAX_LS_VALUE)
            features.append((card.current_skill_level or 0) / c.MAX_SKILL_LEVEL)
            features.extend(self._one_hot(skill.type, c.skill_type_map))
            features.extend(self._one_hot(skill.activation, c.skill_activation_map))
            return np.array(features, dtype=np.float32)
        except (AttributeError, KeyError, TypeError) as e:
            raise ValueError(f"Failed to serialize card_id={card.card_id}: {e}") from e

    def serialize_accessory(self, accessory: Accessory) -> np.ndarray:
        """Converts an Accessory object into a normalized, flat numpy array."""
        if accessory is None:
            raise ValueError("Input accessory cannot be None.")
        try:
            features = []
            c = self.config
            skill = accessory.skill
            features.append(1.0 if accessory.card_id else 0.0)
            features.append(accessory.stats.smile / c.MAX_ACCESSORY_STAT)
            features.append(accessory.stats.pure / c.MAX_ACCESSORY_STAT)
            features.append(accessory.stats.cool / c.MAX_ACCESSORY_STAT)
            features.append(accessory.skill_level / c.MAX_SKILL_LEVEL)
            features.extend(self._one_hot(skill.type, c.ACCESSORY_SKILL_TYPE_MAP))
            return np.array(features, dtype=np.float32)
        except (AttributeError, KeyError, TypeError) as e:
            raise ValueError(
                f"Failed to serialize accessory_id={accessory.accessory_id}: {e}"
            ) from e

    def serialize_sis(self, sis: SIS) -> np.ndarray:
        """Converts a SIS object into a normalized, flat numpy array."""
        if sis is None:
            raise ValueError("Input SIS cannot be None.")
        try:
            features = []
            c = self.config
            features.extend(self._one_hot(sis.effect, c.SIS_EFFECT_MAP))
            features.append(sis.slots / c.MAX_SIS_SLOTS)
            features.append(1.0 if sis.target == "all" else 0.0)
            norm_factor = self._get_sis_norm_factor(sis.effect)
            normalized_value = (
                (sis.value or 0.0) / norm_factor if norm_factor > 0 else 0.0
            )
            features.append(normalized_value)
            return np.array(features, dtype=np.float32)
        except (AttributeError, KeyError, TypeError) as e:
            raise ValueError(f"Failed to serialize sis_id={sis.id}: {e}") from e

    def serialize_guest(self, guest: GuestData) -> np.ndarray:
        """Converts a GuestData object into a normalized, flat numpy array."""
        if guest is None:
            raise ValueError("Input guest cannot be None.")
        try:
            features = []
            c = self.config
            features.append((guest.leader_value or 0.0) / c.MAX_LS_VALUE)
            features.append((guest.leader_extra_value or 0.0) / c.MAX_LS_VALUE)
            return np.array(features, dtype=np.float32)
        except (AttributeError, KeyError, TypeError) as e:
            raise ValueError(
                f"Failed to serialize guest_id={guest.leader_skill_id}: {e}"
            ) from e

    def serialize_note(self, note: "Note") -> np.ndarray:
        """Converts a Note object into a normalized, flat numpy array."""
        c = self.config
        features = []
        features.append(note.start_time / c.MAX_SONG_LENGTH)
        features.append(note.end_time / c.MAX_SONG_LENGTH)
        position_one_hot = np.zeros(9, dtype=np.float32)
        if 1 <= note.position <= 9:
            position_one_hot[note.position - 1] = 1.0
        features.extend(position_one_hot)
        features.append(float(note.is_star))
        features.append(float(note.is_swing))
        return np.array(features, dtype=np.float32)

    def serialize_song(self, song: Optional["Song"]) -> np.ndarray:
        """Converts a Song's metadata into a normalized, flat numpy array."""
        if song is None:
            return np.zeros(self.song_feature_size, dtype=np.float32)
        try:
            features = []
            c = self.config
            features.append(song.length / c.MAX_SONG_LENGTH)
            return np.array(features, dtype=np.float32)
        except (AttributeError, KeyError, TypeError) as e:
            raise ValueError(f"Failed to serialize song_id={song.song_id}: {e}") from e

    def serialize_notes(self, song: Optional["Song"]) -> np.ndarray:
        """Serializes all notes from a song into a (MAX_NOTE_COUNT, 5) array."""
        c = self.config
        notes_features = np.zeros(
            (c.MAX_NOTE_COUNT, self.note_feature_size), dtype=np.float32
        )
        if song is None:
            return notes_features

        for i, note in enumerate(song.notes[: c.MAX_NOTE_COUNT]):
            notes_features[i] = self.serialize_note(note)
        return notes_features

    def serialize_skill_parameters(self, item: Union[Card, Accessory]) -> np.ndarray:
        """
        Serializes skill parameters (threshold, chance, value, duration)
        into a structured (16, 4) numpy array.
        """
        c = self.config
        skill_params = np.zeros((c.MAX_SKILL_LIST_ENTRIES, 4), dtype=np.float32)
        if not hasattr(item, "skill") or item.skill is None:
            return skill_params

        skill = item.skill
        thresholds = self._get_skill_thresholds(item)
        chances = self._get_skill_chances(item)
        values = self._get_skill_values(item)
        durations = self._pad_and_normalize(
            skill.durations, c.MAX_SKILL_LIST_ENTRIES, c.MAX_SKILL_DURATION
        )

        skill_params[:, 0] = thresholds
        skill_params[:, 1] = chances
        skill_params[:, 2] = values
        skill_params[:, 3] = durations
        return skill_params

    def get_multi_hot_vector(
        self,
        mapping: Dict[str, np.ndarray],
        key: Optional[str],
        default_vector: np.ndarray,
    ) -> np.ndarray:
        """Retrieves a multi-hot vector from a mapping."""
        if key is None:
            return default_vector
        return mapping.get(key, default_vector)

    def _get_skill_thresholds(self, item: Union[Card, Accessory]) -> np.ndarray:
        """Helper to get normalized skill thresholds."""
        c = self.config
        skill = item.skill
        if isinstance(item, Card):
            if skill.activation == "Score":
                norm = c.MAX_SKILL_THRESHOLD_SCORE
            elif skill.activation == "Time":
                norm = c.MAX_SKILL_THRESHOLD_TIME
            else:
                norm = c.MAX_SKILL_THRESHOLD_DEFAULT
            return self._pad_and_normalize(
                skill.thresholds, c.MAX_SKILL_LIST_ENTRIES, norm
            )
        elif isinstance(item, Accessory):
            val = (item.skill_threshold or 0.0) / c.MAX_ACC_SKILL_THRESHOLD
            return np.full(c.MAX_SKILL_LIST_ENTRIES, val, dtype=np.float32)
        return np.zeros(c.MAX_SKILL_LIST_ENTRIES, dtype=np.float32)

    def _get_skill_chances(self, item: Union[Card, Accessory]) -> np.ndarray:
        """Helper to get normalized skill chances."""
        c = self.config
        skill = item.skill
        chances_data = skill.chances
        if isinstance(item, Accessory):
            chances_data = [chance / 100.0 for chance in skill.chances]
        return self._pad_and_normalize(chances_data, c.MAX_SKILL_LIST_ENTRIES, 1.0)

    def _get_skill_values(self, item: Union[Card, Accessory]) -> np.ndarray:
        """Helper to get normalized skill values."""
        c = self.config
        skill = item.skill
        skill_values = skill.values

        if isinstance(item, Card):
            if skill.type in ["Appeal Boost", "Skill Rate Up"]:
                norm = c.MAX_SKILL_VALUE_PERCENT
            elif skill.type == "Amplify":
                norm = c.MAX_SKILL_VALUE_AMP
            elif skill.type == "Healer":
                norm = c.MAX_SKILL_VALUE_HEAL
            elif skill.type in ["Combo Bonus Up", "Perfect Score Up"]:
                norm = c.MAX_SKILL_VALUE_FLAT
            else:
                norm = c.MAX_SKILL_VALUE_DEFAULT
        elif isinstance(item, Accessory):
            if skill.type in ["Appeal Boost", "Skill Rate Up"]:
                skill_values = [v / 100.0 for v in skill.values]
                norm = c.MAX_SKILL_VALUE_PERCENT
            elif skill.type == "Amplify":
                norm = c.MAX_SKILL_VALUE_AMP
            elif skill.type == "Healer":
                norm = c.MAX_SKILL_VALUE_HEAL
            elif skill.type in ["Combo Bonus Up", "Perfect Score Up", "Spark"]:
                norm = c.MAX_SKILL_VALUE_FLAT
            else:
                norm = c.MAX_SKILL_VALUE_DEFAULT
        else:
            return np.zeros(c.MAX_SKILL_LIST_ENTRIES, dtype=np.float32)

        return self._pad_and_normalize(skill_values, c.MAX_SKILL_LIST_ENTRIES, norm)

    def _get_sis_norm_factor(self, effect: Optional[str]) -> float:
        """Helper to get the normalization factor for a given SIS effect."""
        c = self.config
        norm_map = {
            "all percent boost": c.MAX_SIS_VALUE_ALL_PERCENT,
            "charm": c.MAX_SIS_VALUE_CHARM,
            "heal": c.MAX_SIS_VALUE_HEAL,
            "self flat boost": c.MAX_SIS_VALUE_SELF_FLAT,
            "self percent boost": c.MAX_SIS_VALUE_SELF_PERCENT,
            "trick": c.MAX_SIS_VALUE_TRICK,
        }
        return norm_map.get(effect, 1.0) if effect else 1.0

    def _one_hot(
        self,
        value: Optional[str],
        mapping: Dict[str, int],
        default: Optional[str] = None,
    ) -> np.ndarray:
        """Creates a one-hot encoded vector from a value and a mapping dict."""
        one_hot = np.zeros(len(mapping), dtype=np.float32)
        key = value or default
        if key is not None and (idx := mapping.get(key)) is not None:
            one_hot[idx] = 1.0
        return one_hot

    def one_hot_mapped(
        self, value: Optional[str], mapping: Dict[str, int]
    ) -> np.ndarray:
        """Creates a one-hot encoded vector for a value that might not be in the mapping."""
        one_hot = np.zeros(len(mapping), dtype=np.float32)
        if value and (idx := mapping.get(value)) is not None:
            one_hot[idx] = 1.0
        return one_hot

    @staticmethod
    def _pad_and_normalize(
        data: Optional[Sequence[Union[int, float]]], max_len: int, norm_factor: float
    ) -> np.ndarray:
        """Returns a zero-padded, normalized array from an input list."""
        padded = np.zeros(max_len, dtype=np.float32)
        if not data:
            return padded
        data_cleaned = [x for x in data if x is not None][:max_len]
        if data_cleaned:
            padded[: len(data_cleaned)] = (
                np.array(data_cleaned, dtype=np.float32) / norm_factor
            )
        return padded
