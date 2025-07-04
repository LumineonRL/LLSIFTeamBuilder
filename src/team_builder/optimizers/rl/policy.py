from typing import Dict

import gymnasium as gym
import torch
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ItemEncoder(nn.Module):
    """
    Encodes each item in a sequence using a 1D CNN and an MLP.
    This architecture is designed to capture local patterns within an item's
    feature vector and then project it into a higher-dimensional embedding space.
    """

    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        # A 1D CNN to capture local relationships in the feature vector.
        self.cnn = nn.Conv1d(
            in_channels=input_dim, out_channels=embed_dim, kernel_size=3, padding=1
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, seq_len, embed_dim).
        """
        # Permute to (batch_size, input_dim, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        x = torch.relu(self.cnn(x))
        # Permute back to (batch_size, seq_len, embed_dim)
        x = x.permute(0, 2, 1)

        x = self.mlp(x)
        x = self.dropout(x)
        x = self.layer_norm(x)

        return x


class TransformerEncoderBlock(nn.Module):
    """
    A single block of a Transformer encoder, consisting of multi-head self-attention
    and a feed-forward network with residual connections and layer normalization.
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(src, src, src, need_weights=False)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        ff_output = self.ffn(src)
        src = src + self.dropout(ff_output)
        src = self.norm2(src)
        return src


class SetAggregator(nn.Module):
    """
    Aggregates a set of item embeddings into a single vector using a learnable
    [CLS] token and a stack of Transformer encoder blocks for deep self-attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=embed_dim * 4,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): A batch of item sequences of shape (batch_size, seq_len, embed_dim).
        Returns:
            torch.Tensor: An aggregated representation of shape (batch_size, embed_dim).
        """
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_with_cls = torch.cat((cls_tokens, x), dim=1)

        for block in self.transformer_blocks:
            x_with_cls = block(x_with_cls)

        cls_output = x_with_cls[:, 0, :]
        return cls_output


class LLSIFTeamBuildingNetwork(nn.Module):
    """
    Network Architecture for LLSIF team building environment.
    It processes the dictionary-based observation space into a single
    feature vector suitable for actor-critic policy heads.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        card_embed_dim: int = 256,
        accessory_embed_dim: int = 128,
        sis_embed_dim: int = 64,
        guest_embed_dim: int = 64,
        song_embed_dim: int = 512,
        state_embed_dim: int = 32,
        attention_heads: int = 8,
        attention_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.debug = False

        spaces = observation_space.spaces

        assert isinstance(spaces["deck"], gym.spaces.Box)
        assert isinstance(spaces["accessories"], gym.spaces.Box)
        assert isinstance(spaces["sis"], gym.spaces.Box)
        assert isinstance(spaces["guest"], gym.spaces.Box)
        assert isinstance(spaces["song"], gym.spaces.Box)
        assert isinstance(spaces["build_phase"], gym.spaces.Discrete)
        assert isinstance(spaces["current_slot"], gym.spaces.Discrete)

        card_feat_size = spaces["deck"].shape[1]
        accessory_feat_size = spaces["accessories"].shape[1]
        sis_feat_size = spaces["sis"].shape[1]
        guest_feat_size = spaces["guest"].shape[1]
        song_feat_size = spaces["song"].shape[0]

        build_phase_vocab_size = int(spaces["build_phase"].n)
        current_slot_vocab_size = int(spaces["current_slot"].n)

        self.build_phase_processor = nn.Linear(build_phase_vocab_size, state_embed_dim)
        self.current_slot_processor = nn.Linear(
            current_slot_vocab_size, state_embed_dim
        )

        self.card_encoder = ItemEncoder(card_feat_size, card_embed_dim, dropout)
        self.accessory_encoder = ItemEncoder(
            accessory_feat_size, accessory_embed_dim, dropout
        )
        self.sis_encoder = ItemEncoder(sis_feat_size, sis_embed_dim, dropout)
        self.guest_encoder = ItemEncoder(guest_feat_size, guest_embed_dim, dropout)

        agg_kwargs = {
            "num_heads": attention_heads,
            "num_layers": attention_layers,
            "dropout": dropout,
        }
        self.deck_aggregator = SetAggregator(card_embed_dim, **agg_kwargs)
        self.accessories_aggregator = SetAggregator(accessory_embed_dim, **agg_kwargs)
        self.sis_aggregator = SetAggregator(sis_embed_dim, **agg_kwargs)
        self.guest_aggregator = SetAggregator(guest_embed_dim, **agg_kwargs)
        self.team_cards_aggregator = SetAggregator(card_embed_dim, **agg_kwargs)
        self.team_accessories_aggregator = SetAggregator(
            accessory_embed_dim, **agg_kwargs
        )
        self.team_sis_aggregator = SetAggregator(sis_embed_dim, **agg_kwargs)
        self.team_guest_aggregator = SetAggregator(guest_embed_dim, **agg_kwargs)

        self.song_mlp = nn.Sequential(
            nn.Linear(song_feat_size, song_embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(song_embed_dim * 2, song_embed_dim),
            nn.LayerNorm(song_embed_dim),
        )

        concatenated_dim = (
            card_embed_dim * 2
            + accessory_embed_dim * 2
            + sis_embed_dim * 2
            + guest_embed_dim * 2
            + song_embed_dim
            + state_embed_dim * 2
        )
        self.features_dim = concatenated_dim // 2
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concatenated_dim, concatenated_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(concatenated_dim, self.features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        build_phase_obs = observations["build_phase"].float().squeeze(1)
        current_slot_obs = observations["current_slot"].float().squeeze(1)
        build_phase_embed = self.build_phase_processor(build_phase_obs)
        current_slot_embed = self.current_slot_processor(current_slot_obs)

        song_embed = self.song_mlp(observations["song"])
        deck_agg = self.deck_aggregator(self.card_encoder(observations["deck"]))
        accs_agg = self.accessories_aggregator(
            self.accessory_encoder(observations["accessories"])
        )
        sis_agg = self.sis_aggregator(self.sis_encoder(observations["sis"]))
        guest_agg = self.guest_aggregator(self.guest_encoder(observations["guest"]))
        team_cards_agg = self.team_cards_aggregator(
            self.card_encoder(observations["team_cards"])
        )
        team_accs_agg = self.team_accessories_aggregator(
            self.accessory_encoder(observations["team_accessories"])
        )
        team_sis_agg = self.team_sis_aggregator(
            self.sis_encoder(observations["team_sis"])
        )
        team_guest_agg = self.team_guest_aggregator(
            self.guest_encoder(observations["team_guest"])
        )

        all_features = [
            deck_agg,
            accs_agg,
            sis_agg,
            guest_agg,
            team_cards_agg,
            team_accs_agg,
            team_sis_agg,
            team_guest_agg,
            song_embed,
            build_phase_embed,
            current_slot_embed,
        ]
        concatenated_features = torch.cat(all_features, dim=1)

        final_features = self.fusion_mlp(concatenated_features)

        return final_features


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    A wrapper for the LLSIFTeamBuildingNetwork that integrates with
    Stable Baselines 3's policy architecture.
    """

    def __init__(self, observation_space: gym.spaces.Dict, **kwargs):
        network = LLSIFTeamBuildingNetwork(observation_space, **kwargs)
        features_dim = network.features_dim

        super().__init__(observation_space, features_dim=features_dim)

        self.network = network

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Passes the dictionary of observations through the custom network.

        Args:
            observations: A dictionary of tensors from the environment.

        Returns:
            A single tensor representing the extracted features.
        """
        return self.network(observations)
