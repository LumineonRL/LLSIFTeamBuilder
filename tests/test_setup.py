import json
import os
from unittest.mock import MagicMock, patch

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from setup import (
    KiraraScraper,
    SchoolidoluScraper,
    DataTransformer,
    assemble_final_json
)

# --- MOCK HTML FIXTURES ---

@pytest.fixture
def mock_kirara_html() -> str:
    """
    Provides a mock HTML content for a Kirara card page.
    Note: The structure includes two divs inside the skill sections to match
    the specific CSS selectors used in the scraper.
    """
    return """
    <html>
        <body>
            <div class="name has-special">
                <small><strong>Initial</strong></small>
                <h2><a><span>Kousaka Honoka</span></a></h2>
            </div>
            <div class="skill"></div>
            <div class="skill"></div>
            <div class="skill"></div>
            <div class="skill"></div>
            <div class="skill">
                <div>Dummy Div</div>
                <div>Skill Text</div>
            </div>
            <div class="skill">
                <div>Dummy Div</div>
                <div>Leader Skill Text</div>
            </div>
        </body>
    </html>
    """

@pytest.fixture
def mock_schoolidolu_html() -> str:
    """Provides a mock HTML content for a Schoolidolu card page."""
    return """
    <html>
        <body>
            <table>
                <tr>
                    <th>Collection</th>
                    <td><a>μ's Initial</a></td>
                </tr>
                 <tr class="hidden-xs">
                    <th>Statistics</th>
                    <td>
                        <div class="col-xs-2 text-Smile">500</div>
                        <div class="col-xs-2 text-Pure">400</div>
                        <div class="col-xs-2 text-Cool">300</div>
                    </td>
                </tr>
            </table>
            <table class="table-player">
                <tr>
                    <td></td>
                    <td>
                        <small>μ's</small>
                        <small>Smile</small>
                        <small>Super Rare</small>
                    </td>
                </tr>
            </table>
            <div class="more levels2 already_collapsed">This is a Promo Card.</div>
        </body>
    </html>
    """

# --- PYTEST FIXTURES FOR DATA ---

@pytest.fixture
def sample_cards_df() -> pl.DataFrame:
    """
    Provides a sample DataFrame mimicking scraped cards data post-character extraction.
    """
    data = {
        "card_id": [1],
        "display_name": ["Initial Honoka"],
        "character": ["Honoka"],
        "rarity": ["UR"],
        "attribute": ["Smile"],
        "idolized_smile": [5000],
        "idolized_pure": [4000],
        "idolized_cool": [4000],
        "unidolized_smile": [4000],
        "unidolized_pure": [3000],
        "unidolized_cool": [3000],
        "is_promo": [False],
        "is_preidolized_non_promo": [False],
        "leader_skill_text": [
            "Raise the team's Smile by 9% of its Pure. "
            "Additional effect: raise the Smile contribution of μ's members by 3%"
        ],
    }
    return pl.DataFrame(data)

@pytest.fixture
def sample_skills_df() -> pl.DataFrame:
    """Provides a sample DataFrame mimicking the scraped skills data."""
    data = {
        "card_id": [1, 1],
        "skill_level": [1, 2],
        "skill_text": [
            "Every 20 notes: 36% chance to restore 1 stamina points.",
            "Every 20 notes: 38% chance to restore 2 stamina points.",
        ],
    }
    return pl.DataFrame(data)


# --- TEST CLASSES ---

# pylint: disable=redefined-outer-name
# class TestScrapers:
#     """Tests for the Scraper base class and its children."""

#     @patch('setup.webdriver.Chrome')
#     @patch('requests.Session.get')
#     def test_kirara_scraper(self, mock_get, _mock_chrome, mock_kirara_html):
#         """Tests the KiraraScraper's static data extraction."""
#         # Arrange: Mock the requests call
#         mock_response = MagicMock()
#         mock_response.status_code = 200
#         mock_response.content = mock_kirara_html.encode('utf-8')
#         mock_get.return_value = mock_response

#         # Act
#         scraper = KiraraScraper()
#         data = scraper.get_card_info(1)

#         # Assert
#         assert data is not None
#         assert data["kirara_title_part"] == "Initial"
#         assert data["character_first_name"] == "Honoka"
#         assert data["leader_skill_text"] == "Leader Skill Text"
#         scraper.quit_driver()

#     @patch('requests.Session.get')
#     def test_schoolidolu_scraper(self, mock_get, mock_schoolidolu_html):
#         """Tests the SchoolidoluScraper's data extraction."""
#         # Arrange
#         mock_response = MagicMock()
#         mock_response.raise_for_status.return_value = None
#         mock_response.content = mock_schoolidolu_html.encode('utf-8')
#         mock_get.return_value = mock_response

#         # Act
#         scraper = SchoolidoluScraper()
#         data = scraper.get_card_info(1)

#         # Assert
#         assert data is not None
#         assert data["schoolidolu_title_part"] == "μ's Initial"
#         assert data["rarity"] == "SR"
#         assert data["attribute"] == "Smile"
#         assert data["idolized_smile"] == 500
#         assert data["is_promo"] is True


# pylint: disable=redefined-outer-name
class TestDataTransformer:
    """Tests for the DataTransformer methods."""

    def test_parse_leader_skill_text(self, sample_cards_df):
        """Tests the parsing of leader skill text."""
        # Act
        transformed_df = DataTransformer.parse_leader_skill_text(sample_cards_df)

        # Assert
        assert "leader_attribute" in transformed_df.columns
        assert transformed_df["leader_attribute"][0] == "Smile"
        assert transformed_df["leader_value"][0] == 0.09
        assert transformed_df["leader_secondary_attribute"][0] == "Pure"
        assert transformed_df["leader_extra_target"][0] == "μ's"
        assert transformed_df["leader_extra_value"][0] == 0.03

    def test_normalize_card_stats(self, sample_cards_df):
        """Tests the normalization of card stats into a long format."""
        # Act
        cards_df, stats_df = DataTransformer.normalize_card_stats(sample_cards_df)

        # Assert
        assert len(cards_df) == 1
        assert "idolized_smile" not in cards_df.columns

        assert len(stats_df) == 2  # One row for idolized, one for unidolized
        assert "is_idolized" in stats_df.columns
        assert stats_df["is_idolized"].to_list() == [True, False]
        assert stats_df.filter(pl.col("is_idolized"))["stat_smile"][0] == 5000
        assert stats_df.filter(~pl.col("is_idolized"))["stat_smile"][0] == 4000

    def test_parse_skill_text(self, sample_skills_df):
        """Tests the parsing of card skill text."""
        # Act
        transformed_df = DataTransformer.parse_skill_text(sample_skills_df)

        # Assert
        expected_df = pl.DataFrame({
            'card_id': [1, 1],
            'skill_level': [1, 2],
            'skill_text': [
                "Every 20 notes: 36% chance to restore 1 stamina points.",
                "Every 20 notes: 38% chance to restore 2 stamina points.",
            ],
            'skill_type': ['Healer', 'Healer'],
            'skill_activation_type': ['Rhythm Icons', 'Rhythm Icons'],
            'skill_target': [None, None],
            'skill_threshold': [20, 20],
            'skill_chance': [0.36, 0.38],
            'skill_value': [1.0, 2.0],
            'skill_duration': [None, None],
        }, schema={
            'card_id': pl.Int64,
            'skill_level': pl.Int64,
            'skill_text': pl.Utf8,
            'skill_type': pl.Utf8,
            'skill_activation_type': pl.Utf8,
            'skill_target': pl.Utf8,
            'skill_threshold': pl.Int64,
            'skill_chance': pl.Float64,
            'skill_value': pl.Float64,
            'skill_duration': pl.Float64,
        })

        # We need to ensure columns are in the same order for comparison
        assert_frame_equal(
            transformed_df.select(expected_df.columns),
            expected_df
        )

# pylint: disable=redefined-outer-name
class TestPipeline:
    """Tests the end-to-end data assembly process."""

    def test_assemble_final_json(self, tmp_path):
        """Tests the final JSON assembly from mock Parquet files."""
        # Arrange: Create dummy data and save as parquet files in a temp directory
        data_dir = tmp_path
        cards_path = data_dir / "cards.parquet"
        stats_path = data_dir / "stats.parquet"
        skills_path = data_dir / "skills.parquet"
        json_path = data_dir / "cards.json"

        # Create data that would be the output of the transformation pipeline
        cards_data = pl.DataFrame({
            "card_id": [101], "display_name": ["Test Card"], "rarity": ["UR"],
            "attribute": ["Cool"], "character": ["Maki"], "is_promo": [False],
            "is_preidolized_non_promo": [False],
            "leader_attribute": ["Cool"], "leader_value": [0.12],
            "leader_secondary_attribute": ["Smile"], "leader_extra_attribute": [None],
            "leader_extra_target": [None], "leader_extra_value": [None]
        })
        stats_data = pl.DataFrame({
            "card_id": [101, 101],
            "is_idolized": [False, True],
            "stat_smile": [3000, 4000], "stat_pure": [4000, 5000],
            "stat_cool": [5000, 6000], "sis_base": [4, 4], "sis_max": [8, 8],
            "image": ["unidolized.png", "idolized.png"],
        })
        skills_data = pl.DataFrame({
            "card_id": [101], "skill_level": [8], "skill_type": ["Scorer"],
            "skill_activation_type": ["Perfects"], "skill_threshold": [25],
            "skill_chance": [0.80], "skill_value": [1200.0], "skill_duration": [None],
            "skill_target": [None],
        })

        cards_data.write_parquet(cards_path)
        stats_data.write_parquet(stats_path)
        skills_data.write_parquet(skills_path)

        # Act
        assemble_final_json(str(cards_path), str(stats_path), str(skills_path), str(json_path))

        # Assert
        assert os.path.exists(json_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            final_data = json.load(f)

        assert len(final_data) == 1
        card = final_data[0]
        assert card["card_id"] == 101
        assert "stats" in card
        assert "unidolized" in card["stats"]
        assert "idolized" in card["stats"]
        assert card["stats"]["idolized"]["cool"] == 6000
        assert "leader_skill" in card
        assert card["leader_skill"]["leader_value"] == 0.12
        assert "skill" in card
        assert card["skill"]["type"] == "Scorer"
