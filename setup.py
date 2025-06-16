import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Type

import polars as pl
import requests
from bs4 import BeautifulSoup, Tag
from lxml import html
from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException, StaleElementReferenceException, TimeoutException
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

class Scraper(ABC):
    """Abstract base class for website scrapers."""

    def __init__(self, base_url: str, site_name: str):
        self.base_url = base_url
        self.site_name = site_name
        self._session = self._create_session()

    @staticmethod
    def _create_session() -> requests.Session:
        """Creates and configures a requests session."""
        session = requests.Session()
        session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36'
            )
        })
        return session

    def _construct_url(self, card_page_id: int) -> str:
        """Constructs the URL for a given card ID."""
        return f"{self.base_url}{card_page_id}"

    def _fetch_page_content(self, url: str) -> bytes | None:
        """Fetches the HTML content of a page."""
        try:
            response = self._session.get(url, timeout=10)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url} for {self.site_name}: {e}")
            return None

    @staticmethod
    def _parse_html_to_soup(html_content: bytes) -> BeautifulSoup | None:
        """Parses HTML content into a BeautifulSoup object."""
        if html_content:
            return BeautifulSoup(html_content, 'html.parser')
        return None

    def get_card_info(self, card_page_id: int) -> dict | None:
        """Fetches, parses, and extracts card information."""
        url = self._construct_url(card_page_id)
        print(f"Fetching data for {self.site_name} (ID: {card_page_id}) from: {url}")

        html_content = self._fetch_page_content(url)
        if not html_content:
            return None

        soup = self._parse_html_to_soup(html_content)
        if not soup:
            return None

        return self._extract_card_data_from_soup(soup, card_page_id)

    @abstractmethod
    def _extract_card_data_from_soup(
        self, soup: BeautifulSoup, card_page_id: int
    ) -> dict | None:
        """Extracts card data from a BeautifulSoup object."""
        raise NotImplementedError

    @abstractmethod
    def get_fields(self) -> list[str]:
        """Returns a list of fields the scraper can extract."""
        raise NotImplementedError

class KiraraScraper(Scraper):
    """Scraper for sif.kirara.ca"""

    def __init__(self):
        super().__init__(
            base_url="https://sif.kirara.ca/card/", site_name="Kirara"
        )
        self.driver = self._initialize_driver()

    @staticmethod
    def _initialize_driver() -> webdriver.Chrome:
        """Initializes and returns a Selenium Chrome WebDriver."""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        return webdriver.Chrome(options=options)

    def get_fields(self) -> list[str]:
        return ["kirara_title_part", "character_first_name", "leader_skill_text"]

    @staticmethod
    def _get_leader_skill_text(soup: BeautifulSoup) -> str | None:
        """Extracts the leader skill text."""
        element = soup.select_one('div.skill:nth-child(6) > div:nth-child(2)')
        return element.get_text(strip=True) if element else None

    def _extract_card_data_from_soup(
        self, soup: BeautifulSoup, card_page_id: int
    ) -> dict | None:
        """Extracts static data that does not require JavaScript interaction."""
        data: dict[str, str | int] = {}
        title_element = soup.select_one('div.name.has-special > small > strong')
        if title_element:
            data["kirara_title_part"] = title_element.get_text(strip=True)

        name_element = soup.select_one('div.name.has-special > h2 > a > span')
        if name_element:
            full_name = name_element.get_text(strip=True)
            name_words = full_name.split()
            if name_words:
                data["character_first_name"] = name_words[-1]

        leader_skill = self._get_leader_skill_text(soup)
        if leader_skill:
            data["leader_skill_text"] = leader_skill

        return data if data else None

    def extract_skill_text_per_level(self, card_page_id: int) -> list[dict]:
        """Uses Selenium to extract skill data for each level."""
        url = self._construct_url(card_page_id)
        print(f"  Kirara: Navigating to {url} for dynamic skill data...")
        self.driver.get(url)

        try:
            skill_id = self._find_skill_id(self.driver.page_source, card_page_id)
            if not skill_id:
                return []

            return self._iterate_through_skill_levels(skill_id)
        except TimeoutException:
            print(
                f"  Kirara: Timeout waiting for skill elements on card ID "
                f"{card_page_id}. It may not have a dynamic skill."
            )
            return []
        except (NoSuchElementException, StaleElementReferenceException) as e:
            print(
                f"  Kirara: A web element was not found for card ID "
                f"{card_page_id}: {e}"
            )
            return []

    @staticmethod
    def _find_skill_id(page_source: str, card_page_id: int) -> str | None:
        """Finds the skill ID from the page source."""
        match = re.search(r'card_skill_vals_init\("(\d+)"\)', page_source)
        if not match:
            print(
                f"  Kirara: Could not find skill initialization script "
                f"for card ID {card_page_id}."
            )
            return None
        return match.group(1)

    def _iterate_through_skill_levels(self, skill_id: str) -> list[dict]:
        """Iterates through skill levels and extracts text."""
        plus_button_locator = (By.ID, f"skill_level_plus{skill_id}")
        level_input_locator = (By.ID, f"skill_level_src{skill_id}")
        skill_text_locator = (
            By.CSS_SELECTOR, 'div.skill:nth-child(5) > div:nth-child(2)'
        )

        plus_button = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located(plus_button_locator)
        )
        plus_button.click()
        WebDriverWait(self.driver, 5).until(
            EC.text_to_be_present_in_element_value(level_input_locator, "1")
        )

        all_skill_data = []
        previous_level = "-1"
        max_skill_level = 16

        for _ in range(max_skill_level):
            level_input = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located(level_input_locator)
            )
            current_level_str = level_input.get_attribute("value")

            if not current_level_str or current_level_str == previous_level:
                print(
                    f"  Kirara: Reached max skill level "
                    f"({previous_level}). Stopping."
                )
                break

            skill_text_element = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located(skill_text_locator)
            )
            skill_text = skill_text_element.text.strip()

            print(f"  Kirara: Recording level {current_level_str} skill text.")
            current_level = int(current_level_str) if current_level_str else 0
            all_skill_data.append(
                {"skill_level": current_level, "skill_text": skill_text}
            )

            previous_level = current_level_str
            self.driver.find_element(*plus_button_locator).click()

            try:
                WebDriverWait(self.driver, 5).until(
                    lambda d: d.find_element(
                        *level_input_locator
                    ).get_attribute("value") != previous_level
                )
            except TimeoutException:
                print(
                    "  Kirara: Timed out waiting for skill level to change, "
                    "likely at max level."
                )
        return all_skill_data

    def quit_driver(self):
        """Quits the Selenium WebDriver."""
        if self.driver:
            print("Quitting Selenium WebDriver for KiraraScraper.")
            self.driver.quit()


class SchoolidoluScraper(Scraper):
    """Scraper for schoolido.lu."""

    def __init__(self):
        super().__init__(
            base_url="https://Schoolidolu.lu/cards/", site_name="Schoolidolu"
        )
        self.rarity_map = {
            "Ultra Rare": "UR", "Super Rare": "SR", "Super Super Rare": "SSR",
            "Rare": "R", "Normal": "N"
        }
        self.attribute_list = ["Cool", "Smile", "Pure", "All"]

    def get_fields(self) -> list[str]:
        return [
            "Schoolidolu_title_part", "rarity", "attribute", "idolized_smile",
            "idolized_pure", "idolized_cool", "unidolized_smile",
            "unidolized_pure", "unidolized_cool", "is_promo"
        ]

    @staticmethod
    def _extract_schoolidolu_title_part(lxml_tree) -> str | None:
        """Extracts the title part from the 'Collection' row."""
        xpath_expr = "//tr[th[normalize-space(.)='Collection']]/td/a[1]"
        elements = lxml_tree.xpath(xpath_expr)
        if elements:
            return elements[0].text_content().strip()
        return None

    def _extract_schoolidolu_rarity_attribute(
        self, soup: BeautifulSoup, card_page_id: int
    ) -> dict:
        """Extracts rarity and attribute from the page."""
        data: dict[str, str | int] = {}
        info_element = self._find_info_element(soup)
        if not info_element:
            print(
                f"  Schoolidolu: Rarity/attribute info element not found for card ID "
                f"{card_page_id}."
            )
            return data

        text_content = info_element.get_text(separator=' ', strip=True)
        data.update(self._parse_rarity(text_content))
        data.update(self._parse_attribute(text_content))
        return data

    @staticmethod
    def _find_info_element(soup: BeautifulSoup) -> Tag | None:
        """Finds the element containing rarity and attribute information."""
        selectors = [
            '.table > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(2) > small:nth-child(3)',
            'table.table-player tr:first-child td:nth-child(2) small:nth-of-type(3)'
        ]
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element

        info_elements = soup.select(
            'table.table-player tr:first-child td:nth-child(2) small'
        )
        if len(info_elements) >= 3:
            return info_elements[2]
        return None

    def _parse_rarity(self, text: str) -> dict:
        """Parses rarity from text content."""
        for long_rarity, short_rarity in self.rarity_map.items():
            if long_rarity in text:
                return {"rarity": short_rarity}
        return {}

    def _parse_attribute(self, text: str) -> dict:
        """Parses attribute from text content."""
        for attr in self.attribute_list:
            if re.search(r'\b' + re.escape(attr) + r'\b', text, re.IGNORECASE):
                return {"attribute": attr}
        return {}

    @staticmethod
    def _get_stats_from_xpath_elements(
        elements: list, stat_name_base: str, card_page_id: int
    ) -> tuple[int | None, int | None]:
        """Extracts idolized and unidolized stats from lxml elements."""
        if not elements:
            return None, None

        try:
            idolized_stat_str = elements[0].text_content().strip()
            idolized_stat = int(idolized_stat_str) if idolized_stat_str else None

            unidolized_stat = None
            if len(elements) == 3:
                unidolized_stat_str = elements[2].text_content().strip()
                unidolized_stat = int(unidolized_stat_str) if unidolized_stat_str else None
            elif len(elements) == 2:
                unidolized_stat = idolized_stat
            return idolized_stat, unidolized_stat
        except (ValueError, IndexError) as e:
            print(
                f"  Schoolidolu: Error processing {stat_name_base} stat for card ID "
                f"{card_page_id}: {e}"
            )
            return None, None

    def _extract_schoolidolu_stats(self, lxml_tree, card_page_id: int) -> dict:
        """Extracts smile, pure, and cool stats."""
        data: dict[str, int] = {}
        stats_to_extract = {
            "smile": "//div[@class='col-xs-2 text-Smile']",
            "pure": "//div[@class='col-xs-2 text-Pure']",
            "cool": "//div[@class='col-xs-2 text-Cool']"
        }
        for stat_key, xpath in stats_to_extract.items():
            elements = lxml_tree.xpath(xpath)
            idolized, unidolized = self._get_stats_from_xpath_elements(
                elements, stat_key, card_page_id
            )
            if idolized is not None:
                data[f"idolized_{stat_key}"] = idolized
            if unidolized is not None:
                data[f"unidolized_{stat_key}"] = unidolized
        return data

    @staticmethod
    def _extract_is_promo(lxml_tree, card_page_id: int) -> bool:
        """Determines if the card is a promo card."""
        xpath = (
            "//div[contains(@class, 'more') and "
            "contains(@class, 'levels2') and "
            "contains(@class, 'already_collapsed')]"
        )
        elements = lxml_tree.xpath(xpath)
        if elements and "Promo Card" in elements[0].text_content():
            print(f"  Schoolidolu: Detected 'Promo Card' for card ID {card_page_id}.")
            return True
        return False

    def _extract_card_data_from_soup(
        self, soup: BeautifulSoup, card_page_id: int
    ) -> dict | None:
        """Extracts all card data from the Schoolidolu page."""
        lxml_tree = html.fromstring(str(soup))
        combined_data: dict[str, str | int | bool] = {}

        title_part = self._extract_schoolidolu_title_part(lxml_tree)
        if title_part:
            combined_data["Schoolidolu_title_part"] = title_part

        rarity_attr_data = self._extract_schoolidolu_rarity_attribute(soup, card_page_id)
        combined_data.update(rarity_attr_data)

        stats_data = self._extract_schoolidolu_stats(lxml_tree, card_page_id)
        combined_data.update(stats_data)

        combined_data["is_promo"] = self._extract_is_promo(lxml_tree, card_page_id)

        return combined_data if combined_data else None

def export_to_parquet(data_list: list[dict], output_path: str, schema: dict):
    """Exports a list of dictionaries to a Parquet file."""
    if not data_list:
        print("No data to export.")
        return

    new_df = pl.DataFrame(data_list, schema=schema)
    try:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        new_df.write_parquet(output_path)
        print(f"DataFrame successfully exported to {output_path}. Rows: {len(new_df)}")
    except (IOError, pl.exceptions.PolarsError) as e:
        print(f"Error during Parquet export to {output_path}: {e}")


def save_dataframe(df: pl.DataFrame, file_path: str):
    """Saves a Polars DataFrame to a Parquet file."""
    print(f"Saving DataFrame to {file_path}...")
    try:
        df.write_parquet(file_path)
        print(f"DataFrame successfully saved to {file_path}")
    except (IOError, pl.exceptions.PolarsError) as e:
        print(f"Error saving DataFrame to {file_path}: {e}")

def _consolidate_data_for_card(
    page_id: int, schoolidolu_data: dict | None, kirara_data: dict | None
) -> dict:
    """Consolidates static data from all scrapers for a single card ID."""
    card_data: dict[str, str | int | bool | None] = {"card_id": page_id}
    schoolidolu_title = kirara_title = character_name = None

    if schoolidolu_data:
        schoolidolu_title = schoolidolu_data.get("Schoolidolu_title_part")
        for key in schoolidolu_data:
            if key in CARDS_SCHEMA:
                card_data[key] = schoolidolu_data[key]

    if kirara_data:
        kirara_title = kirara_data.get("kirara_title_part")
        character_name = kirara_data.get("character_first_name")
        if "leader_skill_text" in kirara_data:
            card_data["leader_skill_text"] = kirara_data["leader_skill_text"]

    title_part = schoolidolu_title or kirara_title
    display_name = None
    if title_part and character_name:
        display_name = f"{title_part} {character_name}".strip()
    elif title_part:
        display_name = str(title_part)
    elif character_name:
        display_name = f"{page_id} {character_name}"

    if display_name:
        card_data["display_name"] = display_name

    for field, field_type in CARDS_SCHEMA.items():
        if field_type == pl.Int64:
            card_data.setdefault(field, 0)
        elif field_type == pl.Boolean:
            card_data.setdefault(field, False)
        else:
            card_data.setdefault(field, None)

    return card_data


def _process_single_card(
    page_id: int, kirara_scraper: KiraraScraper, schoolidolu_scraper: SchoolidoluScraper
) -> tuple[dict | None, list[dict]]:
    """Handles all scraping and data consolidation for a single card ID."""
    print(f"\n--- Processing Card ID: {page_id} ---")

    schoolidolu_page_data = schoolidolu_scraper.get_card_info(page_id)
    time.sleep(0.25)
    kirara_page_data = kirara_scraper.get_card_info(page_id)
    time.sleep(0.25)

    consolidated_card_data = _consolidate_data_for_card(
        page_id, schoolidolu_page_data, kirara_page_data
    )

    skill_levels = kirara_scraper.extract_skill_text_per_level(page_id)
    skill_data_with_ids = [
        {"card_id": page_id, **skill_info} for skill_info in skill_levels
    ]

    return consolidated_card_data, skill_data_with_ids


def run_scraper_session(
    start_id: int,
    end_id: int,
    cards_output_path: str,
    skills_output_path: str
):
    """Manages the entire scraping session."""
    kirara = KiraraScraper()
    schoolidolu = SchoolidoluScraper()
    card_ids_to_process = range(start_id, end_id + 1)
    collected_cards = []
    collected_skills = []

    print(f"Processing card IDs from {start_id} to {end_id}...")
    try:
        for page_id in card_ids_to_process:
            card_data, skill_data = _process_single_card(page_id, kirara, schoolidolu)

            if card_data:
                collected_cards.append(card_data)
            if skill_data:
                collected_skills.extend(skill_data)

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving collected data...")
    except (requests.exceptions.RequestException, pl.exceptions.PolarsError) as e:
        print(f"\nAn error occurred during scraping: {e}. Saving collected data...")
    finally:
        export_to_parquet(collected_cards, cards_output_path, CARDS_SCHEMA)
        export_to_parquet(collected_skills, skills_output_path, CARD_SKILLS_SCHEMA)
        kirara.quit_driver()

CARDS_SCHEMA: dict[str, Type[pl.DataType]] = {
    "card_id": pl.Int64, "display_name": pl.Utf8, "rarity": pl.Utf8,
    "attribute": pl.Utf8, "idolized_smile": pl.Int64, "idolized_pure": pl.Int64,
    "idolized_cool": pl.Int64, "unidolized_smile": pl.Int64,
    "unidolized_pure": pl.Int64, "unidolized_cool": pl.Int64,
    "is_promo": pl.Boolean, "leader_skill_text": pl.Utf8
}

CARD_SKILLS_SCHEMA: dict[str, Type[pl.DataType]] = {
    "card_id": pl.Int64, "skill_level": pl.Int64, "skill_text": pl.Utf8
}

class DataTransformer:
    """A class for data transformation methods on Polars DataFrames."""

    @staticmethod
    def extract_character_from_display_name(df: pl.DataFrame) -> pl.DataFrame:
        """Extracts the character name from the 'display_name' column."""
        print("Extracting character from display_name...")
        return df.with_columns(
            pl.col("display_name").str.split(by=" ").list.last().alias("character")
        )

    @staticmethod
    def filter_null_idolized_smile(df: pl.DataFrame) -> pl.DataFrame:
        """Filters out skill level boosting cards by removing rows where 'idolized_smile' is null."""
        print("Filtering rows with null 'idolized_smile' values...")
        initial_count = len(df)
        filtered_df = df.filter(pl.col("idolized_smile").is_not_null())
        rows_removed = initial_count - len(filtered_df)
        if rows_removed > 0:
            print(f"Removed {rows_removed} rows with fodder cards.")
        return filtered_df

    @staticmethod
    def correct_character_names(df: pl.DataFrame) -> pl.DataFrame:
        """Corrects character names that go First Name + Last Name."""
        print("Correcting specific character names...")
        corrections = {
            "Verde": "Emma",
            "Taylor": "Mia",
            "Zhong": "Lanzhu",
        }
        char_col = pl.col("character")
        for original, corrected in corrections.items():
            char_col = char_col.str.replace_all(original, corrected, literal=True)

        display_name_col = pl.col("display_name")
        for original, corrected in corrections.items():
            display_name_col = display_name_col.str.replace_all(
                original, corrected, literal=True
            )

        return df.with_columns(
            character=char_col, display_name=display_name_col
        )

    @staticmethod
    def apply_bond_bonus(df: pl.DataFrame) -> pl.DataFrame:
        """Applies a bond point bonus to the main stat of each card."""
        print("Applying bond point bonus...")
        idolized_bonus = (
            pl.when(pl.col("rarity") == "N").then(50)
            .when(pl.col("rarity") == "R").then(100)
            .when(pl.col("rarity") == "SR").then(500)
            .when(pl.col("rarity") == "SSR").then(750)
            .when(pl.col("rarity") == "UR").then(1000)
            .otherwise(0).cast(pl.Int64)
        )
        unidolized_bonus = (
            pl.when(pl.col("is_promo")).then(idolized_bonus)
            .otherwise(
                pl.when(pl.col("rarity") == "N").then(25)
                .when(pl.col("rarity") == "R").then(50)
                .when(pl.col("rarity") == "SR").then(250)
                .when(pl.col("rarity") == "SSR").then(375)
                .when(pl.col("rarity") == "UR").then(500)
                .otherwise(0)
            ).cast(pl.Int64)
        )

        df_with_bonuses = df.with_columns(
            unidolized_bonus=unidolized_bonus, idolized_bonus=idolized_bonus
        )

        unidolized_stats = ["unidolized_smile", "unidolized_pure", "unidolized_cool"]
        idolized_stats = ["idolized_smile", "idolized_pure", "idolized_cool"]

        df_with_main_stats = df_with_bonuses.with_columns(
            unidolized_main_stat=pl.max_horizontal(unidolized_stats),
            idolized_main_stat=pl.max_horizontal(idolized_stats),
        )

        update_expressions = []
        for stat in unidolized_stats:
            expr = pl.when(pl.col(stat) == pl.col("unidolized_main_stat")).then(
                pl.col(stat) + pl.col("unidolized_bonus")
            ).otherwise(pl.col(stat))
            update_expressions.append(expr.alias(stat))

        for stat in idolized_stats:
            expr = pl.when(pl.col(stat) == pl.col("idolized_main_stat")).then(
                pl.col(stat) + pl.col("idolized_bonus")
            ).otherwise(pl.col(stat))
            update_expressions.append(expr.alias(stat))

        return df_with_main_stats.with_columns(update_expressions).drop(
            "unidolized_bonus", "idolized_bonus",
            "unidolized_main_stat", "idolized_main_stat"
        )

    @staticmethod
    def parse_leader_skill_text(df: pl.DataFrame) -> pl.DataFrame:
        """Parses the 'leader_skill_text' column into structured data."""
        print("Parsing leader skill text...")
        skill_parts = pl.col("leader_skill_text").fill_null("").str.split(
            "Additional effect:"
        )
        main_part = skill_parts.list.get(0, null_on_oob=True).str.strip_chars()
        add_part = skill_parts.list.get(1, null_on_oob=True).str.strip_chars()

        return df.with_columns(
            leader_attribute=main_part.str.extract(r"Raise the team's (\w+)", 1),
            leader_value=(
                main_part.str.extract(r"by ?(\d+(?:\.\d+)?)%", 1)
                .cast(pl.Float64, strict=False) / 100
            ).round(2),
            leader_secondary_attribute=main_part.str.extract(r"of its (\w+)", 1),
            leader_extra_attribute=add_part.str.extract(
                r"raise the (\w+) contribution", 1
            ),
            leader_extra_target=add_part.str.extract(
                r"contribution of (.+?) members by", 1
            ).str.strip_chars(),
            leader_extra_value=(
                add_part.str.extract(r"members by ?(\d+(?:\.\d+)?)%", 1)
                .cast(pl.Float64, strict=False) / 100
            ).round(2)
        )

    @staticmethod
    def add_preidolized_non_promo_flag(df: pl.DataFrame) -> pl.DataFrame:
        """Adds a boolean flag for pre-idolized non-promo cards. Since schoolidolu 
        classifies them as promo cards even though they function very differently."""
        print("Adding 'is_preidolized_non_promo' flag...")
        required_cols = ["is_promo", "leader_value"]
        if not all(col in df.columns for col in required_cols):
            print("Error: Required columns not found. Skipping transformation.")
            return df

        condition = pl.col("is_promo") & (pl.col("leader_value") > 0.03)
        return df.with_columns(
            is_preidolized_non_promo=pl.when(condition).then(True).otherwise(False)
        )

    @staticmethod
    def normalize_card_stats(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Normalizes card stats into a long format."""
        print("Normalizing card stats...")
        id_vars = [
            "card_id", "display_name", "character", "rarity",
            "is_promo", "is_preidolized_non_promo"
        ]
        idolized_df = df.select(
            *id_vars,
            pl.col("idolized_smile").alias("stat_smile"),
            pl.col("idolized_pure").alias("stat_pure"),
            pl.col("idolized_cool").alias("stat_cool"),
        ).with_columns(is_idolized=pl.lit(True))

        unidolized_df = df.select(
            *id_vars,
            pl.col("unidolized_smile").alias("stat_smile"),
            pl.col("unidolized_pure").alias("stat_pure"),
            pl.col("unidolized_cool").alias("stat_cool"),
        ).with_columns(is_idolized=pl.lit(False))

        card_stats_df = pl.concat([idolized_df, unidolized_df])
        cards_df_normalized = df.drop(
            [
                "idolized_smile", "idolized_pure", "idolized_cool",
                "unidolized_smile", "unidolized_pure", "unidolized_cool"
            ]
        )
        return cards_df_normalized, card_stats_df

    @staticmethod
    def add_image_urls_to_stats(df: pl.DataFrame) -> pl.DataFrame:
        """Adds image URLs to the card_stats DataFrame."""
        print("Adding image URLs to card_stats...")
        base_url = "http://i.schoolidolu.lu/c/"
        unidolized_url = pl.format(
            "{}Round{}.png", pl.lit(base_url) + pl.col("card_id").cast(pl.Utf8), pl.col("character")
        )
        idolized_url = pl.format(
            "{}RoundIdolized{}.png", pl.lit(base_url) + pl.col("card_id").cast(pl.Utf8), pl.col("character")
        )

        return df.with_columns(
            image=pl.when(pl.col("is_idolized")).then(idolized_url).otherwise(
                unidolized_url
            )
        )

    @staticmethod
    def add_sis_slots(df: pl.DataFrame) -> pl.DataFrame:
        """Adds SIS slot information based on card properties."""
        print("Adding SIS slot information...")
        sis_base = (
            pl.when(pl.col("rarity") == "N")
            .then(pl.when(pl.col("is_idolized")).then(1).otherwise(0))
            .when(pl.col("rarity") == "R").then(1)
            .when(
                pl.col("rarity").is_in(["SR", "SSR", "UR"]) &
                pl.col("is_promo") & ~pl.col("is_preidolized_non_promo")
            ).then(2)
            .when(pl.col("rarity") == "SR").then(2)
            .when(pl.col("rarity") == "SSR").then(3)
            .when(pl.col("rarity") == "UR").then(4)
            .otherwise(0)
        )
        sis_max = (
            pl.when(pl.col("rarity") == "N")
            .then(pl.when(pl.col("is_idolized")).then(1).otherwise(0))
            .when(pl.col("rarity") == "R")
            .then(pl.when(pl.col("is_idolized") & ~pl.col("is_promo")).then(2).otherwise(1))
            .when(
                pl.col("rarity").is_in(["SR", "SSR", "UR"]) &
                pl.col("is_promo") & ~pl.col("is_preidolized_non_promo")
            ).then(2)
            .when(pl.col("rarity") == "SR")
            .then(pl.when(pl.col("is_idolized")).then(4).otherwise(2))
            .when(pl.col("rarity") == "SSR")
            .then(pl.when(pl.col("is_idolized")).then(6).otherwise(3))
            .when(pl.col("rarity") == "UR")
            .then(pl.when(pl.col("is_idolized")).then(8).otherwise(4))
            .otherwise(0)
        )

        return df.with_columns(sis_base=sis_base, sis_max=sis_max).drop(
            "rarity", "is_promo", "is_preidolized_non_promo"
        )

    @staticmethod
    def _clean_skill_text(df: pl.DataFrame) -> pl.DataFrame:
        """Cleans the skill_text column by removing trailing level info."""
        print("Cleaning skill text suffix...")
        return df.with_columns(
            skill_text_cleaned=pl.col("skill_text")
            .str.replace(r"(?s)\s*At skill level.*", "")
            .str.strip_chars()
        )

    @staticmethod
    def parse_skill_text(df: pl.DataFrame) -> pl.DataFrame:
        """Parses the 'skill_text' column using a set of regex patterns."""
        print("Parsing skill text...")
        skill_patterns = get_skill_patterns()

        df = df.pipe(DataTransformer._clean_skill_text)

        df = df.with_columns([
            pl.lit(None, dtype=pl.Utf8).alias("skill_type"),
            pl.lit(None, dtype=pl.Utf8).alias("skill_activation_type"),
            pl.lit(None, dtype=pl.Utf8).alias("skill_target"),
            pl.lit(None, dtype=pl.Int64).alias("skill_threshold"),
            pl.lit(None, dtype=pl.Float64).alias("skill_chance"),
            pl.lit(None, dtype=pl.Float64).alias("skill_value"),
            pl.lit(None, dtype=pl.Float64).alias("skill_duration"),
        ])

        for skill_type, activation_type, regex in skill_patterns:
            mask = df["skill_text_cleaned"].str.contains(regex)
            if not mask.any():
                continue

            parsed_struct = df["skill_text_cleaned"].str.extract_groups(regex)

            update_expressions = [
                pl.when(mask).then(pl.lit(skill_type)).otherwise(pl.col("skill_type")).alias("skill_type"),
                pl.when(mask).then(pl.lit(activation_type)).otherwise(pl.col("skill_activation_type")).alias("skill_activation_type"),
            ]

            if "threshold" in parsed_struct.struct.fields:
                update_expressions.append(pl.when(mask).then(parsed_struct.struct.field("threshold").cast(pl.Int64)).otherwise(pl.col("skill_threshold")).alias("skill_threshold"))
            if "chance" in parsed_struct.struct.fields:
                update_expressions.append(pl.when(mask).then((parsed_struct.struct.field("chance").cast(pl.Float64) / 100).round(2)).otherwise(pl.col("skill_chance")).alias("skill_chance"))
            if "duration" in parsed_struct.struct.fields:
                update_expressions.append(pl.when(mask).then(parsed_struct.struct.field("duration").cast(pl.Float64)).otherwise(pl.col("skill_duration")).alias("skill_duration"))
            if "target" in parsed_struct.struct.fields:
                update_expressions.append(pl.when(mask).then(parsed_struct.struct.field("target")).otherwise(pl.col("skill_target")).alias("skill_target"))
            if "value" in parsed_struct.struct.fields:
                value_expr = parsed_struct.struct.field("value").cast(pl.Float64)
                if "% for" in regex:
                    value_expr = (value_expr / 100).round(2)
                update_expressions.append(pl.when(mask).then(value_expr).otherwise(pl.col("skill_value")).alias("skill_value"))

            df = df.with_columns(update_expressions)

        return df.drop("skill_text_cleaned")


def get_skill_patterns() -> list[tuple[str, str, str]]:
    """Returns a list of regex patterns for parsing skill text."""
    return [
        ("Healer", "Rhythm Icons", r"Every (?P<threshold>\d+) notes: (?P<chance>\d+)% chance to restore (?P<value>[\d\.]+) stamina points\."),
        ("Healer", "Combo", r"Every (?P<threshold>\d+)[x×] combo: (?P<chance>\d+)% chance to restore (?P<value>[\d\.]+) stamina points\."),
        ("Healer", "Time", r"Every (?P<threshold>\d+) seconds: (?P<chance>\d+)% chance to restore (?P<value>[\d\.]+) stamina points\."),
        ("Healer", "Perfects", r"Every (?P<threshold>\d+) perfects: (?P<chance>\d+)% chance to restore (?P<value>[\d\.]+) stamina points\."),
        ("Perfect Lock", "Rhythm Icons", r"Every (?P<threshold>\d+) notes: (?P<chance>\d+)% chance to raise the accuracy of all notes.+for (?P<duration>[\d\.]+) seconds\."),
        ("Perfect Lock", "Combo", r"Every (?P<threshold>\d+)[x×] combo: (?P<chance>\d+)% chance to raise the accuracy of all notes.+for (?P<duration>[\d\.]+) seconds\."),
        ("Perfect Lock", "Time", r"Every (?P<threshold>\d+) seconds: (?P<chance>\d+)% chance to raise the accuracy of all notes.+for (?P<duration>[\d\.]+) seconds\."),
        ("Total Trick", "Rhythm Icons", r"Every (?P<threshold>\d+) notes: (?P<chance>\d+)% chance to raise the accuracy of great notes for (?P<duration>[\d\.]+) seconds\."),
        ("Total Trick", "Time", r"Every (?P<threshold>\d+) seconds: (?P<chance>\d+)% chance to raise the accuracy of great notes for (?P<duration>[\d\.]+) seconds\."),
        ("Scorer", "Perfects", r"Every (?P<threshold>\d+) perfects: (?P<chance>\d+)% chance to add (?P<value>[\d\.]+) score points\."),
        ("Scorer", "Rhythm Icons", r"Every (?P<threshold>\d+) notes: (?P<chance>\d+)% chance to add (?P<value>[\d\.]+) score points\."),
        ("Scorer", "Combo", r"Every (?P<threshold>\d+)[x×] combo: (?P<chance>\d+)% chance to add (?P<value>[\d\.]+) score points\."),
        ("Scorer", "Time", r"Every (?P<threshold>\d+) seconds: (?P<chance>\d+)% chance to add (?P<value>[\d\.]+) score points\."),
        ("Scorer", "Score", r"Every (?P<threshold>\d+) score: (?P<chance>\d+)% chance to add (?P<value>[\d\.]+) score points\."),
        ("Scorer", "Star Notes", r"On every perfect star note: (?P<chance>\d+)% chance to add (?P<value>[\d\.]+) score points\."),
        ("Scorer", "Year Group", r"After all other (?P<target>.+?) members' skills activate:\s*(?P<chance>\d+)% chance to add (?P<value>[\d\.]+) score points\."),
        ("Amplify", "Rhythm Icons", r"Every (?P<threshold>\d+) notes: (?P<chance>\d+)% chance to raise the effective level of the next skill that activates by (?P<value>[\d\.]+)\."),
        ("Amplify", "Perfects", r"Every (?P<threshold>\d+) perfects: (?P<chance>\d+)% chance to raise the effective level of the next skill that activates by (?P<value>[\d\.]+)\."),
        ("Amplify", "Combo", r"Every (?P<threshold>\d+)[x×] combo: (?P<chance>\d+)% chance to raise the effective level of the next skill that activates by (?P<value>[\d\.]+)\."),
        ("Combo Bonus Up", "Combo", r"Every (?P<threshold>[\d]+)\.0[x×] combo: (?P<chance>\d+)% chance to raise score gained per note by (?P<value>[\d\.]+),.+? for (?P<duration>[\d\.]+) seconds\.\s*\(Scaling table\.\.\.\)"),
        ("Perfect Score Up", "Perfects", r"Every (?P<threshold>\d+) perfects: (?P<chance>\d+)% chance that perfect notes will give (?P<value>[\d\.]+) more score for (?P<duration>[\d\.]+) seconds\."),
        ("Appeal Boost", "Rhythm Icons", r"Every (?P<threshold>\d+) notes: (?P<chance>\d+)% chance to raise(?: (?P<target_year>first-year|second-year|third-year))? ?(?P<target>.+?)cards' appeal by (?P<value>\d+(?:\.\d+)?)% for (?P<duration>[\d\.]+) seconds\."),
        ("Appeal Boost Solo", "Rhythm Icons", r"Every (?P<threshold>\d+) notes: (?P<chance>\d+)% chance to raise cards' appeal by (?P<value>\d+(?:\.\d+)?)% for (?P<duration>[\d\.]+) seconds\."),
        ("Encore", "Rhythm Icons", r"Every (?P<threshold>\d+) notes: (?P<chance>\d+)% chance to copy the effect of the last activated skill\."),
        ("Encore", "Combo", r"Every (?P<threshold>\d+)[x×] combo: (?P<chance>\d+)% chance to copy the effect of the last activated skill\."),
        ("Encore", "Perfects", r"Every (?P<threshold>\d+) perfects: (?P<chance>\d+)% chance to copy the effect of the last activated skill\."),
        ("Appeal Copy", "Rhythm Icons", r"(?s)Every (?P<threshold>\d+) notes: (?P<chance>\d+)% chance to copy the appeal of another (?:(?P<target_year>first-year|second-year|third-year) )? ?(?P<target_group>.+?)card on the team for (?P<duration>[\d\.]+) seconds\."),
        ("Skill Boost", "Rhythm Icons", r"Every (?P<threshold>\d+) notes: (?P<chance>\d+)% chance to raise other skills' activation chance by (?P<value>\d+(?:\.\d+)?)% for (?P<duration>[\d\.]+) seconds\."),
    ]

def assemble_final_json(
    cards_path: str, stats_path: str, skills_path: str, output_path: str
):
    """Assembles the final nested JSON from transformed Parquet files."""
    print("\n--- Assembling Final JSON ---")
    try:
        cards_df = pl.read_parquet(cards_path)
        stats_df = pl.read_parquet(stats_path)
        skills_df = pl.read_parquet(skills_path)

        stats_agg = _aggregate_stats(stats_df)
        skills_agg = _aggregate_skills(skills_df)
        cards_with_leader_skill = _structure_leader_skill(cards_df)

        final_df = cards_with_leader_skill.join(
            stats_agg, on="card_id", how="left"
        ).join(
            skills_agg, on="card_id", how="left"
        )

        final_records = final_df.select(
            "card_id", "display_name", "rarity", "attribute", "character",
            "is_promo", "is_preidolized_non_promo", "stats", "leader_skill", "skill"
        ).to_dicts()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_records, f, indent=4)

        print(f"Final JSON successfully saved to {output_path}")

    except (FileNotFoundError, pl.exceptions.PolarsError, IOError) as e:
        print(f"An error occurred during JSON assembly: {e}")


def _aggregate_stats(stats_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregates unidolized and idolized stats for each card."""
    unidolized = stats_df.filter(~pl.col("is_idolized")).select(
        "card_id", "stat_smile", "stat_pure", "stat_cool",
        "sis_base", "sis_max", "image"
    ).rename({"stat_smile": "smile", "stat_pure": "pure", "stat_cool": "cool"})

    idolized = stats_df.filter(pl.col("is_idolized")).select(
        "card_id", "stat_smile", "stat_pure", "stat_cool",
        "sis_base", "sis_max", "image"
    ).rename({"stat_smile": "smile", "stat_pure": "pure", "stat_cool": "cool"})

    stats_agg = unidolized.join(idolized, on="card_id", suffix="_idolized")
    return stats_agg.with_columns(
        stats=pl.struct([
            pl.struct([
                "smile", "pure", "cool", "sis_base", "sis_max", "image"
            ]).alias("unidolized"),
            pl.struct([
                pl.col("smile_idolized").alias("smile"),
                pl.col("pure_idolized").alias("pure"),
                pl.col("cool_idolized").alias("cool"),
                pl.col("sis_base_idolized").alias("sis_base"),
                pl.col("sis_max_idolized").alias("sis_max"),
                pl.col("image_idolized").alias("image"),
            ]).alias("idolized")
        ])
    ).select("card_id", "stats")


def _aggregate_skills(skills_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregates skill level data for each card."""
    return (
        skills_df.sort("card_id", "skill_level")
        .group_by("card_id", maintain_order=True)
        .agg(
            level=pl.col("skill_level"),
            type=pl.first("skill_type"),
            activation=pl.first("skill_activation_type"),
            threshold=pl.col("skill_threshold"),
            chance=pl.col("skill_chance"),
            value=pl.col("skill_value"),
            duration=pl.col("skill_duration"),
            target=pl.first("skill_target"),
        )
        .with_columns(skill=pl.struct(pl.all().exclude("card_id")))
        .select("card_id", "skill")
    )


def _structure_leader_skill(cards_df: pl.DataFrame) -> pl.DataFrame:
    """Structures the leader skill data into a nested format."""
    extra_cols = ["leader_extra_attribute", "leader_extra_target", "leader_extra_value"]
    for col in extra_cols:
        if col not in cards_df.columns:
            cards_df = cards_df.with_columns(pl.lit(None).alias(col))

    return cards_df.with_columns(
        extra=pl.struct(extra_cols)
    ).with_columns(
        leader_skill=pl.struct([
            "leader_attribute", "leader_secondary_attribute", "leader_value", "extra"
        ])
    )

def run_transformation_pipeline(
    cards_path: str, skills_path: str, stats_path: str
):
    """Runs the full data transformation pipeline."""
    print("\n--- Starting Data Transformation Pipeline ---")
    try:
        cards_df = pl.read_parquet(cards_path)
        transformed_cards = (
            cards_df.pipe(DataTransformer.filter_null_idolized_smile)
            .pipe(DataTransformer.extract_character_from_display_name)
            .pipe(DataTransformer.correct_character_names)
            .pipe(DataTransformer.apply_bond_bonus)
            .pipe(DataTransformer.parse_leader_skill_text)
            .pipe(DataTransformer.add_preidolized_non_promo_flag)
        )
        cards_final, card_stats = DataTransformer.normalize_card_stats(
            transformed_cards
        )
        card_stats_final = card_stats.pipe(DataTransformer.add_sis_slots).pipe(
            DataTransformer.add_image_urls_to_stats
        ).sort("card_id", "is_idolized")

        save_dataframe(cards_final, cards_path)
        save_dataframe(card_stats_final, stats_path)
        print("--- Cards Transformation Complete ---")

    except (FileNotFoundError, pl.exceptions.PolarsError) as e:
        print(f"An error occurred during cards transformation: {e}")

    try:
        skills_df = pl.read_parquet(skills_path)
        transformed_skills = skills_df.pipe(DataTransformer.parse_skill_text)
        save_dataframe(transformed_skills, skills_path)
        print("--- Skills Transformation Complete ---")

    except (FileNotFoundError, pl.exceptions.PolarsError) as e:
        print(f"An error occurred during skills transformation: {e}")


def main():
    # Configuration
    start_id = 1
    end_id = 3911
    data_dir = "./data"
    cards_output = os.path.join(data_dir, "cards.parquet")
    skills_output = os.path.join(data_dir, "card_skills.parquet")
    stats_output = os.path.join(data_dir, "card_stats.parquet")
    json_output = os.path.join(data_dir, "cards.json")

    # Scrape
    print("--- Starting Scraping Session ---")
    run_scraper_session(start_id, end_id, cards_output, skills_output)
    print("\n--- Scraping Session Finished ---")

    # Transform
    run_transformation_pipeline(cards_output, skills_output, stats_output)
    assemble_final_json(cards_output, stats_output, skills_output, json_output)


if __name__ == "__main__":
    main()
