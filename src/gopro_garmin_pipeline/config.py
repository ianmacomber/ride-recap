"""Configuration via environment variables and .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Pipeline configuration. All values can be overridden via environment
    variables or a .env file in the project root."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # --- Garmin Connect ---
    garmin_email: str = ""
    garmin_password: str = ""

    # --- Strava ---
    strava_client_id: str = ""
    strava_client_secret: str = ""

    # --- Model provider (one provider powers the entire run) ---
    # gemini (default) | openai. Vision scan, narrative select, and prompt
    # eval all use this provider — never mixed in a single run.
    model_provider: str = "gemini"

    # --- Gemini ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"

    # --- OpenStreetMap ---
    # Nominatim and Overpass require a contact address on every request so
    # they can reach you if a client misbehaves. Set this to your own email
    # before running route derivation; without it the OSM lookups are skipped
    # and origin/road fall back to the CLI flags or the generic defaults.
    # https://operations.osmfoundation.org/policies/nominatim/
    osm_contact_email: str = ""

    # --- Athlete zones ---
    # These are one rider's numbers and they are almost certainly not yours.
    # Highlight detection is tuned against them, so a rider with a different
    # FTP will get systematically wrong results until these are set — too many
    # candidates if your FTP is lower, too few if it's higher. Set both in .env.
    ftp: int = 240                    # Functional Threshold Power (watts)
    max_heart_rate: int = 196         # Maximum heart rate (bpm)

    # --- Highlight defaults ---
    # Derived from ftp / max_heart_rate when left unset, so tuning the two
    # values above is enough for most riders. Override individually to taste.
    power_spike_threshold: float | None = None   # defaults to ~1.45x FTP
    speed_spike_threshold: float = 12.0          # m/s (~27 mph)
    hr_spike_threshold: float | None = None      # defaults to ~87% of max HR

    # --- Paths ---
    data_dir: Path = Path("data")

    def model_post_init(self, __context: object) -> None:
        # Scale the spike thresholds to the rider unless they were set
        # explicitly. 1.45x FTP is roughly the bottom of VO2max/anaerobic
        # work; 0.87x max HR is roughly threshold. Both are deliberately
        # loose — they gate candidate *generation*, not the final cut.
        if self.power_spike_threshold is None:
            self.power_spike_threshold = round(self.ftp * 1.45)
        if self.hr_spike_threshold is None:
            self.hr_spike_threshold = round(self.max_heart_rate * 0.87)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings (cached per process)."""
    return Settings()
