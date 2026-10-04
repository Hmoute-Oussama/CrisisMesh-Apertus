"""Runtime configuration.

Design note: there are no API keys anywhere in CrisisMesh. The whole pipeline
runs offline against a local Apertus server. This is intentional and is a core
product claim, so the config has no fields for cloud credentials.
"""

from __future__ import annotations

import re
import warnings
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = REPO_ROOT / "prompts"
DATASETS_DIR = REPO_ROOT / "datasets"
SCHEMAS_DIR = PROMPTS_DIR / "schemas"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="", extra="ignore", case_sensitive=False
    )

    # runtime
    crisismesh_env: str = "dev"
    log_level: str = "INFO"
    database_url: str = "sqlite:///./crisismesh.db"

    # Apertus serving. Any OpenAI-compatible endpoint; llama.cpp is the default
    # because it runs on CPU and supports GBNF-constrained decoding.
    apertus_base_url: str = "http://127.0.0.1:8090"
    apertus_model_id: str = "apertus-v1.1-4b-instruct-q4km"
    apertus_model_label: str = "Apertus-v1.1-4B-Instruct"
    apertus_quant: str = "Q4_K_M"
    apertus_family: str = "Apertus"
    apertus_license: str = "Apache-2.0"

    apertus_escalation_id: str = "apertus-8b-instruct-2509-q4km"
    apertus_escalation_label: str = "Apertus-8B-Instruct-2509"

    apertus_host: str = "127.0.0.1"
    apertus_port: int = 8090
    apertus_threads: int = 12
    apertus_ctx: int = 8192

    # Deterministic by design. The official Apertus card recommends
    # temperature 0.8 / top_p 0.9 for general chat; we deviate because
    # CrisisMesh extraction must be reproducible. See docs/MODEL_CARD.md.
    apertus_temperature: float = 0.0
    apertus_top_p: float = 1.0
    apertus_max_tokens: int = 300
    apertus_timeout_s: int = 600

    prompt_version_extraction: str = "extraction_v2"
    prompt_version_conflict: str = "conflict_v1"
    prompt_version_normalization: str = "normalization_v1"
    prompt_version_place: str = "place_v1"

    # Cross-language place identity. Costs one model call per distinct place
    # mention and is what lets a French and an Arabic report be recognised as
    # describing the same bridge. Without it, contradiction detection silently
    # finds nothing in a multilingual corpus.
    enable_entity_resolution: bool = True
    entity_resolution_max_candidates: int = 6

    evidence_confidence_floor: float = 0.45
    dedup_strong_threshold: float = 0.92
    dedup_review_threshold: float = 0.72
    # Pairs of mutually exclusive event types. Members are separated by ':' and
    # pairs by ',' or '|', e.g. "road_blocked:road_open,power_outage:power_restored".
    conflict_status_pairs: str = (
        "road_blocked:road_open,power_outage:power_restored,"
        "water_outage:water_restored,fire:no_fire"
    )

    operator_token: str = "crisismesh-dev-operator"
    retention_days: int = 30

    # ---- derived ----------------------------------------------------------
    @property
    def chat_url(self) -> str:
        return f"{self.apertus_base_url.rstrip('/')}/v1/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.apertus_base_url.rstrip('/')}/v1/models"

    @property
    def health_url(self) -> str:
        return f"{self.apertus_base_url.rstrip('/')}/health"

    @property
    def conflict_pairs(self) -> set[tuple[str, str]]:
        """Symmetric status pairs that constitute a contradiction.

        Format: members separated by ':', pairs separated by ',' or '|'.

        A malformed entry is reported rather than skipped. The earlier version
        of this parser split on '|' and expected ':' inside, so the shipped
        default "road_blocked|road_open" parsed to an empty set and every status
        contradiction went undetected with no error anywhere. A check that can
        only fail by being wrong has to complain when it cannot parse its own
        input.
        """
        out: set[tuple[str, str]] = set()
        for chunk in re.split(r"[,|]", self.conflict_status_pairs or ""):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = [p.strip() for p in chunk.split(":") if p.strip()]
            if len(parts) == 2:
                out.add((parts[0], parts[1]))
                out.add((parts[1], parts[0]))
            else:
                warnings.warn(
                    f"ignoring malformed conflict pair {chunk!r}: expected "
                    f"'type_a:type_b', pairs separated by ',' or '|'",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()


__all__ = ["Settings", "get_settings", "reset_settings_cache", "REPO_ROOT",
           "PROMPTS_DIR", "DATASETS_DIR", "SCHEMAS_DIR", "Field", "os"]