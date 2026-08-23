"""Single source of truth: paths, seed, YAML configs, env/secrets.

Import `cfg` (a module-level singleton) everywhere else. Boring on purpose.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

SEED = 42

# ---------------------------------------------------------------------------
# Paths (repo-root relative; this file lives at <root>/src/config.py)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
STIMULI_DIR = DATA_DIR / "stimuli"
RESPONSES_DIR = DATA_DIR / "responses"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

MANIFEST_PATH = DATA_DIR / "manifest.parquet"
FIDELITY_PATH = DATA_DIR / "fidelity.parquet"
RESULTS_PATH = DATA_DIR / "results.parquet"

# ---------------------------------------------------------------------------
# CREMA-D corpus facts
# ---------------------------------------------------------------------------
CREMAD_REPO = "CheyneyComputerScience/CREMA-D"
CREMAD_BRANCH = "master"
# Git LFS: real bytes come from the media host, not raw.githubusercontent.com.
CREMAD_MEDIA_BASE = (
    f"https://media.githubusercontent.com/media/{CREMAD_REPO}/{CREMAD_BRANCH}"
)
CREMAD_RAW_BASE = f"https://raw.githubusercontent.com/{CREMAD_REPO}/{CREMAD_BRANCH}"
CREMAD_TREE_API = (
    f"https://api.github.com/repos/{CREMAD_REPO}/git/trees/{CREMAD_BRANCH}?recursive=1"
)

EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]  # gold label codes
# Map corpus codes -> the lexicon the emotion prompt asks the model to use.
EMOTION_CODE_TO_WORD = {
    "ANG": "angry",
    "DIS": "disgusted",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}
CHANCE = 1.0 / len(EMOTIONS)  # 1/6

# The 12 fixed sentences (codes -> text), from the CREMA-D corpus documentation.
# Fixed lexicon: every speaker says these, so words carry no emotion signal.
SENTENCES = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}


@dataclass
class Config:
    transforms: dict = field(default_factory=dict)
    prompts: dict = field(default_factory=dict)
    # Secrets / runtime knobs from .env
    model_backend: str = "aoai"
    cost_ceiling_usd: float = 25.0
    price_input_per_1m: float = 2.50
    price_output_per_1m: float = 10.00
    env: dict = field(default_factory=dict)

    # --- convenience accessors -------------------------------------------
    @property
    def gen_order(self) -> list[str]:
        """All transforms to generate (Step 2)."""
        return self.transforms["order"]

    @property
    def model_transforms(self) -> list[str]:
        """Transforms actually submitted to the model (subset of gen_order)."""
        return self.transforms["model_transforms"]

    @property
    def emotion_labels(self) -> list[str]:
        return self.prompts["emotion"]["labels"]

    def prompt_text(self, task: str, prompt_id: str) -> str:
        return self.prompts[task]["prompts"][prompt_id]

    def canonical_prompt(self, task: str) -> str:
        return self.prompts[task]["canonical_prompt"]

    def ensure_dirs(self) -> None:
        for d in (
            RAW_DIR,
            STIMULI_DIR,
            RESPONSES_DIR,
            FIGURES_DIR,
            TABLES_DIR,
        ):
            d.mkdir(parents=True, exist_ok=True)


def _load() -> Config:
    load_dotenv(ROOT / ".env")
    transforms = yaml.safe_load((CONFIGS_DIR / "transforms.yaml").read_text())
    prompts = yaml.safe_load((CONFIGS_DIR / "prompts.yaml").read_text())
    return Config(
        transforms=transforms,
        prompts=prompts,
        model_backend=os.getenv("MODEL_BACKEND", "aoai"),
        cost_ceiling_usd=float(os.getenv("COST_CEILING_USD", "25.0")),
        price_input_per_1m=float(os.getenv("PRICE_INPUT_PER_1M", "2.50")),
        price_output_per_1m=float(os.getenv("PRICE_OUTPUT_PER_1M", "10.00")),
        env=dict(os.environ),
    )


cfg = _load()
