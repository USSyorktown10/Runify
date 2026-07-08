"""Seed demo data — delegates to the bulk seeder with sensible defaults."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.seed_bulk import run_seed

if __name__ == "__main__":
    run_seed(athletes_count=50, activities_per_athlete=15, clear=False, seed=7)
