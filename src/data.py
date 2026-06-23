"""
src/data.py
===========
Data acquisition and loading for the NASA C-MAPSS Turbofan Engine Degradation
dataset (FD001 subset).

The C-MAPSS dataset ships as plain whitespace-delimited text files with NO header
row. Each file has 26 columns in a fixed order:

    1  unit_number    -> engine id (1..100 for FD001)
    2  time_cycles    -> operational cycle counter for that engine (starts at 1)
    3  op_setting_1   -> operational setting 1
    4  op_setting_2   -> operational setting 2
    5  op_setting_3   -> operational setting 3
    6  sensor_1       -> sensor measurement 1
    ...
    26 sensor_21      -> sensor measurement 21

FD001 specifics (worth knowing for interviews):
    - 100 engines in train, 100 in test.
    - Single operating condition, single fault mode (HPC degradation).
    - train: each engine runs to failure (last cycle == failure).
    - test : each engine is truncated some time BEFORE failure; the true
             Remaining Useful Life (RUL) at that truncation point is given in
             RUL_FD001.txt (one value per engine, 100 rows).

Source / citation:
    A. Saxena, K. Goebel, D. Simon, and N. Eklund,
    "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,"
    Int. Conf. on Prognostics and Health Management (PHM08), 2008.
    NASA Prognostics Center of Excellence (PCoE) Data Repository.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Resolve paths relative to the project root (parent of this src/ directory) so
# the module works no matter what the current working directory is.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# The three FD001 files we need.
TRAIN_FILE = RAW_DIR / "train_FD001.txt"
TEST_FILE = RAW_DIR / "test_FD001.txt"
RUL_FILE = RAW_DIR / "RUL_FD001.txt"

# --------------------------------------------------------------------------- #
# Column names (26 total) — defined once, reused everywhere.
# --------------------------------------------------------------------------- #
INDEX_COLS = ["unit_number", "time_cycles"]
SETTING_COLS = [f"op_setting_{i}" for i in range(1, 4)]   # op_setting_1..3
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]       # sensor_1..21
COLUMN_NAMES = INDEX_COLS + SETTING_COLS + SENSOR_COLS    # 2 + 3 + 21 = 26

# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
# The official NASA PCoE host is frequently offline, so we pull the three FD001
# files directly from a widely-used public GitHub mirror. Downloading the plain
# .txt files (rather than the multi-dataset zip) keeps this simple and robust;
# we fall back to clear manual instructions if anything goes wrong, never failing
# silently.
_MIRROR_BASE = (
    "https://raw.githubusercontent.com/"
    "hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/"
)

# The three files we need, mapped to their on-disk targets.
_WANTED = {
    "train_FD001.txt": TRAIN_FILE,
    "test_FD001.txt": TEST_FILE,
    "RUL_FD001.txt": RUL_FILE,
}

_MANUAL_INSTRUCTIONS = f"""
============================================================================
Could not automatically download the NASA C-MAPSS dataset.

Please download it manually and place these three files into:
    {RAW_DIR}

Required files:
    - train_FD001.txt
    - test_FD001.txt
    - RUL_FD001.txt

Where to get them:
    1. NASA PCoE Prognostics Data Repository (official, may be intermittent):
       https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
       Download "Turbofan Engine Degradation Simulation Data Set" (CMAPSSData.zip),
       unzip, and copy the three FD001 files above into data/raw/.

    2. Kaggle mirror (requires a free account):
       https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

Then re-run:  poetry run python src/data.py
============================================================================
"""


def download_data(force: bool = False) -> bool:
    """Download the FD001 files into data/raw/ if they are not already present.

    Returns True if all three files are available afterwards, False otherwise.
    Prints manual-download instructions on any failure (never raises).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Skip the download entirely if the files are already on disk.
    if not force and all(f.exists() for f in (TRAIN_FILE, TEST_FILE, RUL_FILE)):
        print(f"[data] FD001 files already present in {RAW_DIR} — skipping download.")
        return True

    print(f"[data] Attempting download from mirror:\n       {_MIRROR_BASE}")
    try:
        # Download each FD001 file directly into data/raw/.
        for name, target in _WANTED.items():
            url = _MIRROR_BASE + name
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                target.write_bytes(resp.read())
            print(f"[data] Downloaded {name}")

        ok = all(f.exists() for f in (TRAIN_FILE, TEST_FILE, RUL_FILE))
        if not ok:
            print("[data] Download completed but expected files are missing.")
            print(_MANUAL_INSTRUCTIONS)
        return ok

    except Exception as exc:  # noqa: BLE001 - we want any failure to fall back gracefully
        print(f"[data] Automatic download failed: {exc!r}")
        print(_MANUAL_INSTRUCTIONS)
        return False


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _read_cmapss_file(path: Path) -> pd.DataFrame:
    """Read one C-MAPSS data file into a DataFrame with proper column names.

    The files are space-delimited with a couple of trailing spaces per line,
    which makes pandas emit two phantom all-NaN columns. We use a generic
    whitespace separator and explicitly assign our 26 column names, which
    discards those trailing empties cleanly.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `download_data()` or follow the manual "
            f"instructions printed by `python src/data.py`."
        )

    df = pd.read_csv(
        path,
        sep=r"\s+",          # one-or-more whitespace = column separator
        header=None,          # files have no header row
        names=COLUMN_NAMES,   # assign our 26 names; trailing blanks are dropped
    )
    return df


def load_train() -> pd.DataFrame:
    """Load train_FD001.txt — 100 engines, each run to failure (~20,631 rows)."""
    return _read_cmapss_file(TRAIN_FILE)


def load_test() -> pd.DataFrame:
    """Load test_FD001.txt — 100 engines, each truncated before failure (~13,096 rows)."""
    return _read_cmapss_file(TEST_FILE)


def load_rul() -> pd.DataFrame:
    """Load RUL_FD001.txt — the true RUL for each test engine at its truncation point.

    The file is a single column of 100 integers (one per test engine, in engine
    order 1..100). We attach an explicit ``unit_number`` so it can be joined to
    the test set later without relying on positional alignment.
    """
    if not RUL_FILE.exists():
        raise FileNotFoundError(
            f"{RUL_FILE} not found. Run `download_data()` or follow the manual "
            f"instructions printed by `python src/data.py`."
        )

    rul = pd.read_csv(RUL_FILE, sep=r"\s+", header=None, names=["RUL"])
    # Engine ids are 1-based and match the row order in the file.
    rul.insert(0, "unit_number", range(1, len(rul) + 1))
    return rul


# --------------------------------------------------------------------------- #
# Smoke test / phase-1 visible output
# --------------------------------------------------------------------------- #
def _summarize(name: str, df: pd.DataFrame) -> None:
    """Print a short summary of a DataFrame for sanity-checking."""
    n_engines = df["unit_number"].nunique() if "unit_number" in df.columns else "n/a"
    print(f"\n{name}: shape={df.shape}, engines={n_engines}")
    print(df.head().to_string())


if __name__ == "__main__":
    print("=" * 76)
    print("NASA C-MAPSS FD001 — Phase 1 data loading")
    print("=" * 76)

    # 1. Ensure the raw files exist (download if needed, else print instructions).
    if not download_data():
        raise SystemExit(1)

    # 2. Load all three and print summaries so the phase has visible output.
    train = load_train()
    test = load_test()
    rul = load_rul()

    _summarize("TRAIN (run-to-failure)", train)
    _summarize("TEST  (truncated)", test)
    _summarize("RUL   (true RUL per test engine)", rul)

    print("\n[data] All FD001 files loaded successfully.")
