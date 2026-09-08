import pandas as pd
from pathlib import Path
import shutil
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def ensure_player_data():
    csv_path = DATA_DIR / "player_data.csv"
    zip_path = DATA_DIR / "player_data.csv.zip"

    if csv_path.exists():
        return csv_path

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Neither {csv_path.name} nor {zip_path.name} was found in {DATA_DIR}"
        )

    with zipfile.ZipFile(zip_path, "r") as archive:
        matching_files = [
            name
            for name in archive.namelist()
            if Path(name).name == "player_data.csv"
        ]

        if not matching_files:
            raise FileNotFoundError(
                f"{zip_path.name} does not contain player_data.csv"
            )

        with archive.open(matching_files[0]) as source:
            with csv_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)

    print(f"Extracted {zip_path.name} to {csv_path}")
    return csv_path


def load_data():
    """Load player and transfer data with no scaling applied.

    Standardization used to happen here, before DataProcessor splits the
    data into train/test -- which fits the scaler's mean/std on the full
    dataset, leaking test-set statistics into every downstream feature.
    Numeric features are now scaled inside DataProcessor.fit_scaler(),
    fit on the training split only and applied to test.
    """
    player_data_path = ensure_player_data()
    player_data = pd.read_csv(player_data_path)
    transfer_data = pd.read_csv(DATA_DIR / "transfer_data.csv")
    return player_data, transfer_data
