import os
import kagglehub

def ensure_sqlite_dataset(download_dir: str) -> str:
    """
    Ensures the Wikipedia SQLite dataset is downloaded.
    If the dataset already exists in download_dir, it is reused.
    Otherwise, it is downloaded via kagglehub.
    Returns: absolute path to dataset directory.
    """
    expected_subdir = os.path.join(download_dir, "wikipedia-sqlite-portable-db-huge-5m-rows")
    
    if os.path.isdir(expected_subdir) and len(os.listdir(expected_subdir)) > 0:
        print(f"Using existing SQLite dataset at: {expected_subdir}")
        return expected_subdir

    print("Dataset not found locally. Downloading via kagglehub...")
    path = kagglehub.dataset_download(
        "christernyc/wikipedia-sqlite-portable-db-huge-5m-rows"
    )

    print(f"Dataset downloaded to: {path}")
    return path

