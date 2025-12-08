import os
import urllib.request
import urllib.parse
import json
import time
import kagglehub
import concurrent.futures

OPENIMAGES_BASE_URL = "https://storage.googleapis.com/openimages/v6"

def download_wikipedia_data(download_dir: str) -> str:
    """
    Ensures the Wikipedia SQLite dataset is downloaded.
    """
    expected_subdir = os.path.join(download_dir, "wikipedia-sqlite-portable-db-huge-5m-rows")
    
    if os.path.isdir(expected_subdir) and len(os.listdir(expected_subdir)) > 0:
        print(f"Using existing SQLite dataset at: {expected_subdir}")
        return expected_subdir

    print("Dataset not found locally. Downloading via kagglehub...")
    try:
        path = kagglehub.dataset_download(
            "christernyc/wikipedia-sqlite-portable-db-huge-5m-rows"
        )
        print(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"KaggleHub reported: {e}")
        if os.path.exists(expected_subdir):
            return expected_subdir
        raise e

def get_year_month_list(start_year, start_month, end_year, end_month):
    """Generates a list of (year, month) tuples within the specified range."""
    ym_list = []
    year = start_year
    month = start_month
    
    while True:
        ym_list.append((year, month))
        
        if year == end_year and month == end_month:
            break
            
        month += 1
        if month > 12:
            month = 1
            year += 1
            
    return ym_list

def download_taxi_data(year_month_list, local_dir="data/nyc_taxi"):
    """
    Downloads NYC Taxi data for the given list of (year, month) tuples,
    adding a 3-minute delay every 6 years.
    """
    os.makedirs(local_dir, exist_ok=True)
    BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_"
    
    print(f"--- Checking for local data in '{local_dir}' ---")
    downloaded_files = []
    
    current_block_start_year = year_month_list[0][0] if year_month_list else None
    
    for i, (year, month) in enumerate(year_month_list):
        month_str = f"{month:02d}"
        file_name = f"yellow_tripdata_{year}-{month_str}.parquet"
        local_path = os.path.join(local_dir, file_name)
        
        if current_block_start_year is not None and year >= current_block_start_year + 6 and month == 1 and i > 0:
            delay_minutes = 3
            print(f"\n--- PAUSING: Applying a {delay_minutes}-minute delay for NYC Taxi data (end of a 6-year block) ---")
            time.sleep(delay_minutes * 60)
            current_block_start_year = year
            print("--- RESUMING Download ---")

        downloaded_files.append(local_path)
        
        if os.path.exists(local_path):
            continue
            
        print(f"  Downloading: {file_name} to {local_dir}/")
        url = f"{BASE_URL}{year}-{month_str}.parquet"
        try:
            urllib.request.urlretrieve(url, local_path)
            print(f"  Success: Downloaded {file_name}")
        except Exception as e:
            print(f"!!! Failed to download {url}: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            downloaded_files.pop()
            
    return sorted(downloaded_files)

def download_aircheck_data(local_dir="data/aircheck_wdr91"):
    """Fetches the signed URL and downloads the WDR91 parquet file."""
    os.makedirs(local_dir, exist_ok=True)
    file_name = "WDR91.parquet"
    local_path = os.path.join(local_dir, file_name)
    
    if os.path.exists(local_path):
        print(f"  Aircheck data already exists at {local_path}")
        return [local_path]

    API_URL = "https://aircheck.ai/api/gcp/getsignedurl?datasetId=Hitgen_standardization_V2/WDR91.parquet&partner=HitGen"
    
    print(f"--- Fetching signed URL from Aircheck API ---")
    try:
        with urllib.request.urlopen(API_URL) as response:
            resp_body = response.read().decode('utf-8')
            
            try:
                data = json.loads(resp_body)
                signed_url = data.get('signedUrl') or data.get('url') or list(data.values())[0]
            except json.JSONDecodeError:
                signed_url = resp_body.strip().strip('"')

        print(f"  Downloading WDR91.parquet to {local_dir}/...")
        urllib.request.urlretrieve(signed_url, local_path)
        print(f"  Success: Downloaded {file_name}")
        return [local_path]

    except Exception as e:
        print(f"!!! Failed to download Aircheck data: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return []

def _download_file(url: str, dest_path: str) -> bool:
    """Downloads a file to the destination path if it does not exist."""
    try:
        tmp_path = f"{dest_path}.tmp"
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, dest_path)
        return True
    except Exception as exc:
        print(f"!!! Failed to download {url}: {exc}")
        tmp_path = f"{dest_path}.tmp"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def download_openimages_detection(
    local_dir: str = "data/cv_openimages",
    annotation_filename: str = "oidv6-train-annotations-bbox.csv",
) -> str:
    os.makedirs(local_dir, exist_ok=True)
    dest_path = os.path.join(local_dir, annotation_filename)
    if os.path.exists(dest_path):
        return dest_path

    mirrors = [
        f"https://raw.githubusercontent.com/cvdfoundation/open-images-dataset/master/dataset/oidv6/{annotation_filename}",
        f"https://storage.googleapis.com/openimages/v6/{annotation_filename}",
    ]

    print(f"  Downloading Open Images detection annotations to {dest_path}...")
    for url in mirrors:
        print(f"    Trying {url}")
        if _download_file(url, dest_path):
            print("  Success!")
            return dest_path

    raise RuntimeError("All OpenImages mirrors failed.")

def download_openimages_class_descriptions(
    local_dir: str = "data/cv_openimages",
    class_filename: str = "oidv6-class-descriptions.csv",
) -> str:
    """Ensures the Open Images class description mapping exists locally."""
    os.makedirs(local_dir, exist_ok=True)
    dest_path = os.path.join(local_dir, class_filename)
    if os.path.exists(dest_path):
        print(f"  Open Images class descriptions already exist at {dest_path}")
        return dest_path

    url = f"{OPENIMAGES_BASE_URL}/{class_filename}"
    print(f"  Downloading Open Images class descriptions to {dest_path}...")
    if _download_file(url, dest_path):
        print("  Success: downloaded class descriptions")
        return dest_path
    raise RuntimeError(f"Failed to download class descriptions from {url}")

if __name__ == "__main__":
    print("--- Executing automated data download (Setup Phase) ---")
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    ac_dir = os.path.join(base_path, "data", "aircheck_wdr91")
    taxi_dir = os.path.join(base_path, "data", "nyc_taxi")
    wiki_dir = os.path.join(base_path, "data", "sqlite_datasets")
    cv_dir = os.path.join(base_path, "data", "cv_openimages")

    os.makedirs(wiki_dir, exist_ok=True)
    os.makedirs(cv_dir, exist_ok=True)

    ym_list = get_year_month_list(2009, 1, 2024, 12)
    
    tasks = [
        (download_aircheck_data, [ac_dir]),
        (download_wikipedia_data, [wiki_dir]),
        (download_openimages_detection, [cv_dir]),
        (download_openimages_class_descriptions, [cv_dir]),
    ]

    print("\n--- Starting Parallel Downloads (Aircheck, Wikipedia, OpenImages) ---")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_task = {
            executor.submit(func, *args): f"Downloading {func.__name__}"
            for func, args in tasks
        }
        
        for future in concurrent.futures.as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                result = future.result()
                print(f"{task_name} completed successfully.")
            except Exception as exc:
                print(f"{task_name} generated an exception: {exc}")
                
    print("\n--- Starting Sequential NYC Taxi Download with Delays ---")
    download_taxi_data(ym_list, taxi_dir)
    
    print("\n--- All Downloads Completed ---")