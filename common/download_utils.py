import os
import urllib.request
import time

def get_year_month_list(start_year, start_month, end_year, end_month):
    """
    Generates a list of (year, month) tuples within the specified range.
    """
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
    Downloads NYC Taxi data for the given list of (year, month) tuples.
    Returns a sorted list of all downloaded file paths.
    """
    os.makedirs(local_dir, exist_ok=True)
    BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_"
    
    print(f"--- Checking for local data in '{local_dir}' ---")
    
    downloaded_files = []
    
    for year, month in year_month_list:
        month_str = f"{month:02d}"
        file_name = f"yellow_tripdata_{year}-{month_str}.parquet"
        local_path = os.path.join(local_dir, file_name)
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
            
    print(f"--- Data download check complete. Found {len(downloaded_files)} files. ---")
    return sorted(downloaded_files)