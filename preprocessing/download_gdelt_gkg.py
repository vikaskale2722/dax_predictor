import requests
import os
from datetime import datetime, timedelta, time as dt_time
from tqdm import tqdm
import time

# --- Configuration ---
DOWNLOAD_DIR = "data/gdelt_gkg_data"
GDELT_INTERVAL_GKG_BASE_URL = "http://data.gdeltproject.org/gdeltv2/{datetime_str}.gkg.csv.zip"
# --- End Configuration ---

def generate_2_hour_timestamps():
    """Generates HHMMSS strings for every 2 hours in a day."""
    timestamps = []
    for hour in range(0, 24, 2):  # 0, 2, 4, ..., 22
        t = dt_time(hour, 0, 0)
        timestamps.append(t.strftime("%H%M%S"))
    return timestamps

def download_file(url, target_path):
    """Downloads a file from a URL to a target path."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=81920):
                f.write(chunk)
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            pass
        else:
            print(f"HTTP error {e.response.status_code} downloading {url}: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error downloading {url}: {e}")
        return False

if __name__ == "__main__":
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Compute last 26 days automatically
    today = datetime.today()
    start_date = today - timedelta(days=26)
    end_date = today - timedelta(days=1)  # yesterday

    current_date_iter = start_date
    date_list_to_process = []
    while current_date_iter <= end_date:
        date_list_to_process.append(current_date_iter)
        current_date_iter += timedelta(days=1)

    if not date_list_to_process:
        print("No dates to process.")
        exit()

    print(f"Attempting to download 2-hour GDELT GKG files from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Files will be saved in '{DOWNLOAD_DIR}'")

    two_hour_timestamps = generate_2_hour_timestamps()
    total_files_to_attempt = len(date_list_to_process) * len(two_hour_timestamps)

    files_downloaded = 0
    files_skipped = 0
    files_failed = 0

    with tqdm(total=total_files_to_attempt, desc="Downloading GDELT files") as pbar:
        for dt_obj in date_list_to_process:
            date_str = dt_obj.strftime("%Y%m%d")
            for time_str in two_hour_timestamps:
                datetime_str_for_url = date_str + time_str
                file_url = GDELT_INTERVAL_GKG_BASE_URL.format(datetime_str=datetime_str_for_url)
                filename = f"{datetime_str_for_url}.gkg.csv.zip"
                target_filepath = os.path.join(DOWNLOAD_DIR, filename)

                if os.path.exists(target_filepath):
                    files_skipped += 1
                    pbar.update(1)
                    continue

                if download_file(file_url, target_filepath):
                    files_downloaded += 1
                else:
                    files_failed += 1

                pbar.update(1)
                time.sleep(0.1)  # Politeness delay

    print("\n--- GDELT Download Summary ---")
    print(f"Total 2-hour intervals attempted: {total_files_to_attempt}")
    print(f"New files downloaded: {files_downloaded}")
    print(f"Files skipped (already existed): {files_skipped}")
    print(f"Files failed (includes 404s): {files_failed}")
    print(f"Check '{DOWNLOAD_DIR}' for your downloaded files.")
