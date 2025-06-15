import numpy as np
import pandas as pd
import os
import re
import glob
import h5py
from datetime import datetime
from pathlib import Path
from typing import Union

# Constants
SPEED_CHANNEL = [1, 3, 5, 7]  # Speed channels
VOLUME_CHANNEL = [0, 2, 4, 6]  # Volume channels
GRID_ROWS = 495
GRID_COLUMNS = 436
DIRECTIONS = ['NE', 'SE', 'SW', 'NW']

# Function to load an HDF5 file
def load_h5_file(file_path: Union[str, Path]) -> np.ndarray:
    """
    Loads data from an HDF5 file.
    
    Args:
        file_path (str or Path): Path to the HDF5 file.
    
    Returns:
        np.ndarray: Loaded data array.
    """
    with h5py.File(str(file_path) if isinstance(file_path, Path) else file_path, "r") as fr:
        data = fr.get("array")
        return np.array(data)

# Function to extract all HDF5 files for each city
def extract_h5_files(data_folder: Path):
    """
    Extracts all HDF5 files for each city in the data folder.
    
    Args:
        data_folder (Path): Path to the main data folder.
    
    Returns:
        dict: Dictionary of cities and their corresponding file paths.
    """
    input_files = {}
    cities = [m.group(1) for s in glob.glob(os.path.normpath(f"{data_folder}/*/")) if (m := re.search(r".*\\([^\\]+)$", s))]
    for city in cities:
        input_files[city] = sorted((data_folder / city / "training").rglob("*_[8-9]ch.h5"))
    return input_files

# Function to extract the date from a file name
def extract_date(file_path: Union[str, Path]) -> datetime:
    """
    Extracts the date from the file name.
    
    Args:
        file_path (str or Path): Path to the file.
    
    Returns:
        datetime: Extracted date.
    """
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(file_path))
    if match:
        year, month, day = match.groups()
        return datetime.strptime(f"{day}-{month}-{year}", '%d-%m-%Y')
    else:
        raise ValueError("Date not found in file name.")
    
# Function to aggregate data by 15 minutes with median for volume and avg for speed
def aggregate_by_15(data: np.ndarray) -> np.ndarray:
    """
    Aggregates the data by 15-minute intervals. Computes the median for volume and weighted average for speed.
    
    Args:
        data (np.ndarray): Input data with shape (288, 495, 436, 8)
    
    Returns:
        np.ndarray: Aggregated data with shape (96, 495, 436, 8)
    """
    num_intervals = 96  # 4 intervals per hour * 24 hours
    aggregated_data = np.zeros((num_intervals, 495, 436, 8))
    
    for minute in range(num_intervals):
        start_idx = minute * 3  # Start index of the 15-minute interval
        end_idx = (minute + 1) * 3  # End index of the 15-minute interval
        
        # Weighted average for speed channels
        for ch in SPEED_CHANNEL:
            volume_channel = ch - 1  # Corresponding volume channel
            volume_sum = np.sum(data[start_idx:end_idx, :, :, volume_channel], axis=0)
            speed_weighted_sum = np.sum(data[start_idx:end_idx, :, :, ch] * data[start_idx:end_idx, :, :, volume_channel], axis=0)

            # Handle division by zero using np.where
            with np.errstate(divide='ignore', invalid='ignore'):
                aggregated_data[minute, :, :, ch] = np.where(
                    volume_sum != 0,
                    speed_weighted_sum / volume_sum,
                    0  # Default value when volume_sum == 0
                )

            aggregated_data[minute, :, :, volume_channel] = volume_sum 
    
    return aggregated_data


def get_non_nan_indices_table(codi_emo_grid: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame containing rows and columns indices where the values are not NaN, 
    along with the associated 'codi_emo' values.
    
    Args:
        codi_emo_grid (pd.DataFrame): Input DataFrame with possible NaN values.
    
    Returns:
        pd.DataFrame: A DataFrame with columns 'row', 'col', and 'codi_emo' 
                      representing non-NaN indices and the associated values.
    """
    # Create a boolean mask for non-NaN values
    non_nan_indices = codi_emo_grid.notna().stack()

    # Filter to keep only non-NaN entries (True values in the mask)
    non_nan_pairs = [(i, j) for (i, j), value in non_nan_indices.items() if value]
    
    # Create a list of corresponding 'codi_emo' values
    codi_emo_values = [codi_emo_grid.iloc[i, j] for i, j in non_nan_pairs]
    
    # Convert the list of pairs and 'codi_emo' values into a DataFrame
    non_nan_df = pd.DataFrame(non_nan_pairs, columns=['row', 'col'])
    non_nan_df['codi_emo'] = codi_emo_values
    
    return non_nan_df


def create_traffic_dataframe(aggregated_data: np.ndarray, codi_emo_pair: pd.DataFrame, date: datetime) -> pd.DataFrame:
    """
    Fast version: Transform 4D traffic array into 2D DataFrame for non-NaN codi_emo pixels,
    excluding rows where all volume/speed values are 0.
    
    Args:
        aggregated_data (np.ndarray): Shape (96, 495, 436, 8)
        codi_emo_pair (pd.DataFrame): Columns ['row', 'col', 'codi_emo']
        date (str): Date string (e.g. '2019-01-02')
    
    Returns:
        pd.DataFrame: Long format DataFrame with columns:
                      date, hour, row, col, speed_NE, volume_NE, ..., codi_emo
    """
    rows = codi_emo_pair['row'].to_numpy()
    cols = codi_emo_pair['col'].to_numpy()
    codis = codi_emo_pair['codi_emo'].to_numpy()
    n_time = aggregated_data.shape[0]

    # Extract relevant data
    extracted = aggregated_data[:, rows, cols, :]  # shape: (96, N, 8)
    flat_data = extracted.reshape(-1, 8)  # shape: (96*N, 8)

    # Filter where the sum of 8 values (volume+speed) is > 0
    valid_mask = flat_data.sum(axis=1) > 0
    flat_data = flat_data[valid_mask]

    # Repeat & tile metadata
    hour = [f"{t // 4:02d}:{(t % 4) * 15:02d}" for t in range(n_time)]
    hour_repeated = np.repeat(hour, len(rows))[valid_mask]
    dates_repeated = np.repeat([date] * n_time, len(rows))[valid_mask]
    row_col_repeated = np.tile(rows, n_time)[valid_mask]
    col_col_repeated = np.tile(cols, n_time)[valid_mask]
    codi_emo_repeated = np.tile(codis, n_time)[valid_mask]

    # Apply mask to metadata
    df = pd.DataFrame({
        "date": dates_repeated,
        "hour": hour_repeated,
        "row": row_col_repeated,
        "col": col_col_repeated,
        "speed_NE": flat_data[:, 1],
        "volume_NE": flat_data[:, 0],
        "speed_SE": flat_data[:, 3],
        "volume_SE": flat_data[:, 2],
        "speed_SW": flat_data[:, 5],
        "volume_SW": flat_data[:, 4],
        "speed_NW": flat_data[:, 7],
        "volume_NW": flat_data[:, 6],
        "codi_emo": codi_emo_repeated
    })

    return df



if __name__ == "__main__":
    # Define data folder and extract files
    input_files = {}
    data_folder = Path("C:/Users/HP/Documents/Uni/Spring/Data")
    input_files = extract_h5_files(data_folder / "movie")

    # Select a city and its files
    city = "BARCELONA"
    file_paths = input_files.get(city, [])
    file_paths = [fp for fp in file_paths if '2019' in fp.name]
  
    # Load the codi_emo grid data
    codi_emo_grid = pd.read_csv(data_folder / "map/BARCELONA/codi_emo_grid_interpolated.csv", header=None)

    codi_emo_pair = get_non_nan_indices_table(codi_emo_grid)
    codi_emo_pair['codi_emo'] = codi_emo_pair['codi_emo'].astype('int64')

    # Combine all days into a single DataFrame
    all_days_df = []

    for file_path in file_paths:
        print(f"Processing file: {file_path.name}")
        data = load_h5_file(file_path)
        date = extract_date(file_path)
        aggregated_data = aggregate_by_15(data)
        df_day = create_traffic_dataframe(aggregated_data, codi_emo_pair, date)
        all_days_df.append(df_day)

    # Concat all data
    traffic_df = pd.concat(all_days_df, ignore_index=True)

    # Save as CSV or Parquet
    output_path = data_folder / f"matrix/{city}/{city}_traffic_dataframe"
    traffic_df.to_parquet(f"{output_path}.parquet", index=False)
    print(f"Saved full DataFrame to {output_path}")

    # traffic_df.to_csv(f"{output_path}.csv", index=False)
    # print(f"Saved full DataFrame to {output_path}")


