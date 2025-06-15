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

# BERLIN 150 TO GRID_ROWS AND 0 TO GRID_COLUMNS
# BARCELONA 150 TO GRID_ROWS AND 0 TO GRID_COLUMNS - 50
# ANTWERP 100 TO GRID_ROWS - 100 AND 75 TO GRID_COLUMNS - 50

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
    

# Function to aggregate data by hour
def aggregate_by_hour_V1(data: np.ndarray) -> np.ndarray:
    """
    Aggregates the data by hour by averaging over 12 intervals (5-minute intervals).
    
    Args:
        data (np.ndarray): Input data with shape (188, 495, 436, 8)
    
    Returns:
        np.ndarray: Aggregated data with shape (24, 495, 436, 8)
    """
    num_hours = 24
    aggregated_data = np.zeros((num_hours, 495, 436, 8))
    
    for hour in range(num_hours):
        start_idx = hour * 12  # Start index of the hour
        end_idx = (hour + 1) * 12  # End index of the hour
        aggregated_data[hour] = data[start_idx:end_idx].mean(axis=0)  # Average over 12 intervals
    
    return aggregated_data

# Function to aggregate data by hour with median for volume and avg for speed
def aggregate_by_hour_V2(data: np.ndarray) -> np.ndarray:
    """
    Aggregates the data by hour for each channel. Computes the median for volume and average for speed.
    
    Args:
        data (np.ndarray): Input data with shape (188, 495, 436, 8)
    
    Returns:
        np.ndarray: Aggregated data with shape (24, 495, 436, 8)
    """
    num_hours = 24
    aggregated_data = np.zeros((num_hours, 495, 436, 8))
    
    for hour in range(num_hours):
        start_idx = hour * 12  # Start index of the hour
        end_idx = (hour + 1) * 12  # End index of the hour
        
        # Median for volume channels
        for ch in VOLUME_CHANNEL:
            aggregated_data[hour, :, :, ch] = np.median(data[start_idx:end_idx, :, :, ch], axis=0)
        
        # Weighted average for speed channels
        for ch in SPEED_CHANNEL:
            aggregated_data[hour, :, :, ch] = np.mean(data[start_idx:end_idx, :, :, ch], axis=0)
    
    return aggregated_data

# Function to create a DataFrame from aggregated data
def create_dataframe_V1(aggregated_data: np.ndarray, date: datetime) -> pd.DataFrame:
    """
    Creates a DataFrame from the aggregated data.
    
    Args:
        aggregated_data (np.ndarray): Aggregated data with shape (24, 495, 436, 8)
        date (datetime): Date of the data.
    
    Returns:
        pd.DataFrame: DataFrame with rows as date-hour and columns as grid cells and directions.
    """
    num_hours = aggregated_data.shape[0]
    rows = []
    for hour in range(num_hours):
        date_str = date
        hour_str = hour

        for row in range(GRID_ROWS):
            for col in range(GRID_COLUMNS):
                if (row >= 150) and (col < (GRID_COLUMNS - 50)):
                    row_data = [date_str, hour_str, row, col] + list(aggregated_data[hour, row, col, :])
                    rows.append(row_data)

    columns = ["Date", "Hour", "Dimension1", "Dimension2"] + [f"Channel_{i+1}" for i in range(aggregated_data.shape[3])]
    df = pd.DataFrame(rows, columns=columns)
    df = df[df.iloc[:, 4:].sum(axis=1) != 0]
    return df

def create_dataframe_V2(aggregated_data: np.ndarray, date: datetime) -> pd.DataFrame:
    """
    Crée un DataFrame où chaque cellule est une colonne sous la forme 'row_col_{Speed|Volume}_{Direction}'.
    L'index des lignes est 'date_heure'.
    
    Args:
        aggregated_data (np.ndarray): Données agrégées de forme (24, 495, 436, 8).
        date (datetime): Date des données.
    
    Returns:
        pd.DataFrame: DataFrame restructuré.
    """
    num_hours = aggregated_data.shape[0]
    data_dict = {}

    for hour in range(num_hours):
        date_hour_str = f"{date.strftime('%d/%m/%Y')}_{hour:02d}h"

        for row in range(150, GRID_ROWS):
            for col in range(GRID_COLUMNS - 50):
                for i, direction in enumerate(DIRECTIONS):
                    volume_value = aggregated_data[hour, row, col, VOLUME_CHANNEL[i]]
                    speed_value = aggregated_data[hour, row, col, SPEED_CHANNEL[i]]

                    if date_hour_str not in data_dict:
                        data_dict[date_hour_str] = {}

                    data_dict[date_hour_str]["date_hour"] = date_hour_str
                    data_dict[date_hour_str][f"{row}_{col}_Volume_{direction}"] = volume_value
                    data_dict[date_hour_str][f"{row}_{col}_Speed_{direction}"] = speed_value

    # Converte df
    df = pd.DataFrame.from_dict(data_dict, orient="index")
    df.index.name = "Date_Hour"  

    return df

# Function to aggregate data by hour with median for volume and avg for speed
def aggregate_by_hour_V3(data: np.ndarray) -> np.ndarray:
    """
    Aggregates the data by hour for each channel. Computes the median for volume and average for speed.
    
    Args:
        data (np.ndarray): Input data with shape (188, 495, 436, 8)
    
    Returns:
        np.ndarray: Aggregated data with shape (24, 495, 436, 8)
    """
    num_hours = 24
    aggregated_data = np.zeros((num_hours, 495, 436, 8))
    
    for hour in range(num_hours):
        start_idx = hour * 12  # Start index of the hour
        end_idx = (hour + 1) * 12  # End index of the hour
        
        # Median for volume channels
        for ch in VOLUME_CHANNEL:
            aggregated_data[hour, :, :, ch] = np.median(data[start_idx:end_idx, :, :, ch], axis=0)
        
        # Weighted average for speed channels
        for ch in SPEED_CHANNEL:
            volume_channel = ch - 1  # Corresponding volume channel
            volume_sum = np.sum(data[start_idx:end_idx, :, :, volume_channel], axis=0)
            speed_weighted_sum = np.sum(data[start_idx:end_idx, :, :, ch] * data[start_idx:end_idx, :, :, volume_channel], axis=0)

            # Handle division by zero using np.where
            with np.errstate(divide='ignore', invalid='ignore'):
                aggregated_data[hour, :, :, ch] = np.where(
                    volume_sum != 0,
                    speed_weighted_sum / volume_sum,
                    0  # Default value when volume_sum == 0
                )
    
    return aggregated_data

# Optimized function to create a DataFrame from aggregated data
def create_dataframe_V3(aggregated_data: np.ndarray, date: datetime) -> pd.DataFrame:
    """
    Creates a DataFrame where each cell is a column under the format 'row_col_{Speed|Volume}_{Direction}'.
    The index of the rows is 'Date_Hour'.

    Args:
        aggregated_data (np.ndarray): Aggregated data with shape (24, 495, 436, 8).
        date (datetime): Date of the data.

    Returns:
        pd.DataFrame: Restructured DataFrame.
    """
    num_hours = aggregated_data.shape[0]
    rows = []  # List to store all row dictionaries

    # Precompute column names for faster access
    column_names = {}
    for row in range(150, GRID_ROWS):
        for col in range(GRID_COLUMNS - 50):
            for i, direction in enumerate(DIRECTIONS):
                key_volume = f"{row}_{col}_Volume_{direction}"
                key_speed = f"{row}_{col}_Speed_{direction}"
                column_names[(row, col, i, 'volume')] = key_volume
                column_names[(row, col, i, 'speed')] = key_speed

    # Populate rows
    for hour in range(num_hours):
        date_hour_str = f"{date.strftime('%d/%m/%Y')}_{hour:02d}h"
        row_data = {"Date_Hour": date_hour_str}

        for row in range(100, GRID_ROWS-150):
            for col in range(100,GRID_COLUMNS-50):
                for i, _ in enumerate(DIRECTIONS):
                    row_data[column_names[(row, col, i, 'volume')]] = aggregated_data[hour, row, col, VOLUME_CHANNEL[i]]
                    row_data[column_names[(row, col, i, 'speed')]] = aggregated_data[hour, row, col, SPEED_CHANNEL[i]]

        rows.append(row_data)

    # Create DataFrame directly from the list of dictionaries
    df = pd.DataFrame(rows)
    return df


# Function to aggregate data by hour with median for volume and avg for speed
def aggregate_by_hour_V4(data: np.ndarray) -> np.ndarray:

    """
    Aggregates the data by hour. Computes the median for volume and average for speed.
    
    Args:
        data (np.ndarray): Input data with shape (188, 495, 436, 8)
    
    Returns:
        np.ndarray: Aggregated data with shape (24, 495, 436, 2) 0 for volume and 1 for speed
    """
    num_hours = 24
    aggregated_data = np.zeros((num_hours, 495, 436, 2))
    
    for hour in range(num_hours):
        start_idx = hour * 12  # Start index of the hour
        end_idx = (hour + 1) * 12  # End index of the hour
        
        # Mean for volume channels
        aggregated_data[hour, :, :, 0] = np.mean(data[start_idx:end_idx, :, :, VOLUME_CHANNEL], axis=(0, -1))
        
        # Weighted average for speed channels
        volume_sum = np.sum(data[start_idx:end_idx, :, :, VOLUME_CHANNEL], axis=(0, -1))
        speed_weighted_sum = np.sum(
            data[start_idx:end_idx, :, :, SPEED_CHANNEL] * data[start_idx:end_idx, :, :, VOLUME_CHANNEL],
            axis=(0, -1)
        )
        
        with np.errstate(divide='ignore', invalid='ignore'):
            aggregated_data[hour, :, :, 1] = np.where(
                volume_sum != 0,
                speed_weighted_sum / volume_sum,
                0
            )        
    
    return aggregated_data

# Optimized function to create a DataFrame from aggregated data
def create_dataframe_V4(aggregated_data: np.ndarray, date: datetime) -> pd.DataFrame:
    """
    Creates a DataFrame where each cell is a column in the form 'row_col_{Speed|Volume}'.
    The row index is 'date_time'.
    
    Args:
        aggregated_data (np.ndarray): Shape aggregated data (24, 495, 436, 2).
        date (datetime): Date of data.
    
    Returns:
        pd.DataFrame: Restructured DataFrame.
    """
    num_hours = aggregated_data.shape[0]
    data_dict = {}

    for hour in range(num_hours):
        date_hour_str = f"{date.strftime('%d/%m/%Y')}_{hour:02d}h"
        data_dict[date_hour_str] = {"date_hour": date_hour_str}  # Initialisation directe

        for row in range(150, GRID_ROWS):
            for col in range(GRID_COLUMNS - 50):
                data_dict[date_hour_str][f"{row}_{col}_Volume"] = aggregated_data[hour, row, col, 0]
                data_dict[date_hour_str][f"{row}_{col}_Speed"] = aggregated_data[hour, row, col, 1]

    return pd.DataFrame.from_dict(data_dict, orient="index")

# Function to aggregate data by hour with median for volume and avg for speed
def aggregate_by_hour_V5(data: np.ndarray) -> np.ndarray:

    """
    Aggregates the data by direction only. 
    
    Args:
        data (np.ndarray): Input data with shape (288, 495, 436, 8)
    
    Returns:
        np.ndarray: Aggregated data with shape (288, 495, 436, 2) 0 for volume and 1 for speed
    """
    aggregated_data = np.zeros((288, 495, 436, 2))
    
    aggregated_data[:, :, :, 0] = np.mean(data[:, :, :, VOLUME_CHANNEL], axis=(0, -1))
        
    aggregated_data[:, :, :, 1] = np.mean(data[:, :, :, VOLUME_CHANNEL], axis=(0, -1))      
    
    return aggregated_data

# Optimized function to create a DataFrame from aggregated data
def create_dataframe_V5(aggregated_data: np.ndarray, date: datetime) -> pd.DataFrame:
    """
    Creates a DataFrame where each cell is a column in the form 'row_col_{Speed|Volume}'.
    The row index is 'date_time'.
    
    Args:
        aggregated_data (np.ndarray): Shape aggregated data (188, 495, 436, 2).
        date (datetime): Date of data.
    
    Returns:
        pd.DataFrame: Restructured DataFrame.
    """
    num_time = aggregated_data.shape[0]
    data_dict = {}

    for time in range(num_time):
        date_hour_str = f"{date.strftime('%d/%m/%Y')}_{(time//12):02d}:{((time%12)*5):02d}"
        data_dict[date_hour_str] = {"date_hour": date_hour_str}  # Initialisation directe

        for row in range(150, GRID_ROWS):
            for col in range(GRID_COLUMNS - 50):
                data_dict[date_hour_str][f"{row}_{col}_Volume"] = aggregated_data[time, row, col, 0]
                data_dict[date_hour_str][f"{row}_{col}_Speed"] = aggregated_data[time, row, col, 1]

    return pd.DataFrame.from_dict(data_dict, orient="index")


# Main script
if __name__ == "__main__":
    # Define data folder and extract files
    input_files = {}
    data_folder = Path("C:/Users/HP/Documents/Uni/Spring/Data/movie")
    input_files = extract_h5_files(data_folder)

    # Select a city and its files
    city = "BARCELONA"
    file_paths = input_files.get(city, [])
    
    if not file_paths:
        print(f"No files found for city: {city}")
        exit()

    # Group files by month
    monthly_dataframes = {}
    for file_path in file_paths:
        date = extract_date(file_path)
        if date.year != 2020:
            month_key = (date.year, date.month)
            if month_key not in monthly_dataframes:
                monthly_dataframes[month_key] = []
            monthly_dataframes[month_key].append(file_path)
    print("Month separation done...")

    # Process each month's files
    output_directory = f"C:/Users/HP/Documents/Uni/Spring/Data/matrix/{city}/"
    os.makedirs(output_directory, exist_ok=True)
    
    for (year, month), month_files in monthly_dataframes.items():

        # output_file = os.path.join(output_directory, f"{city}_{month:02d}-{year}_traffic_data.csv")
        output_file = os.path.join(output_directory, f"{city}_{month:02d}-{year}_traffic_data_without_aggregation.parquet")
        # Accumulate DataFrames in a list
        dfs = []
        i = 0
        for file_path in month_files:
            print(f"Processing file: {file_path}")
            data = load_h5_file(file_path)
            date = extract_date(file_path)
            aggregated_data = aggregate_by_hour_V4(data)
            df = create_dataframe_V4(aggregated_data, date)
            dfs.append(df)
            del data, aggregated_data 


        if dfs:  
            print("Concatenation of dfs")
            final_df = pd.concat(dfs, ignore_index=True)
            df_cleaned = final_df.loc[:, (final_df != 0).any(axis=0)]
            print("To csv...")
            # df_cleaned.to_csv(output_file, index=False)
            df_cleaned.to_parquet(output_file, index=False, compression="snappy")
            print(f"Saved aggregated data to {output_file}")
            print(f"Shape cleaned dataframe: {df_cleaned.shape}")
            print(f"Shape dataframe: {final_df.shape}")
