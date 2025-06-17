import numpy as np
import h5py
from pathlib import Path
from typing import Union
import rasterio
from rasterio.transform import Affine
from pyproj import Transformer

# Constants
SPEED_CHANNEL = [1, 3, 5, 7]     # Indices of speed channels in the data
VOLUME_CHANNEL = [0, 2, 4, 6]    # Indices of volume channels in the data
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

# Define input file paths
data_folder = Path("C:/Users/HP/Documents/Uni/Spring/Data/movie/BARCELONA/")
input_file_map_high_res = data_folder / "BARCELONA_map_high_res.h5"
input_file_static = data_folder / "BARCELONA_static.h5"

# Function to convert DMS (degrees, minutes, seconds) to decimal degrees
def dms_to_decimal(degrees, minutes, seconds):
    return degrees + minutes / 60 + seconds / 3600

# Load the static data
data = load_h5_file(input_file_static)  # Expected shape: (1, 495, 436)

# Define the subset of the grid to be extracted
row_start, row_end = 0, GRID_ROWS
col_start, col_end = 0, GRID_COLUMNS

# Extract the desired portion of the grid (removes singleton dimension)
selected_data = data[0, row_start:row_end, col_start:col_end]  # Shape: (495, 436)

# Define top-left corner coordinates in DMS (manually estimated or measured)
# Convert to decimal degrees
top_left_lat = dms_to_decimal(41, 74, 37.67)  # Latitude
top_left_lon = dms_to_decimal(1, 91, 93.35)   # Longitude

# Transform coordinates from WGS84 (EPSG:4326) to projected system EPSG:25831
transformer = Transformer.from_crs("EPSG:4326", "EPSG:25831", always_xy=True)
top_left_x, top_left_y = transformer.transform(top_left_lon, top_left_lat)

# Define the affine transformation (pixel size: 100x100m, origin at top-left)
transform = Affine(
    100, 0, top_left_x,   # pixel width, rotation, X origin
    0, -100, top_left_y   # rotation, negative pixel height, Y origin
)

# Define output file path
output_tif = "C:/Users/HP/Documents/Uni/Spring/Data/Map/BARCELONA/output_raster_25831.tif"

# Save the extracted grid as a GeoTIFF with georeferencing
with rasterio.open(
    output_tif,
    "w",
    driver="GTiff",
    height=selected_data.shape[0],
    width=selected_data.shape[1],
    count=1,                      # Single band raster
    dtype=selected_data.dtype,   # Use original data type
    crs="EPSG:25831",            # Coordinate reference system
    transform=transform          # Affine transformation
) as dst:
    dst.write(selected_data, 1)  # Write data to the first band

print("GeoTIFF saved:", output_tif)
