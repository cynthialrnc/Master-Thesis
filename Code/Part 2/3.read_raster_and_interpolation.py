import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio

# Load the raster and get its affine transformation
with rasterio.open("C:/Users/HP/Documents/Uni/Spring/Data/Map/BARCELONA/output_raster_25831_georef.tif") as src:
    transform = src.transform 

# Dimensions after georeferencing and before (original)
rows, cols = 547, 368
original_rows, original_cols = 495, 436

# Function to retrieve original row/column indices based on coordinates
def get_original_row_col(x, y, scale_factor_row, scale_factor_col, transform):
    col, row = ~transform * (x, y)  # Inverse transform to get raster indices
    orig_row = int(row * scale_factor_row)
    orig_col = int(col * scale_factor_col)
    return orig_row, orig_col

# Load shapefile with points
shapefile_path = "C:/Users/HP/Documents/Uni/Spring/Data/map/BARCELONA/join/join_point_full.shp"
gdf = gpd.read_file(shapefile_path)

print(gdf.head())
print("Shapefile dimensions:", gdf.shape)  # (number of rows, number of columns)
print("Coordinate Reference System (CRS):", gdf.crs)
print("Shapefile columns:", gdf.columns)

# Compute scale factors to map resized image to original dimensions
scale_factor_row = original_rows / rows
scale_factor_col = original_cols / cols

# Apply reverse mapping to get original grid indices
gdf[['row', 'col']] = gdf.apply(
    lambda p: pd.Series(get_original_row_col(p.geometry.x, p.geometry.y, scale_factor_row, scale_factor_col, transform)),
    axis=1
)
print(gdf[['geometry', 'row', 'col']].head())

# Create an empty matrix with NaN
grid_original = np.full((original_rows, original_cols), np.nan)

# Fill the matrix with codi_emo values using the mapped indices
for _, row in gdf.iterrows():
    r, c = int(row['row']), int(row['col'])  # Get row/column indices
    if 0 <= r < original_rows and 0 <= c < original_cols:  # Check bounds
        grid_original[r, c] = row['codi_emo']

# Check matrix shape
print("Reconstructed 2D grid shape:", grid_original.shape)

# Convert to DataFrame for export
df_original = pd.DataFrame(grid_original)
print("Example cell value [100, 0]:", df_original.iloc[100, 0])

grid_original = df_original.values

# Create a copy for interpolation
grid_interpolated = grid_original.copy()

# Check number of NaN before interpolation
print(f"Number of NaN values before interpolation: {np.isnan(grid_original).sum()}")

# Function to fill NaN using left and right neighbors
def fill_nans_with_left_neighbor(grid):
    filled_any = True
    
    while filled_any:
        filled_any = False
        
        for r in range(original_rows):
            for c in range(1, original_cols - 1):  # Avoid border columns
                if np.isnan(grid[r, c]):  # If the cell is NaN
                    left_neighbor = grid[r, c - 1]
                    right_neighbor = grid[r, c + 1]
                    
                    if not np.isnan(left_neighbor) and not np.isnan(right_neighbor):
                        grid[r, c] = left_neighbor  # Fill using left neighbor
                        filled_any = True

# Apply the filling function
fill_nans_with_left_neighbor(grid_interpolated)

# Check number of NaN after interpolation
print(f"Number of NaN values after interpolation: {np.isnan(grid_interpolated).sum()}")

# Convert to DataFrame and save
df_original_interpolated = pd.DataFrame(grid_interpolated)
print("Interpolated DataFrame shape:", df_original_interpolated.shape)
print("Example interpolated value [2, 0]:", df_original_interpolated.iloc[2, 0])

df_original_interpolated.to_csv("C:/Users/HP/Documents/Uni/Spring/Data/Map/BARCELONA/codi_emo_grid_interpolated.csv", index=False, header=False)

# Visualization before and after interpolation
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(grid_original, cmap='viridis')
plt.title("Grid before interpolation")
plt.colorbar()
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(grid_interpolated, cmap='viridis')
plt.title("Grid after local interpolation")
plt.colorbar()
plt.axis('off')

plt.tight_layout()
plt.show()
