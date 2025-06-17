
# Advanced Data Science Techniques for Spatio-Temporal Analysis of Urban Traffic Patterns
## Project Overview

This project investigates urban traffic patterns using floating vehicle data sourced from HERE Technologies. Leveraging advanced data science methods such as Principal Component Analysis (PCA) and clustering, it aims to uncover meaningful patterns in traffic speed and volume across major cities. The study integrates spatial and temporal dimensions to enhance traditional traffic analysis approaches, ultimately contributing to better urban mobility management.

## Context and Motivation

Urban mobility faces increasing pressure due to population growth and rising vehicle ownership. Effective traffic management is crucial not only to alleviate congestion and reduce travel times but also to minimize environmental impacts and improve overall urban quality of life.

Conventional traffic data sources, such as fixed sensors and cameras, often provide limited spatial coverage and fail to capture complex city-wide mobility patterns. The rise of GPS-equipped vehicles has enabled the collection of floating car data (FCD), offering high-resolution, grid-based traffic information across cities. This data empowers more comprehensive analysis of traffic dynamics in space and time.

Alongside, advances in Big Data analytics, machine learning, and IoT technologies enable the processing of vast and heterogeneous datasets. This project capitalizes on these advances to explore spatio-temporal traffic patterns more deeply than traditional descriptive analyses allow.

## About this repository

This is a github repo to share code for the Master Thesis paper.

## Data

The core dataset comprises floating vehicle data collected at five-minute intervals from ten major global cities between 2019 and 2021. Unlike typical road segment-based data, this dataset is structured on a regular geographic grid dividing the urban environment into cells. Each cell records vehicle speed and volume in four diagonal directions (NW, NE, SW, SE), facilitating high-resolution spatial and temporal analyses independent of road network constraints.

For the Barcelona metropolitan area, additional spatial datasets—including administrative boundaries, socioeconomic indicators, land use variables, and points of interest—are incorporated to enrich the analysis and explore correlations with traffic patterns.

## Get the Data

You can obtain the traffic data used in this project from [HERE Technologies](https://developer.here.com/sample-data), available for academic and non-commercial purposes.

**Only download the data for the cities of Berlin, Barcelona, and Antwerp for the year 2021.**

Once downloaded, place the files in the `movie/` directory at the root of this repository.

**Warning:** These files are large and may exceed several tens of gigabytes. Ensure you have sufficient disk space and a stable internet connection.

**Note:** The data for **Barcelona metropolitan area** (shapfile and land use variables dataset) is already included in this repository under the folder named `02-BCN DATA/`.

## Data pipeline

The following diagram gives an overview:
<img src="./Data Pipeline.png">

## Setup

This project integrates **Python**, **R**, and **QGIS** for data processing, analysis, and visualization. Below are setup instructions and dependencies by environment.

### Python

Python is used for raw data preprocessing, especially handling HDF5 traffic data.

#### Key Libraries:

* `h5py`: Efficient access to large `.h5` traffic datasets.
* `numpy`, `pandas`: Data wrangling and aggregation (e.g., hourly average speed and volume per direction).
* `pyarrow`: Export processed data as `.parquet` for compatibility with R.
* `os`, `pathlib`, `re`: File system navigation and filename parsing.
* `rasterio`, `pyproj`: Spatial alignment of traffic data with shapefiles (affine transforms, CRS conversion).

#### Setup:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Your `requirements.txt` should include:

```text
h5py
numpy
pandas
pyarrow
rasterio
pyproj
```

### R

R is used for statistical analysis, clustering, visualization, and spatial mapping.

#### Key Packages:

**Data Manipulation**

* `dplyr`, `tidyr`, `tibble`, `data.table`

**Visualization**

* `ggplot2`, `viridis`, `tmap`, `fmsb`

**Statistical Analysis & Clustering**

* `stats`, `FSA`, `FactoMineR`, `factoextra`, `NbClust`, `cluster`, `matrixStats`, `MASS`, `car`, `chemometrics`

**Spatial Data**

* `sf`, `sp`, `spgwr`, `tmap`

**Utilities**

* `arrow`, `lubridate`, `stringr`, `readxl`, `Hmisc`, `scales`, `forcats`

#### Setup:

Install packages via R or RStudio:

```r
install.packages(c(
  "dplyr", "tidyr", "tibble", "data.table",
  "ggplot2", "viridis", "tmap", "fmsb",
  "stats", "FSA", "FactoMineR", "factoextra", "NbClust", "cluster", 
  "matrixStats", "MASS", "car", "chemometrics",
  "sf", "sp", "spgwr",
  "arrow", "lubridate", "stringr", "readxl", "Hmisc", "scales", "forcats"
))
```

### QGIS

QGIS is used for advanced geospatial analysis and visualization.

#### Requirements:

* [QGIS 3.28+](https://qgis.org/en/site/forusers/download.html)
* Ensure the **Processing** plugin is enabled.
* Compatible with GeoPackage, Shapefile, and GeoJSON layers used in this project.


