# Traffic Data Analysis Pipeline - README

## Folder Structure Overview

```
C:.
├───Part 1
│ 1.preprocessing_here_data_pca.py
│ 2.Transform.R
│ 3.Merge.R
│ 4.PCA.R
│
├───Part 2
│ .Rhistory
│ 1.create_raster_for_qgis.py
│ 2.Georeferenced_qgis.qgz
│ 3.read_raster_and_interpolation.py
│ 4.Map.R
│
└───Part 3
1.preprocessing_analyzing.py
2.Create_Here_dataset_with_clusters.R
3.Analyzing_traffic_data.R
4.GWR.R
```

## Project Description and Workflow

This project implements a comprehensive analytical pipeline to process and analyze large-scale traffic data across multiple cities. The workflow is modular and structured into three main parts, each targeting a specific phase of data preparation, analysis, and spatial integration.

### Part 1: Data Preprocessing and Dimensionality Reduction

- Scripts in this section handle the initial preprocessing of HERE traffic data.
- They perform temporal aggregation and transformation to prepare the data for analysis.
- A key step here is the application of Principal Component Analysis (PCA) for dimensionality reduction, enabling extraction of the main traffic patterns.
- The PCA-based clustering results form the basis for subsequent analysis.
- This module is consistently applied across all three cities studied: Antwerp, Berlin, and Barcelona, ensuring comparability of traffic dynamics analysis.

### Part 2: Spatial Data Preparation and Integration (Barcelona-Specific)

- This part focuses on creating spatial raster layers compatible with QGIS, georeferencing them properly, and performing spatial interpolation.
- It includes merging traffic data with spatial layers using the `codi_emo` administrative zoning system available only for Barcelona.
- Due to richer spatial data availability for Barcelona, this module enables spatial enrichment of traffic data, linking traffic patterns with land use and administrative boundaries.
- The QGIS project file and raster processing scripts support visualization and geospatial operations.

### Part 3: Advanced Analysis and Spatial Modeling (Barcelona-Specific)

- Here the enriched dataset is further analyzed through clustering, detailed profiling, and spatial statistical modeling.
- Scripts include creating clustered datasets, in-depth traffic data analysis, and geographically weighted regression (GWR) to explore spatial heterogeneity.
- These analyses incorporate socioeconomic and land use context unique to Barcelona, providing deeper insights into spatial mobility patterns.
- This spatial enrichment and modeling step was not feasible for Antwerp and Berlin due to the lack of equivalent detailed spatial data.

