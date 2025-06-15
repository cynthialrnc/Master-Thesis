
# Advanced Data Science Techniques for Spatio-Temporal Analysis of Urban Traffic Patterns
## Project Overview

This project investigates urban traffic patterns using floating vehicle data sourced from HERE Technologies. Leveraging advanced data science methods such as Principal Component Analysis (PCA) and clustering, it aims to uncover meaningful patterns in traffic speed and volume across major cities. The study integrates spatial and temporal dimensions to enhance traditional traffic analysis approaches, ultimately contributing to better urban mobility management.

## Context and Motivation

Urban mobility faces increasing pressure due to population growth and rising vehicle ownership. Effective traffic management is crucial not only to alleviate congestion and reduce travel times but also to minimize environmental impacts and improve overall urban quality of life.

Conventional traffic data sources, such as fixed sensors and cameras, often provide limited spatial coverage and fail to capture complex city-wide mobility patterns. The rise of GPS-equipped vehicles has enabled the collection of floating car data (FCD), offering high-resolution, grid-based traffic information across cities. This data empowers more comprehensive analysis of traffic dynamics in space and time.

Alongside, advances in Big Data analytics, machine learning, and IoT technologies enable the processing of vast and heterogeneous datasets. This project capitalizes on these advances to explore spatio-temporal traffic patterns more deeply than traditional descriptive analyses allow.

## Data Description

The core dataset comprises floating vehicle data collected at five-minute intervals from ten major global cities between 2019 and 2021. Unlike typical road segment-based data, this dataset is structured on a regular geographic grid dividing the urban environment into cells. Each cell records vehicle speed and volume in four diagonal directions (NW, NE, SW, SE), facilitating high-resolution spatial and temporal analyses independent of road network constraints.

For the Barcelona metropolitan area, additional spatial datasets—including administrative boundaries, socioeconomic indicators, land use variables, and points of interest—are incorporated to enrich the analysis and explore correlations with traffic patterns.

## Research Objectives

This project addresses several key objectives:

- To explore and preprocess large-scale floating vehicle data from HERE Technologies.

- To apply multivariate statistical methods (PCA, clustering, cluster profiling) to identify dominant traffic patterns in speed and volume.

- To integrate spatial indicators for enriched analysis within the Barcelona metropolitan area.

- To analyze temporal variability and detect congestion trends across multiple time scales (hourly, daily, seasonal).

- To assess the utility of HERE data for advancing transport modeling and understanding urban travel behavior.

## Research Questions

Key questions guiding the study include:

- What are the primary temporal and spatial traffic patterns observable in major urban areas using HERE floating vehicle data?

- How effectively can clustering techniques uncover recurring congestion patterns or anomalous traffic behaviors?

- How can this grid-based traffic data be transformed to support and complement existing traffic engineering practices?

## Methodology

The analysis combines Python and R programming environments, employing libraries for data preprocessing, statistical modeling, and spatial visualization. The methodology integrates time series analysis, dimensionality reduction, clustering, and spatial data enrichment to deliver a comprehensive view of urban traffic dynamics.
