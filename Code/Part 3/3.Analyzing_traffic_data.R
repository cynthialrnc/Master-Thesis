# Load required libraries
library(data.table)
library(tmap)
library(sf)
library(sp)
library(dplyr)
library(tidyr)
library(ggplot2)
library(viridisLite)
library(lubridate)
library(FactoMineR)
library(factoextra)
library(arrow)
library(readxl)
library(xtable)
library(stringr)
library(Hmisc)
library(spgwr)
library(chemometrics)
library(matrixStats)
library(MASS)
library(car) 
library(FSA) 
library(scales)
library(fmsb)
library(NbClust)
library(cluster)

# Load HERE traffic dataset
file_path <- "C:/Users/HP/Documents/Uni/Spring/Data/matrix/BARCELONA/BARCELONA_traffic_dataframe_preparation.parquet"
df_here <- read_parquet(file_path)

# Define holidays in Barcelona
holidays_BARCELONA <- c("2019-04-19", "2019-04-22", "2019-05-01", "2019-06-10", "2019-06-24")

# Extract temporal dimensions
Sys.setlocale("LC_TIME", "English")
df_here_date <- df_here %>%
  distinct(date) %>%
  mutate(
    weekday = lubridate::wday(date, label = TRUE), # Weekday label (Mon, Tue, ...)
    month = lubridate::month(date, label = TRUE),  # Month label
    holiday = date %in% ymd(holidays_BARCELONA)    # Flag holidays
  )

# Normalize hour format (HH:00)
df_here <- df_here %>%
  mutate(hour_full = str_sub(hour, 1, 2) %>% paste0(":00"))

# Load land use data (codi_emo)
codi_emo_excel <- read_excel("C:/Users/HP/Documents/Uni/Spring/02-BCN DATA/Cadastre20_usos_RMB_EMO.XLSX")

# Prepare land use percentages per region
codi_emo_excel$codi_emo <- as.factor(codi_emo_excel$codi_emo)
codi_emo_excel <- codi_emo_excel %>% dplyr::select(-codi_emo_t)
codi_emo_excel <- codi_emo_excel %>%
  rowwise() %>%  
  mutate(across(where(is.numeric), ~ . / sum(c_across(where(is.numeric))) * 100)) %>%  
  ungroup()

# Load shapefile with EMO zones
shp <- st_read("C:/Users/HP/Documents/Uni/Spring/02-BCN DATA/ZonesEMO_MacroZones/ZonesEMO_MacroZones.shp")

# Define variables of interest
vars <- c("speed_NE", "speed_NW", "speed_SE", "speed_SW", "volume_NE", "volume_NW", "volume_SE", "volume_SW")
speed_vars <- c("speed_NE", "speed_NW", "speed_SE", "speed_SW")
volume_vars <- c("volume_NE", "volume_NW", "volume_SE", "volume_SW")

# Convert to data.table for efficient aggregation
setDT(df_here)

# Compute volume-weighted speeds and total volumes per direction
traffic_agg <- df_here[, .(
  speed_NE = if (sum(volume_NE, na.rm = TRUE) == 0) NA else sum(speed_NE * volume_NE, na.rm = TRUE) / sum(volume_NE, na.rm = TRUE),
  volume_NE = sum(volume_NE, na.rm = TRUE),
  speed_NW = if (sum(volume_NW, na.rm = TRUE) == 0) NA else sum(speed_NW * volume_NW, na.rm = TRUE) / sum(volume_NW, na.rm = TRUE),
  volume_NW = sum(volume_NW, na.rm = TRUE),
  speed_SE = if (sum(volume_SE, na.rm = TRUE) == 0) NA else sum(speed_SE * volume_SE, na.rm = TRUE) / sum(volume_SE, na.rm = TRUE),
  volume_SE = sum(volume_SE, na.rm = TRUE),
  speed_SW = if (sum(volume_SW, na.rm = TRUE) == 0) NA else sum(speed_SW * volume_SW, na.rm = TRUE) / sum(volume_SW, na.rm = TRUE),
  volume_SW = sum(volume_SW, na.rm = TRUE)
), by = .(codi_emo)]

# Join shapefile with traffic aggregates
shp_emo_data <- merge(shp, traffic_agg, by = "codi_emo")

# Generate decile categories for visualization
shp_emo_data_decile <- shp_emo_data %>%
  mutate(across(
    all_of(vars),
    ~ ntile(., 10),
    .names = "{.col}_decile"
  )) %>%
  mutate(across(
    contains("_decile"),
    ~ factor(., labels = paste("Decile ", 1:10))
  ))

# Map deciles by variable
for (v in vars) {
  speed_var <- paste0(v, "_decile")
  print(tm_shape(shp_emo_data_decile) +
          tm_polygons(
            fill = speed_var,
            palette = viridis(10),
            title = paste(v, "(Decile)")
          ) +
          tm_layout(title = paste(v, "Decile Map"),
                    legend.outside = TRUE))
}

# --- Temporal visualization on a specific date and hours ---

target_date <- as.Date("2019-05-01")
target_hour <- c("06:00", "07:00", "08:00", "09:00")

df_filtered <- df_here[date == target_date & hour_full  %in% target_hour]

# Aggregate by region and hour
data_hourly <- df_here[, .(
  speed_NE = if (sum(volume_NE, na.rm = TRUE) == 0) 0 else sum(speed_NE * volume_NE, na.rm = TRUE) / sum(volume_NE, na.rm = TRUE),
  volume_NE = sum(volume_NE, na.rm = TRUE)
), by = .(codi_emo, hour_full)]

data_hourly_filtered <- data_hourly[hour_full %in% target_hour]

# Merge with shapefile and classify into deciles
shp_hourly <- merge(shp, data_hourly_filtered, by = "codi_emo")
shp_hourly$hour_full <- as.factor(shp_hourly$hour_full)
shp_hourly$speed_NE_decile <- cut2(shp_hourly$speed_NE, g = 10)
shp_hourly$speed_NE_decile <- as.numeric(factor(shp_hourly$speed_NE_decile))
shp_hourly$speed_NE_decile <- factor(shp_hourly$speed_NE_decile, levels = 1:10, labels = paste0("Decile ", 1:10))

# Faceted map by hour
tmap_mode("view")
tm_shape(shp_hourly) +
  tm_polygons(
    fill = "speed_NE_decile",
    fill.scale = tm_scale(values = "viridis"),
    fill.legend = tm_legend(title = "Speed NE")
  ) +
  tm_facets(by = "hour_full", free.coords = FALSE, ncol = 1) +
  tm_layout(legend.outside = TRUE)


# --- Clustering visualization ---

# Load hourly cluster results
data_cluster <- read.csv("C:/Users/HP/Documents/Uni/Spring/Data/matrix/BARCELONA/data_cluster.csv")
data_cluster$date <- as.Date(data_cluster$date)

# Merge cluster information into main dataset
df_here <- df_here %>%
  left_join(data_cluster, by = c("date" = "date", "hour_full" = "hour")) %>%
  mutate(cluster = as.factor(cluster))

# Combine with weekday and holiday metadata
df_here_analyze <- df_here %>%
  left_join(df_here_date, by = c("date" = "date"))

# Count records per hour and cluster
df_here_analyze %>%
  group_by(hour_full, cluster) %>%
  summarise(n = n()) %>%
  pivot_wider(names_from = cluster, values_from = n, values_fill = 0)

# Cluster frequency heatmap by weekday and hour
df_here_analyze %>%
  count(weekday, hour_full, cluster) %>%
  mutate(
    hour_full = sub(":.*", "", hour_full),
    frequency = n
  ) %>%
  ggplot(aes(x = hour_full, y = weekday, fill = frequency)) +
  geom_tile() +
  facet_wrap(~cluster) +
  scale_fill_viridis_c() + 
  labs(title = "Heatmap of cluster frequency by hour and weekday",
       x = "Hour", y = "Weekday", fill = "Frequency") +
  theme_minimal()

# Volume-weighted average speed per cluster
df_here_analyze %>%
  mutate(
    hour = as.integer(sub(":.*", "", hour_full)),
    weighted_NE = speed_NE * volume_NE,
    weighted_NW = speed_NW * volume_NW,
    weighted_SE = speed_SE * volume_SE,
    weighted_SW = speed_SW * volume_SW,
    total_weighted_speed = rowSums(across(starts_with("weighted_")), na.rm = TRUE),
    total_volume = rowSums(across(starts_with("volume_")), na.rm = TRUE)
  ) %>%
  group_by(weekday, hour, cluster) %>%
  summarise(
    weighted_speed = sum(total_weighted_speed, na.rm = TRUE) / sum(total_volume, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  ggplot(aes(x = hour, y = weekday, fill = weighted_speed)) +
  geom_tile() +
  facet_wrap(~cluster) +
  scale_fill_viridis_c(option = "D") +
  labs(
    title = "Heatmap of Volume-Weighted Speed by Hour, Weekday and Cluster",
    x = "Hour", y = "Weekday", fill = "Weighted Speed"
  ) +
  theme_minimal()

# Total volume per cluster
df_here_analyze %>%
  mutate(
    hour = as.integer(sub(":.*", "", hour_full)),
    total_volume_line = rowSums(across(all_of(volume_vars)), na.rm = TRUE)
  ) %>%
  group_by(weekday, hour, cluster) %>%
  summarise(total_volume = sum(total_volume_line, na.rm = TRUE), .groups = "drop") %>%
  ggplot(aes(x = hour, y = weekday, fill = total_volume)) +
  geom_tile() +
  facet_wrap(~cluster) +
  scale_fill_viridis_c(option = "D") +
  labs(
    title = "Heatmap of Total Volume by Hour, Weekday and Cluster",
    x = "Hour", y = "Weekday", fill = "Total Volume"
  ) +
  theme_minimal()


# Identify the top 6 clusters for each codi_emo by frequency
top_clusters <- df_here %>%
  group_by(codi_emo, cluster) %>%
  count() %>%
  arrange(codi_emo, desc(n)) %>%
  group_by(codi_emo) %>%
  slice(1:6)  # Get top 6 clusters for each codi_emo

# Create a new column to indicate the rank of the cluster
top_clusters <- top_clusters %>%
  mutate(cluster_rank = dense_rank(desc(n)))  # Rank the clusters by frequency

# Merge the top clusters with the shapefile data
shp_emo_data_with_top_clusters <- merge(shp_emo_data_decile, top_clusters, by = "codi_emo")

# Loop through the top 6 clusters and create a map for each
for (i in 1:6) {
  map_data <- shp_emo_data_with_top_clusters %>%
    filter(cluster_rank == i)  # Filter for the i-th most frequent cluster
  
  # Create and display the map for the current rank
  map_plot <- tm_shape(map_data) +
    tm_polygons(
      col = "cluster",  # Color by the most frequent cluster for this rank
      palette = "viridis",  # Color scale for clusters
      title = paste("Top", i, "Most Frequent Cluster")  # Title for the map
    ) +
    tm_layout(
      legend.outside = TRUE, 
      legend.title.size = 1.2,  # Increase legend title size if needed
      legend.text.size = 0.8    # Adjust legend text size if needed
    )
  
  # Print the map
  print(map_plot)  # Using print() explicitly to display the map
}
rm(map_plot, map_data, i)




## PCA (Principal Component Analysis)

# Aggregate data by codi_emo, date, and hour with weighted speed calculations
data_agg_pca <- df_here %>%
  group_by(codi_emo, date, hour) %>%
  summarise(
    speed_NE = if (sum(volume_NE, na.rm = TRUE) == 0) 0 else sum(speed_NE * volume_NE, na.rm = TRUE) / sum(volume_NE, na.rm = TRUE),
    volume_NE = sum(volume_NE, na.rm = TRUE),
    speed_NW = if (sum(volume_NW, na.rm = TRUE) == 0) 0 else sum(speed_NW * volume_NW, na.rm = TRUE) / sum(volume_NW, na.rm = TRUE),
    volume_NW = sum(volume_NW, na.rm = TRUE),
    speed_SE = if (sum(volume_SE, na.rm = TRUE) == 0) 0 else sum(speed_SE * volume_SE, na.rm = TRUE) / sum(volume_SE, na.rm = TRUE),
    volume_SE = sum(volume_SE, na.rm = TRUE),
    speed_SW = if (sum(volume_SW, na.rm = TRUE) == 0) 0 else sum(speed_SW * volume_SW, na.rm = TRUE) / sum(volume_SW, na.rm = TRUE),
    volume_SW = sum(volume_SW, na.rm = TRUE)
  )

# Convert codi_emo to factor in both datasets
shp$codi_emo <- as.factor(shp$codi_emo)
data_agg_pca$codi_emo <- as.factor(data_agg_pca$codi_emo)
data_agg_pca$date <- as.Date(data_agg_pca$date)

# Join additional data: codi_emo_excel, shp, and date features
data_agg_pca <- data_agg_pca %>%
  left_join(codi_emo_excel, by = c("codi_emo" = "codi_emo")) %>%
  left_join(shp[c(5,7)], by = c("codi_emo" = "codi_emo")) %>%
  left_join(df_here_date, by = c("date" = "date"))

# Date and time feature engineering
data_agg_pca$date <- as.factor(data_agg_pca$date)
data_agg_pca$holiday <- (as.numeric(data_agg_pca$holiday)) - 1
data_agg_pca$month <- lubridate::month(data_agg_pca$date, label = FALSE)
data_agg_pca$weekday <- lubridate::wday(data_agg_pca$date, label = FALSE)
data_agg_pca$hour_int <- as.integer(substr(as.character(data_agg_pca$hour), 1, 2))

# Keep only numeric columns for PCA
data_agg_pca_num <- data_agg_pca %>%
  ungroup() %>%
  select(where(~ is.numeric(.)))

# Perform PCA and summarize results
result_pca <- PCA(data_agg_pca_num, graph = FALSE)
summary(result_pca)
fviz_eig(result_pca, addlabels = TRUE)

# Get variable contributions and quality of representation
result_pca$var$contrib
result_pca$var$cos2


## PCA and Clustering based on codi_emo

# Prepare traffic_agg dataset for clustering
traffic_agg$codi_emo <- as.factor(traffic_agg$codi_emo)
traffic_agg_pca <- traffic_agg %>%
  left_join(codi_emo_excel, by = c("codi_emo" = "codi_emo")) %>%
  left_join(shp[c(5,7)], by = c("codi_emo" = "codi_emo")) 

# Keep only numeric variables for PCA
traffic_agg_pca_num <- traffic_agg_pca %>%
  ungroup() %>%
  select(where(~ is.numeric(.)))

# Perform PCA with supplementary quantitative variables
result_pca_codi_emo <- PCA(traffic_agg_pca_num, graph = FALSE, quanti.sup = 9:26)

# Hierarchical clustering on principal components
result_hcpc_codi_emo <- HCPC(result_pca_codi_emo, nb.clust = 4, graph = FALSE)

# Cluster statistics and visualization
table(result_hcpc_codi_emo$data.clust$clust)
result_hcpc_codi_emo$desc.var
result_hcpc_codi_emo$desc.ind
fviz_cluster(result_hcpc_codi_emo)

# Add cluster assignment back to original data
traffic_agg$cluster <- result_hcpc_codi_emo$data.clust$clust

# Merge shapefile with cluster data
shp_emo_data_cluster <- merge(shp, traffic_agg, by = "codi_emo")

# Map clusters spatially
tmap_mode("view")
tm_shape(shp_emo_data_cluster) +
  tm_polygons("cluster", palette = "viridis", title = "Cluster", style = "cat", showNA = FALSE) +
  tm_layout(title = "Traffic zone clusters (emo_code)", legend.outside = TRUE)

# Run PCA again to get scores
pca_vars <- traffic_agg_pca_num
pca_scores <- PCA(pca_vars, graph = FALSE)$ind$coord
pca_clean <- as.data.frame(pca_scores) %>% select(where(is.numeric)) %>% na.omit()
pca_clean <- pca_clean[, apply(pca_clean, 2, sd) > 0]

# Mahalanobis distance for outlier detection
center <- colMeans(pca_clean)
cov_matrix <- cov(pca_clean)
mahal_dist <- mahalanobis(pca_clean, center, cov_matrix)
cutoff <- qchisq(0.999, df = ncol(pca_clean))
outlier_indices <- which(mahal_dist > cutoff)

# Remove outliers and perform new PCA
length(outlier_indices)
pca_clean_no_outliers <- pca_clean[-outlier_indices, ]
res_pca_no_outliers <- PCA(pca_clean_no_outliers, graph = FALSE)

# New hierarchical clustering without outliers
res_hcpc_no_outliers <- HCPC(res_pca_no_outliers, nb.clust = 4, graph = FALSE)


############################################################################

fviz_nbclust(result_pca_codi_emo$ind$coord[, 1:5], FUNcluster=kmeans, method="silhouette")
NbClust(data = result_pca_codi_emo$ind$coord[, 1:5], distance = "euclidean", min.nc = 2, max.nc = 10, method = "ward.D")

############################################################################


# Visualize new clustering
fviz_cluster(res_hcpc_no_outliers)

# Assign new cluster labels including NA for outliers
traffic_agg$cluster_outlier <- NA 
traffic_agg$cluster_outlier[-outlier_indices] <- res_hcpc_no_outliers$data.clust$clust

# Merge shapefile and visualize final clusters
shp_emo_data_cluster <- merge(shp, traffic_agg, by = "codi_emo")
tmap_mode("view")
tm_shape(shp_emo_data_cluster) +
  tm_polygons("cluster_outlier", palette = "viridis", title = "Cluster", style = "cat", showNA = FALSE) +
  tm_layout(title = "Traffic zone clusters without outliers (emo_code)", legend.outside = TRUE)

## Profiling Clusters

# Summary statistics per cluster
table(shp_emo_data_cluster$cluster)

# Aggregate statistics for each direction (NE, NW, SE, SW)
shp_emo_data_cluster %>%
  st_drop_geometry() %>%
  group_by(cluster) %>%
  summarise(across(c(speed_NE, speed_NW, speed_SE, speed_SW,
                     volume_NE, volume_NW, volume_SE, volume_SW),
                   list(mean = ~mean(., na.rm = TRUE),
                        sd = ~sd(., na.rm = TRUE))))

# Boxplot of NE Speed by Cluster
ggplot(shp_emo_data_cluster %>% st_drop_geometry(), aes(x = as.factor(cluster), y = speed_NE)) +
  geom_boxplot() +
  labs(title = "Distribution of NE Speed by Cluster", x = "Cluster", y = "Speed NE")

# Aggregation at codi_emo level
traffic_agg_codi <- traffic_agg %>%
  group_by(codi_emo, cluster) %>%
  summarise(
    speed_mean = mean(c_across(all_of(speed_vars)), na.rm = TRUE),
    volume_mean = mean(c_across(all_of(volume_vars)), na.rm = TRUE),
    speed_median = median(c_across(all_of(speed_vars)), na.rm = TRUE),
    volume_median = median(c_across(all_of(volume_vars)), na.rm = TRUE),
    speed_iqr = IQR(unlist(across(all_of(speed_vars))), na.rm = TRUE),
    volume_iqr = IQR(unlist(across(all_of(volume_vars))), na.rm = TRUE),
    speed_sum = sum(c_across(all_of(speed_vars)), na.rm = TRUE),
    volume_sum = sum(c_across(all_of(volume_vars)), na.rm = TRUE),
    speed_weighted = sum(c_across(all_of(speed_vars)) * c_across(all_of(volume_vars)), na.rm = TRUE) / 
      sum(c_across(all_of(volume_vars)), na.rm = TRUE),
    .groups = "drop"
  )


# Aggregation at cluster level
traffic_agg_cluster <- traffic_agg %>%
  group_by(cluster) %>%
  summarise(
    speed_mean = mean(rowMeans(dplyr::select(cur_data(), all_of(speed_vars)) %>% as.matrix(), na.rm = TRUE), na.rm = TRUE),
    volume_mean = mean(rowMeans(dplyr::select(cur_data(), all_of(volume_vars)) %>% as.matrix(), na.rm = TRUE), na.rm = TRUE),
    speed_median = median(rowMedians(dplyr::select(cur_data(), all_of(speed_vars)) %>% as.matrix(), na.rm = TRUE), na.rm = TRUE),
    volume_median = median(rowMedians(dplyr::select(cur_data(), all_of(volume_vars)) %>% as.matrix(), na.rm = TRUE), na.rm = TRUE),
    speed_iqr = median(rowIQRs(dplyr::select(cur_data(), all_of(speed_vars)) %>% as.matrix(), na.rm = TRUE), na.rm = TRUE),
    volume_iqr = median(rowIQRs(dplyr::select(cur_data(), all_of(volume_vars)) %>% as.matrix(), na.rm = TRUE), na.rm = TRUE),
    speed_sum = sum(rowSums(dplyr::select(cur_data(), all_of(speed_vars)) %>% as.matrix(), na.rm = TRUE), na.rm = TRUE),
    volume_sum = sum(rowSums(dplyr::select(cur_data(), all_of(volume_vars)) %>% as.matrix(), na.rm = TRUE), na.rm = TRUE),
    speed_weighted = sum(rowSums(dplyr::select(cur_data(), all_of(speed_vars)) * dplyr::select(cur_data(), all_of(volume_vars)), na.rm = TRUE), na.rm = TRUE) / 
      sum(rowSums(dplyr::select(cur_data(), all_of(volume_vars)), na.rm = TRUE), na.rm = TRUE),
    .groups = "drop"
  )

shp_traffic_agg_cluster <- merge(shp, traffic_agg_codi, by = "codi_emo")

# Mapping weighted speed and volume by cluster
tm_shape(shp_traffic_agg_cluster) +
  tm_fill("speed_weighted", palette = "viridis", style = "quantile", title = "Weighted Speed") +
  tm_borders() +
  tm_facets(by = "cluster")

tm_shape(shp_traffic_agg_cluster) +
  tm_fill("volume_sum", palette = "viridis", style = "quantile", title = "Sum Volume") +
  tm_borders() +
  tm_facets(by = "cluster")

# Boxplots
ggplot(shp_traffic_agg_cluster, aes(x = as.factor(cluster), y = speed_weighted, fill = as.factor(cluster))) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Boxplot of Weighted Speed by Cluster", x = "Cluster", y = "Weighted Speed") +
  theme_minimal() + theme(legend.position = "none", axis.text.y = element_blank(),axis.ticks.y = element_blank()) +
  scale_fill_viridis_d()

ggplot(shp_traffic_agg_cluster, aes(x = as.factor(cluster), y = volume_sum, fill = as.factor(cluster))) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Boxplot of Sum Volume by Cluster", x = "Cluster", y = "Sum Volume") +
  theme_minimal() + theme(legend.position = "none", axis.text.y = element_blank(),axis.ticks.y = element_blank()) +
  scale_fill_viridis_d()


## Usage Type Analysis

# Define usage variables
codi_emo_excel <- codi_emo_excel %>% rename("Sense_us_definit" = "Sense us definit")
usage_vars <- names(codi_emo_excel[,2:18])

# Join with usage data
traffic_agg_excel <- traffic_agg %>%
  left_join(codi_emo_excel, by = "codi_emo") 

# Reshape to long format for plotting
df_long <- traffic_agg_excel %>%
  dplyr::select(codi_emo, cluster, all_of(usage_vars)) %>%
  pivot_longer(cols = all_of(usage_vars), names_to = "usage_type", values_to = "value")


# Boxplot
df_long %>%
  group_by(usage_type) %>%
  mutate(
    Q1 = quantile(value, 0.25, na.rm = TRUE),
    Q3 = quantile(value, 0.75, na.rm = TRUE),
    IQR = Q3 - Q1
  ) %>%
  filter(value >= (Q1 - 1.5 * IQR) & value <= (Q3 + 1.5 * IQR)) %>%
  ungroup() %>%
  ggplot(aes(x = as.factor(cluster), y = value, fill = as.factor(cluster))) +
  geom_boxplot(alpha = 0.7, outlier.shape = 0.6) +
  stat_summary(
    fun = mean,
    geom = "point",
    shape = 15,
    size = 3,
    color = "black"
  ) +
  
  facet_wrap(~ usage_type, scales = "free_y") +
  labs(
    title = "Boxplots of Usage Variables by Cluster",
    x = "Cluster", 
    y = "Value", 
    fill = "Cluster"
  ) +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_fill_viridis_d()



traffic_agg_excel_agg_usage <- traffic_agg_excel %>%
  group_by(cluster) %>%
  dplyr::summarise(across(all_of(usage_vars), mean, na.rm = TRUE)) %>%
  ungroup()
mean_long <- traffic_agg_excel_agg_usage %>%
  pivot_longer(cols = -cluster, names_to = "usage_var", values_to = "mean_value")

# Heatmap
mean_long <- mean_long %>%
  mutate(mean_log = log1p(mean_value))  # log(1 + x), garde 0 définie

ggplot(mean_long, aes(x = usage_var, y = factor(cluster), fill = mean_log)) +
  geom_tile() +
  scale_fill_gradientn(colors = viridis(5),
                       values = rescale(c(0, 1, 5, 10, max(mean_long$mean_value)))) +
  labs(title = "Heatmap (log scale) - Mean Usage Variables by Cluster",
       x = "Usage Variable",
       y = "Cluster",
       fill = "log(1 + Mean Value)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


## Reduction of dimensionality

# Prepare data for LDA
df_lda <- traffic_agg_excel %>%
  dplyr::select(all_of(usage_vars), cluster) %>%
  filter(!is.na(cluster))

# Ensure cluster is a factor
df_lda$cluster <- as.factor(df_lda$cluster)

# Linear Discriminant Analysis
lda_model <- lda(cluster ~ ., data = df_lda)
print(lda_model)

# Predict and visualize LDA
lda_pred <- predict(lda_model)
df_scores <- data.frame(lda_pred$x, cluster = df_lda$cluster)

ggplot(df_scores, aes(x = LD1, y = LD2, color = cluster)) +
  geom_point(alpha = 0.7) +
  labs(title = "LDA - Projection of Clusters on Discriminant Axes",
       x = "Linear Discriminant 1", y = "Linear Discriminant 2", color = "Cluster") +
  theme_minimal() +
  scale_color_viridis_d()
  
# The graph shows how distinct the clusters are according to these linear combinations (LD1, LD2).

# Show variable importance on discriminant axes
print(lda_model$scaling)

## PCA

df_pca <- traffic_agg_excel %>%
  dplyr::select(all_of(usage_vars), cluster) %>%
  filter(!is.na(cluster))

# PCA
pca_model <- prcomp(df_pca %>% dplyr::select(-cluster), scale. = TRUE)
df_pca_result <- data.frame(pca_model$x, cluster = df_pca$cluster)

# Visualization
ggplot(df_pca_result, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(alpha = 0.7) +
  labs(title = "PCA - Projection of Clusters on Principal Components",
       x = "Principal Component 1", y = "Principal Component 2", color = "Cluster") +
  theme_minimal() + 
  scale_color_viridis_d()


## Land use variables by cluster

# Merge with shapefile for mapping
shp_profiling <- merge(shp, traffic_agg_excel, by = "codi_emo")

# Plot land use variables by cluster
for (var in usage_vars) {
  print(tm_shape(shp_profiling) +
          tm_fill(col = var, palette = "viridis", style = "quantile", title = paste(var, "land use")) +
          tm_borders() +
          tm_facets(by = "cluster") +
          tm_layout(legend.outside = TRUE, main.title = paste("Distribution of", var, "by Cluster")))
}

## Significance usage variables for cluster

# Kruskal-Wallis test for significance
kruskal_test <- function(var){
  formula <- as.formula(paste(var, "~ cluster"))
  test <- kruskal.test(formula, data = traffic_agg_excel)
  return(data.frame(variable = var, p_value = test$p.value))
}
results_kruskal <- bind_rows(lapply(usage_vars, kruskal_test))
print(results_kruskal)


## Which clusters are distinguished by each significant socio-economic variable?
# Significant socio-economic variables
signif_vars_kruskal <- c("Agrari", "Comercial", "Esportiu", "Estacionament",
                         "Industria", "Magatzem", "Oficines", "Residencial", "NC")

# Reshape for boxplots
data_plot <- traffic_agg_excel %>%
  dplyr::select(cluster, all_of(signif_vars_kruskal)) %>%
  pivot_longer(cols = -cluster, names_to = "variable", values_to = "value")

# Boxplot of socio-economic variables by cluster
data_plot_clean <- data_plot %>%
  group_by(variable, cluster) %>%
  mutate(
    Q1 = quantile(value, 0.25, na.rm = TRUE),
    Q3 = quantile(value, 0.75, na.rm = TRUE),
    IQR = Q3 - Q1,
    lower_bound = Q1 - 1.5 * IQR,
    upper_bound = Q3 + 1.5 * IQR
  ) %>%
  ungroup() %>%
  filter(value >= lower_bound & value <= upper_bound)

ggplot(data_plot_clean, aes(x = as.factor(cluster), y = value, fill = as.factor(cluster))) +
  geom_boxplot(alpha = 0.7, outlier.shape = 0.2) +  # Boxplot sans outliers affichés
  stat_summary(
    fun = mean,
    geom = "point",
    shape = 15,
    size = 1,
    color = "black"
  ) +
  facet_wrap(~ variable, scales = "free_y") +
  labs(
    title = "Boxplots of Significant Socio-Economic Variables by Cluster",
    x = "Cluster", 
    y = "Value", 
    fill = "Cluster"
  ) +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_fill_viridis_d()

# dunnTest test post-hoc
posthoc_results <- list()
for (var in signif_vars_kruskal) {
  dunn_res <- FSA::dunnTest(as.formula(paste(var, "~ cluster")), data = traffic_agg_excel, method = "bh")
  res_df <- as_tibble(dunn_res$res)
  
  sig_pairs <- res_df %>%
    filter(P.adj < 0.05) %>%
    dplyr::select(Comparison, P.adj)
  
  posthoc_results[[var]] <- sig_pairs
}
posthoc_results


## Typical profile by cluster

usage_scaled <- traffic_agg_excel %>%
  dplyr::select(all_of(usage_vars)) %>%
  scale() %>%
  as.data.frame()

usage_scaled$cluster <- traffic_agg_excel$cluster

cluster_means <- usage_scaled %>%
  group_by(cluster) %>%
  summarise(across(everything(), mean, na.rm = TRUE)) %>%
  ungroup()

# fmsb::radarchart demande un format particulier : 
# 1re ligne = max, 2e = min, 3+ = données
radar_data <- cluster_means %>%
  tibble::column_to_rownames("cluster")

radar_ready <- rbind(
  apply(radar_data, 2, max),
  apply(radar_data, 2, min),
  radar_data
)

colors <- viridis(nrow(radar_ready) - 2, alpha = 1)

par(mfrow = c(1, nrow(cluster_means)))
for (i in 3:nrow(radar_ready)) {
  cluster_id <- i - 2
  print(radarchart(
    radar_ready[c(1, 2, i), ],
    axistype = 1,
    title = paste("Cluster", cluster_id),
    pcol = colors[cluster_id],
    pfcol = alpha(colors[cluster_id], 0.4),
    plwd = 2
  ))
}

