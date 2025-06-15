library(arrow)
library(dplyr)
library(tidyr)
library(ggplot2)
library(FactoMineR)
library(factoextra)
library(xtable)
library(viridis)
library(NbClust)
library(stringr)


# Set city and load transformed dataset
city <- "ANTWERP"
file_path <- paste0("C:/Users/HP/Documents/Uni/Spring/Data/matrix/", city)
data <- read_parquet(paste0(file_path, "/final_data_transformed.parquet"))
# head(data)

# Only numeric variables
categorical_var <- names(data)[sapply(data, function(x) !is.numeric(x))]
data_numeric <- data %>% select(-all_of(categorical_var))

# Classify variables into 'Volume', 'Speed', or 'Other' 
col_names <- colnames(data_numeric)
var_types <- data.frame(
  variable = col_names,
  type = ifelse(grepl("Volume", col_names), "Volume",
                ifelse(grepl("Speed", col_names), "Speed", "Other"))
)

# Classify variables into 'Volume', 'Speed', or 'Other'
var_counts <- var_types %>%
  group_by(type) %>%
  summarise(count = n())

# Bar plot of variable type distribution
ggplot(var_counts, aes(x = type, y = count, fill = type)) +
  geom_bar(stat = "identity") +
  ggtitle("Distribution of variables (Volume vs Speed)") +
  theme_minimal() +
  scale_fill_viridis_d(option = "D")



# faire un plot speed and volume
library(data.table)
# Convert to data.table
dt <- as.data.table(data_numeric)

# Get all column names
cols <- names(dt)

# Extract unique position names (e.g., "150_257")
positions <- unique(gsub("_(Speed|Volume)$", "", cols))

# Initialize vectors to store combined values
volume_list <- list()
speed_list <- list()

# Loop over each position and extract matching Speed and Volume columns
for (pos in positions) {
  vol_col <- paste0(pos, "_Volume")
  spd_col <- paste0(pos, "_Speed")
  
  # Check both columns exist
  if (vol_col %in% cols && spd_col %in% cols) {
    volume_list[[pos]] <- dt[[vol_col]]
    speed_list[[pos]]  <- dt[[spd_col]]
  }
}

# Combine all into long vectors
Volume <- unlist(volume_list, use.names = FALSE)
Speed  <- unlist(speed_list, use.names = FALSE)

# Create final data.table
scatter_dt <- data.table(Volume = Volume, Speed = Speed)

# Optional: remove NA or 0s
scatter_dt <- scatter_dt[!is.na(Volume) & !is.na(Speed)]

# Plot
ggplot(scatter_dt, aes(x = Speed, y = Volume)) +
  geom_point(alpha = 0.05) +
  labs(
    title = "Scatter Plot of Volume vs Speed",
    x = "Speed",
    y = "Volume"
  ) +
  theme_minimal()

# Apply log1p transformation (log(1 + x)) to avoid log(0)
scatter_dt[, Volume_log := log1p(Volume)]
scatter_dt[, Speed_log := log1p(Speed)]

# Plot using the transformed values
ggplot(scatter_dt, aes(x = Speed_log, y = Volume_log)) +
  geom_point(alpha = 0.05) +
  labs(
    title = "Log-Transformed Scatter Plot of Volume vs Speed",
    x = "log(1 + Speed)",
    y = "log(1 + Volume)"
  ) +
  theme_minimal()



### PCA

# Load precomputed PCA result (or compute using PCA(...))
pca_result <- PCA(data_numeric, graph = FALSE)
file_path_pca <- paste0(file_path, "/pca_result.rds")
# pca_result <- readRDS(file_path_pca)
saveRDS(pca_result, file_path_pca)

# Calculate number of components needed to explain 70% varianc
cumulative_variance <- cumsum(pca_result$eig / sum(pca_result$eig))
threshold <- 0.70
num_components <- which(cumulative_variance >= threshold)[1]
num_components
cumulative_variance[num_components]
# comp 11792: cumulative percentage of variance -> 70%

# View coordinates of variables on main dimensions
var_coords <- as.data.frame(pca_result$var$coord)
head(var_coords[order(abs(var_coords$Dim.1), decreasing = TRUE), ], 10)
print(xtable(head(var_coords[order(abs(var_coords$Dim.1), decreasing = TRUE), ], 10), type = "latex"))
head(var_coords[order(abs(var_coords$Dim.2), decreasing = TRUE), ], 10)
print(xtable(head(var_coords[order(abs(var_coords$Dim.2), decreasing = TRUE), ], 10), type = "latex"))

dimdesc_pca <- dimdesc(pca_result, axes=1:2, proba=0.0000001)
print(xtable(head(as.data.frame(dimdesc_pca$Dim.1), 15)))
print(xtable(head(as.data.frame(dimdesc_pca$Dim.2), 15)))

# Print coordinates of individuals in PCA space
print(pca_result$ind$coord)
# This suggests that the first few components capture most of the differences in the dataset.

# Scree Plot (explained variance)
fviz_eig(pca_result)

# Variable contributions to principal components
# fviz_pca_var(pca_result, col.var = "contrib", label="None")
fviz_pca_var(pca_result, label="None")

var_names <- rownames(pca_result$var$coord)
var_type <- ifelse(grepl("Speed$", var_names), "Speed", "Volume")

# Plot with custom color by group
fviz_pca_var(pca_result, label = "none", col.var = var_type, geom = "point") +
  scale_color_manual(
    values = c("Speed" = "#482173", "Volume" = "#bddf26")
  ) +
  labs(color = "Variable Type")

viridis_palette <- viridis::viridis(length(unique(data$weekday)), option = "D", begin = 0, end = 1)

fviz_pca_ind(pca_result, 
             geom = "point", 
             habillage = data$weekday, 
             palette = viridis_palette, 
             addEllipses = TRUE, 
             repel = TRUE) + 
  ggtitle("Dim.1 et Dim.2 (colored by weekday)")

fviz_pca_ind(pca_result, 
             geom = "point", 
             habillage = data$month, 
             palette = viridis_palette, 
             addEllipses = TRUE, 
             repel = TRUE) + 
  ggtitle("Dim.1 et Dim.2 (colored by month)")

fviz_pca_ind(pca_result, 
             geom = "point", 
             habillage = data$holiday, 
             palette = viridis_palette, 
             addEllipses = TRUE, 
             repel = TRUE) + 
  ggtitle("Dim.1 et Dim.2 (coloré par holiday)")

fviz_pca_ind(pca_result, 
             geom = "point", 
             habillage = data$hour_group,
             palette = viridis(length(unique(data$hour_group)), option = "D"),
             addEllipses = TRUE,   
             repel = TRUE) + 
  ggtitle("Dim.1 et Dim.2 (colored by hour)")

data$hour_group <- factor(data$hour_group, levels = c(
  "5-6h", "7-8h", "9-11h", "12-14h", "15-16h", "17-19h", "20-22h", "23-4h"
))
fviz_pca_ind(pca_result, 
             geom = "point", 
             habillage = data$hour_group,
             palette = viridis(length(levels(data$hour_group)), option = "D"),
             addEllipses = TRUE,   
             repel = TRUE) + 
  ggtitle("Dim.1 et Dim.2 (colored by hour)")


# Scatter plot of variables colored by type (Speed/Volume)
var_coords$type <- as.factor(var_types$type) 
ggplot(var_coords, aes(x = Dim.1, y = Dim.2, color = type)) +
  geom_point(size = 0.2) +
  scale_color_viridis_d(option = "D") +
  labs(title = "Volume vs Speed") +
  theme_minimal() +
  theme(legend.title = element_blank())

fviz_pca_ind(pca_result, 
             geom = "point", 
             habillage = data$cluster,
             palette = viridis(length(unique(data$cluster)), option = "D"),
             addEllipses = TRUE,   
             repel = TRUE) + 
  ggtitle("Dim.1 et Dim.2 (colored by cluster)")

### Clustering

# Check optimal number of clusters (optional visual tools)
fviz_nbclust(pca_result$ind$coord, method = "wss", FUNcluster = kmeans)
fviz_nbclust(pca_result$ind$coord, method = "silhouette", FUNcluster = kmeans)
gap_stat <- clusGap(pca_result$ind$coord, FUNcluster = kmeans, K.max = 50, B = 100)
fviz_gap_stat(gap_stat)

# Run k-means with chosen number of clusters
kmeans_result <- kmeans(pca_result$ind$coord, centers = 4)
fviz_cluster(kmeans_result, data = pca_result$ind$coord)

# Evaluation to choose number of clusters
coords <- pca_result$ind$coord 
NbClust(data = coords, distance = "euclidean", min.nc = 2, max.nc = 10, method = "ward.D")

### HCPC clustering (hierarchical clustering on principal components)

res.hcpc <- HCPC(pca_result, nb.clust = 4, graph = FALSE)

file_path_hcpc <- paste0(file_path, "/res_hcpc_4.rds")

# res.hcpc <- readRDS(file_path_hcpc)
saveRDS(res.hcpc, file_path_hcpc)

# plot(res.hcpc, choice = "3D.map")

# Inspect cluster distribution and descriptors
table(res.hcpc$data.clust$clust)
res.hcpc$desc.var
res.hcpc$desc.ind

print(xtable(table(res.hcpc$data.clust$clust), type = "latex"))

# Visualize clusters in PCA space
# fviz_cluster(res.hcpc)
fviz_cluster(res.hcpc, 
             palette = viridis(length(unique(res.hcpc$data.clust$clust)), option = "D"),
             geom = "point", 
             ellipse.type = "convex",
             repel = TRUE,
             ggtheme = theme_minimal())


### Profiling I

# Add cluster labels to numeric and original data
data_numeric$cluster <- res.hcpc$data.clust$clust
data_numeric$cluster <- as.numeric(data_numeric$cluster)
data$cluster <- res.hcpc$data.clust$clust
data$cluster <- as.factor(data$cluster)

## Frequency tables for each factor variable by cluster
qualitative_profiling <- data %>%
  group_by(cluster) %>%
  summarise(across(where(is.factor), ~ list(table(.)), .names = "freq_{col}"))

# Frequency of hours by cluster
data %>%
  group_by(cluster, hour) %>%
  summarise(count = n(), .groups = 'drop') %>%
  mutate(freq = count / sum(count)) %>%
  ggplot(aes(x = cluster, y = freq, fill = as.factor(hour))) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_viridis_d(option = "D", name = "Hour") +
  labs(title = "Frequency of hours by cluster", x = "Cluster", y = "Frequencie") +
  theme_minimal()

# Frequency of weekdays by cluster
data %>%
  group_by(cluster, weekday) %>%
  summarise(count = n(), .groups = 'drop') %>%
  mutate(freq = count / sum(count)) %>%
  ggplot(aes(x = as.factor(cluster), y = freq, fill = as.factor(weekday))) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_viridis_d(option = "D", name = "Weekday") +
  labs(title = "Frequency of weekdays by cluster", x = "Cluster", y = "Frequencie") +
  theme_minimal()

# Frequency of months by cluster
data %>%
  group_by(cluster, month) %>%
  summarise(count = n(), .groups = 'drop') %>%
  mutate(freq = count / sum(count)) %>%
  ggplot(aes(x = as.factor(cluster), y = freq, fill = as.factor(month))) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_viridis_d(option = "D", name = "Month") +
  labs(title = "Frequency of months by cluster", x = "Cluster", y = "Frequencie") +
  theme_minimal()

# Holiday proportion per cluster vs global
df_holiday <- data %>%
  group_by(cluster) %>%
  summarise(freq_holiday = mean(holiday, na.rm = TRUE), 
            count = n(), .groups = 'drop')
prop_holiday_global <- mean(data$holiday, na.rm = TRUE)

ggplot(df_holiday, aes(x = as.factor(cluster), y = freq_holiday, fill = as.factor(cluster))) +
  geom_bar(stat = "identity") +
  scale_fill_viridis_d(option = "D", name = "Cluster") +
  geom_hline(yintercept = prop_holiday_global, linetype = "dashed", color = "black") +  # Ligne horizontale à la moyenne globale
  labs(title = "Proportion of holidays per cluster",
       x = "Cluster", 
       y = "Proportion of holidays") +
  theme_minimal()



### Profiling II

# Extract volume and speed variable names
volume_vars <- var_types$variable[var_types$type == "Volume"]
speed_vars <- var_types$variable[var_types$type == "Speed"]

# Reshape volume and speed data to long format
data_long_volume <- data %>%
  select(all_of(c("date_hour", "hour", "cluster", "weekday", "holiday", "month", volume_vars))) %>%
  pivot_longer(cols = all_of(volume_vars), names_to = "sensor", values_to = "volume") %>%
  mutate(sensor_id = str_remove(sensor, "_Volume"), type = "volume")

data_long_speed <- data %>%
  select(all_of(c("date_hour", "hour", "cluster", "weekday", "holiday", "month", speed_vars))) %>%
  pivot_longer(cols = all_of(speed_vars), names_to = "sensor", values_to = "speed") %>%
  mutate(sensor_id = str_remove(sensor, "_Speed"), type = "speed")

data_long_combined <- data_long_volume %>%
  select(date_hour, hour, cluster, weekday, holiday, month, sensor_id, volume) %>%
  left_join(
    data_long_speed %>%
      select(date_hour, cluster, sensor_id, speed),
    by = c("date_hour", "cluster", "sensor_id")
  )

summary(data_long_combined %>%
  filter(volume > 0))
first_quartile_volume <- data_long_combined %>%
  filter(volume > 0) %>%
  summarise(
    q1 = quantile(volume, probs = 0.25, na.rm = TRUE)
  ) %>%
  pull(q1)

# barna : first_quartile_volume <- 0.021
# antwerp : first_quartile_volume <- 0.04166667
# berlin : first_quartile_volume <- 0.1458333

data_long_filtered <- data_long_combined %>%
  filter(volume > 0.1458333)

## Compute median, SD, and IQR per cluster
cluster_variability_volume <- data_long_filtered %>%
  group_by(cluster) %>%
  summarise(across(volume, list(median = \(x) median(x, na.rm = TRUE),
                                sd = \(x) sd(x, na.rm = TRUE), 
                                iqr = \(x) IQR(x, na.rm = TRUE))))

cluster_variability_speed <- data_long_filtered %>%
  group_by(cluster) %>%
  summarise(across(speed, list(median = \(x) median(x, na.rm = TRUE),
                               sd = \(x) sd(x, na.rm = TRUE), 
                               iqr = \(x) IQR(x, na.rm = TRUE))))

# Median + IQR error bars for volume and speed
ggplot(cluster_variability_speed, aes(x = cluster, y = speed_median, fill = cluster)) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = speed_median - speed_iqr / 2, ymax = speed_median + speed_iqr / 2),
                width = 0.2, color = "black") +
  labs(title = "Median Traffic Speed by Cluster",
       y = "Speed", x = "Cluster") +
  theme_minimal()


ggplot(cluster_variability_volume, aes(x = cluster, y = volume_median, fill = cluster)) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = volume_median - volume_iqr / 2, ymax = volume_median + volume_iqr / 2),
                width = 0.2, color = "black") +
  labs(title = "Median Traffic Volume by Cluster",
       y = "Volume", x = "Cluster") +
  theme_minimal()


## Identify extreme congestion (volume[!0] > 90th percentile)
volume_thresholds <- data_long_filtered %>%
  group_by(sensor_id) %>%
  summarise(threshold_90 = quantile(volume, 0.90, na.rm = TRUE), .groups = "drop")

extreme_congestion <- data_long_filtered %>%
  inner_join(volume_thresholds, by = "sensor_id") %>%
  filter(volume > threshold_90)

extreme_congestion_count <- extreme_congestion %>%
  group_by(cluster) %>%
  summarise(count = n(), .groups = "drop")

ggplot(extreme_congestion_count, aes(x = as.factor(cluster), y = count, fill = as.factor(cluster))) +
  geom_bar(stat = "identity") +
  scale_fill_viridis_d(option = "D", name = "Cluster") +
  labs(title = "Extreme Congestion Events per Cluster", x = "Cluster", y = "Count") +
  theme_minimal()

### Identify extreme slowdowns (speed[!0] < 10th percentile)
# Compute 10th percentile for each sensor
speed_thresholds <- data_long_filtered %>%
  group_by(sensor_id) %>%
  summarise(threshold_10 = quantile(speed, 0.10, na.rm = TRUE), .groups = "drop")

# Identify extreme slowdown events
extreme_slowdowns <- data_long_filtered %>%
  inner_join(speed_thresholds, by = "sensor_id") %>%
  filter(speed < threshold_10)

# Count occurrences per cluster
extreme_slowdown_count <- extreme_slowdowns %>%
  group_by(cluster) %>%
  summarise(count = n(), .groups = "drop")

# Plot distribution of extreme slowdowns per cluster
ggplot(extreme_slowdown_count, aes(x = as.factor(cluster), y = count, fill = cluster)) +
  geom_bar(stat = "identity") +
  scale_fill_viridis_d(option = "D", name = "Cluster") +
  labs(title = "Extreme Slowdown Events per Cluster", x = "Cluster", y = "Count") +
  theme_minimal()


## Correlation Between Volume and Speed per Cluster and Sensor
sensor_ids_volume <- gsub("_Volume", "", volume_vars)
sensor_ids_speed <- gsub("_Speed", "", speed_vars)
common_sensors <- intersect(sensor_ids_volume, sensor_ids_speed)

cor_results <- list()  # Store correlation results

for (clus in unique(data$cluster)) {
  cluster_data <- data %>% filter(cluster == clus)
  
  for (sensor in common_sensors) {
    print(paste0(clus, " ", sensor))
    vol_col <- paste0(sensor, "_Volume")
    spd_col <- paste0(sensor, "_Speed")
    
    if (all(c(vol_col, spd_col) %in% names(cluster_data))) {
      tmp <- cluster_data %>% select(all_of(c(vol_col, spd_col))) %>% drop_na()
      
      if (nrow(tmp) > 1) {
        cor_value <- cor(tmp[[vol_col]], tmp[[spd_col]], use = "complete.obs")
      } else {
        cor_value <- NA
      }

      cor_results[[length(cor_results) + 1]] <- data.frame(
        cluster = clus,
        sensor = sensor,
        correlation = cor_value
      )
    }
  }
}

cor_results <- bind_rows(cor_results)
write.csv(cor_results, paste0(file_path, "/cor_results_4.csv"))

# Correlation bar chart and heatmap
ggplot(cor_results, aes(x = as.factor(cluster), y = correlation, fill = as.factor(cluster))) +
  geom_violin(trim = FALSE, alpha = 0.6) +
  geom_boxplot(width = 0.1, outlier.shape = NA) +
  scale_fill_viridis_d(option = "D", name = "Cluster") + 
  labs(title = "Distribution of Correlations per Cluster",
       x = "Cluster", y = "Correlation") +
  theme_minimal()

ggplot(cor_results, aes(x = correlation, fill = as.factor(cluster))) +
  geom_histogram(binwidth = 0.05, alpha = 0.7, color = "white") +
  scale_fill_viridis_d(option = "D", name = "Cluster") + 
  facet_wrap(~ cluster, scales = "free_y") +
  labs(title = "Histogram of Correlations by Cluster", x = "Correlation", y = "Count") +
  theme_minimal()


## Hourly Traffic Analysis by Cluster
hourly_traffic <- data_long_filtered %>%
  group_by(cluster, hour) %>%
  summarise(sum_volume = sum(volume, na.rm = TRUE),
            mean_speed = mean(speed, na.rm = TRUE),
            weighted_speed = if (sum(volume, na.rm = TRUE) == 0) NA else sum(speed * volume, na.rm = TRUE) / sum(volume, na.rm = TRUE),
            .groups = 'drop')

# Volume sum heatmap
hourly_traffic_volume_sum <- hourly_traffic %>%
  mutate(volume_decile = cut(sum_volume,
                             breaks = quantile(sum_volume, probs = seq(0, 1, 0.1), na.rm = TRUE),
                             include.lowest = TRUE,
                             labels = paste0("Decile ", 1:10)))  # Labels: D1 to D10
ggplot(hourly_traffic_volume_sum, aes(x = hour, y = as.factor(cluster), fill = volume_decile)) +
  geom_tile() +
  scale_fill_viridis_d(option = "D") +  
  labs(title = "Heatmap of Sum Traffic Volume by Hour and Cluster",
       x = "Hour", y = "Cluster", fill = "Volume Decile") +
  theme_minimal()

# Speed weighted heatmap
hourly_traffic_speed_weighted <- hourly_traffic %>%
  mutate(speed_decile = cut(weighted_speed,
                            breaks = quantile(weighted_speed, probs = seq(0, 1, 0.1), na.rm = TRUE),
                            include.lowest = TRUE,
                            labels = paste0("Decile ", 1:10)))  # Labels: D1 to D10
ggplot(hourly_traffic_speed_weighted, aes(x = hour, y = as.factor(cluster), fill = speed_decile)) +
  geom_tile() +
  scale_fill_viridis_d(option = "D") +  
  labs(title = "Heatmap of Weighted Traffic Speed by Hour and Cluster",
       x = "Hour", y = "Cluster", fill = "Speed Decile") +
  theme_minimal()

# Speed average heatmap
hourly_traffic_speed_mean <- hourly_traffic %>%
  mutate(speed_decile = cut(mean_speed,
                            breaks = quantile(mean_speed, probs = seq(0, 1, 0.1), na.rm = TRUE),
                            include.lowest = TRUE,
                            labels = paste0("Decile ", 1:10)))  # Labels: D1 to D10
ggplot(hourly_traffic_speed_mean, aes(x = hour, y = as.factor(cluster), fill = speed_decile)) +
  geom_tile() +
  scale_fill_viridis_d(option = "D") +  
  labs(title = "Heatmap of Average Traffic Speed by Hour and Cluster",
       x = "Hour", y = "Cluster", fill = "Speed Decile") +
  theme_minimal()









################################################################################
## OLD

# Aggregation : average and median per hour and cluster
hourly_traffic_volume <- data_long_filtered %>%
  group_by(cluster, hour) %>%
  summarise(mean_volume = mean(volume, na.rm = TRUE),
            median_volume = median(volume, na.rm = TRUE),
            sum_volume = sum(volume, na.rm = TRUE),
            .groups = 'drop')
hourly_traffic_volume_sum <- hourly_traffic_volume %>%
  mutate(volume_decile = cut(sum_volume,
                             breaks = quantile(sum_volume, probs = seq(0, 1, 0.1), na.rm = TRUE),
                             include.lowest = TRUE,
                             labels = paste0("Decile ", 1:10)))  # Labels: D1 to D10

# Volume heatmap
ggplot(hourly_traffic_volume_sum, aes(x = hour, y = as.factor(cluster), fill = volume_decile)) +
  geom_tile() +
  scale_fill_viridis_d(option = "D") +  
  labs(title = "Heatmap of Sum Traffic Volume by Hour and Cluster",
       x = "Hour", y = "Cluster", fill = "Volume Decile") +
  theme_minimal()

# Volume trends
ggplot(hourly_traffic_volume, aes(x = hour, y = mean_volume, color = as.factor(cluster), group = cluster)) +
  geom_line() +
  scale_color_viridis_d(option = "D", name = "Cluster") +
  labs(title = "Traffic Volume Trends by Hour and Cluster",
       x = "Hour", y = "Average Volume") +
  theme_minimal()


# Aggregation : average and median per hour and cluster
hourly_traffic_speed <- data_long_filtered %>%
  group_by(cluster, hour) %>%
  summarise(mean_speed = mean(speed, na.rm = TRUE),
            median_speed = median(speed, na.rm = TRUE),
            sum_speed = sum(speed, na.rm = TRUE),
            .groups = 'drop')
# For average
hourly_traffic_speed_mean <- hourly_traffic %>%
  mutate(speed_decile = cut(mean_speed,
                            breaks = quantile(mean_speed, probs = seq(0, 1, 0.1), na.rm = TRUE),
                            include.lowest = TRUE,
                            labels = paste0("Decile ", 1:10)))  # Labels: D1 to D10

# Speed heatmap
ggplot(hourly_traffic_speed_mean, aes(x = hour, y = as.factor(cluster), fill = speed_decile)) +
  geom_tile() +
  scale_fill_viridis_d(option = "D") +  
  labs(title = "Heatmap of Average Traffic Speed by Hour and Cluster",
       x = "Hour", y = "Cluster", fill = "Speed Decile") +
  theme_minimal()

## Weekly Trends
filtered_speed_data <- data_long_speed %>%
  filter(sensor %in% gsub("Volume", "Speed", data_long_volume$sensor[data_long_volume$volume == 0]))

weekly_traffic_volume <- data_long_volume %>%
  group_by(cluster, weekday) %>%
  summarise(mean_volume = mean(volume, na.rm = TRUE),
            median_volume = median(volume, na.rm = TRUE),
            sum_volume = sum(volume, na.rm = TRUE),
            .groups = 'drop')

weekly_traffic_speed <- filtered_speed_data %>%
  group_by(cluster, weekday) %>%
  summarise(mean_speed = mean(speed, na.rm = TRUE),
            median_speed = median(speed, na.rm = TRUE),
            sum_speed = sum(speed, na.rm = TRUE),
            .groups = 'drop')

ggplot(weekly_traffic_speed, aes(x = as.factor(weekday), y = mean_speed, fill = factor(cluster))) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_viridis_d(option = "D") +  
  labs(title = "Average speed by day of the week and cluster", x = "Weekday", y = "Mean Volume") +
  theme_minimal()


# Bar plots by weekday for Sum
ggplot(weekly_traffic_volume, aes(x = as.factor(weekday), y = sum_volume, fill = factor(cluster))) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_viridis_d(option = "D") + 
  labs(title = "Sum volume by day of the week and cluster", x = "Weekday", y = "Sum Volume") +
  theme_minimal()



