library(arrow)
library(dplyr)
library(tidyr)
library(stringr)

# Set the city and load the transformed dataset
city <- "BARCELONA"
file_path <- paste0("C:/Users/HP/Documents/Uni/Spring/Data/matrix/", city)
data <- read_parquet(paste0(file_path, "/final_data_transformed_clustered.parquet"))

# Extract only date_time and cluster columns
data_cluster <- data[, c("date_time", "cluster")]

# Create date and hour columns, then remove date_time
data_cluster <- data_cluster %>%
  mutate(
    date = as.Date(date_time),            # Extract date
    hour = format(date_time, "%H:%M")     # Extract hour as HH:MM
  ) %>%
  select(-date_time)

# Save the result to a CSV file
write.csv(data_cluster, paste0(file_path, "/data_cluster.csv"), row.names = FALSE)
