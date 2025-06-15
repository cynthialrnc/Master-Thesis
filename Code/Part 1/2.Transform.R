library(arrow)       # For reading and writing Parquet files
library(dplyr)       # For data manipulation
library(lubridate)   # For working with dates and times

Sys.setlocale("LC_TIME", "English")  # Ensure weekdays/months are in English

city <- "ANTWERP"
directory_path <- paste0("C:/Users/HP/Documents/Uni/Spring/Data/matrix/", city)
data <- read_parquet(paste0(directory_path, "/final_data.parquet"))  # Load raw data

# Define public holidays for each city
holidays_list <- list(
  BARCELONA = c("2019-04-19", "2019-04-22", 
                "2019-05-01", "2019-06-10", "2019-06-24"),
  BERLIN = c("2019-04-19", "2019-04-22", "2019-05-01", "2019-06-10"),
  ANTWERP = c("2019-04-21", "2019-04-22", 
              "2019-05-01", "2019-05-30", "2019-06-10")
)
holiday <- holidays_list[[city]]  # Select holidays for the chosen city

# Remove columns where over 50% of values are zero
threshold <- 0.5
data_filtered <- data[, (colMeans(data == 0)) < threshold]

# Add time-related features
data_filtered <- data_filtered %>%
  mutate(
    date_time = dmy_h(date_hour),                      # Convert to datetime
    hour = as.factor(hour(date_time)),                 # Extract hour as a factor
    weekday = wday(date_time, label = TRUE),           # Extract day of the week
    month = month(date_time, label = TRUE),            # Extract month
    holiday = as.Date(data_filtered$date_hour, format = "%d/%m/%Y") %in% ymd(holiday)  # Mark holidays
  )

# Group hours into custom time intervals
data_filtered <- data_filtered %>%
  mutate(
    hour_group = case_when(
      hour %in% 0:4 | hour == 23 ~ "23-4h", 
      hour %in% 5:6 ~ "5-6h",
      hour %in% 7:8 ~ "7-8h",
      hour %in% 9:11 ~ "9-11h",
      hour %in% 12:14 ~ "12-14h",
      hour %in% 15:16 ~ "15-16h",
      hour %in% 17:19 ~ "17-19h",
      hour %in% 20:22 ~ "20-22h"
    )
  )

data_filtered$hour_group <- as.factor(data_filtered$hour_group)  # Convert to factor

# Save the transformed dataset
write_parquet(data_filtered, paste0(directory_path, "/final_data_transformed.parquet"))
