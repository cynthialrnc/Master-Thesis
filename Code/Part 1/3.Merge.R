library(arrow)
library(dplyr)

# Read all files
city <- "ANTWERP"
directory_path <- paste0("C:/Users/HP/Documents/Uni/Spring/Data/matrix/", city)
file_paths <- list.files(
  directory_path, 
  pattern = "\\.parquet$", 
  full.names = TRUE,       
  recursive = FALSE  
)

data_list <- lapply(file_paths, function(file) {
  read_parquet(file)
})

# Identify same columns
column_names_list <- lapply(data_list, colnames)
common_columns <- Reduce(intersect, column_names_list)

# Filter columns for each file
filtered_data_list <- lapply(data_list, function(df) {
  df[, common_columns, drop = FALSE]
})
data <- bind_rows(filtered_data_list)

# Write new file
write_parquet(data, paste0(directory_path, "/final_data.parquet"))
