library(dplyr)
library(ggplot2)
library(readxl)
library(tidyr)

city <- "BARCELONA"
file_path_map <- "C:/Users/HP/Documents/Uni/Spring/Data/map/BARCELONA/"
codi_emo_grid <- read.csv(paste0(file_path_map, "codi_emo_grid_interpolated.csv"))

file_path_excel <- "C:/Users/HP/Documents/Uni/Spring/02-BCN DATA/"
codi_emo_excel <- read_excel(paste0(file_path_excel, "Cadastre20_usos_RMB_EMO.XLSX"), col_names = TRUE)

head(codi_emo_grid)
head(codi_emo_excel)

codi_emo_excel$codi_emo <- as.factor(codi_emo_excel$codi_emo)
codi_emo_excel <- codi_emo_excel %>% select(-codi_emo_t)

# Normalize numeric columns row-wise to percentages
codi_emo_excel <- codi_emo_excel %>%
  rowwise() %>%  # Treat each row individually
  mutate(across(where(is.numeric), ~ . / sum(c_across(where(is.numeric))) * 100)) %>%
  ungroup()

# Convert codi_emo_grid to long format with row and column indices
colnames(codi_emo_grid) <- 0:(ncol(codi_emo_grid) - 1)
codi_emo_long <- codi_emo_grid %>%
  mutate(row = row_number() - 1) %>%  # Add row index
  pivot_longer(
    cols = -row,                 # All columns except "row"
    names_to = "col",            # New column name for former column names
    values_to = "codi_emo"       # New column name for values
  ) %>%
  mutate(col = as.numeric(col))

# Convert to factors
codi_emo_long$codi_emo <- as.factor(codi_emo_long$codi_emo)
codi_emo_long$row <- as.factor(codi_emo_long$row)
codi_emo_long$col <- as.factor(codi_emo_long$col)

# Join with codi_emo_excel using codi_emo as key
codi_emo_joined <- codi_emo_long %>%
  left_join(codi_emo_excel, by = "codi_emo")
codi_emo_final <- codi_emo_joined

head(codi_emo_final)

# Check variable types
sapply(codi_emo_final, class)

# Save final table to CSV
write.csv(codi_emo_final, paste0(file_path_map, "codi_emo.csv"))
