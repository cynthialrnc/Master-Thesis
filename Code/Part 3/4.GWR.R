library(arrow)  
library(dplyr)     
library(lubridate)   
library(stringr)  
library(readxl)  
library(data.table)
library(sf)
library(sp)
library(spgwr)     
library(tmap)     


# Load here dataset
file_path <- "C:/Users/HP/Documents/Uni/Spring/Data/matrix/BARCELONA/BARCELONA_traffic_dataframe_preparation.parquet"
df_here <- read_parquet(file_path)

holidays_BARCELONA <- c("2019-04-19", "2019-04-22", "2019-05-01", "2019-06-10", "2019-06-24")

Sys.setlocale("LC_TIME", "English")
df_here_date <- df_here %>%
  distinct(date) %>%
  mutate(
    date,
    weekday = lubridate::wday(date, label = TRUE),
    month = lubridate::month(date, label = TRUE),
    holiday = date %in% ymd(holidays_BARCELONA)
  )

df_here <- df_here %>%
  mutate(hour_full = str_sub(hour, 1, 2) %>% paste0(":00"))

## Load codi_emo dataset
codi_emo_excel <- read_excel("C:/Users/HP/Documents/Uni/Spring/02-BCN DATA/Cadastre20_usos_RMB_EMO.XLSX")

# Data preparation
codi_emo_excel$codi_emo <- as.factor(codi_emo_excel$codi_emo)
codi_emo_excel <- codi_emo_excel %>% select(-codi_emo_t)
codi_emo_excel <- codi_emo_excel %>%
  rowwise() %>%
  mutate(across(where(is.numeric), ~ . / sum(c_across(where(is.numeric))) * 100)) %>%
  ungroup()


## Load shapefile dataset
shp <- st_read("C:/Users/HP/Documents/Uni/Spring/02-BCN DATA/ZonesEMO_MacroZones/ZonesEMO_MacroZones.shp")

vars <- c(
  "speed_NE", "speed_NW", "speed_SE", "speed_SW",
  "volume_NE", "volume_NW", "volume_SE", "volume_SW"
)
speed_vars <- c("speed_NE", "speed_NW", "speed_SE", "speed_SW")
volume_vars <- c("volume_NE", "volume_NW", "volume_SE", "volume_SW")

## Visualize aggregated variables on the shapefile
setDT(df_here)

# Aggregation
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


# Merge Here and shapefile
shp_emo_data <- merge(shp, traffic_agg, by = "codi_emo")

# Convert to Spatial object (centroids for GWR)
shp_centroids <- st_centroid(shp_emo_data)
shp_sp <- as(shp_centroids, "Spatial")

# Aggregate volumes and speeds
weighted_speed_sum <- Reduce(`+`, mapply(function(spd, vol) {
  shp_sp@data[[spd]] * shp_sp@data[[vol]]
}, speed_vars, volume_vars, SIMPLIFY = FALSE))
volume_sum <- rowSums(shp_sp@data[, volume_vars], na.rm = TRUE)

shp_sp@data$aggregated_volume <- volume_sum
shp_sp@data$aggregated_speed <- weighted_speed_sum / volume_sum


# Standardize response and predictors
shp_sp@data$aggregated_volume_std <- scale(shp_sp@data$aggregated_volume)

shp_sp@data <- shp_sp@data %>%
  left_join(codi_emo_excel, by = "codi_emo")

pred_ec = c("Comercial", "Industria", "Oficines", "Magatzem", "Oci_i_hostaleria")
pred_res = c("Residencial", "Educacio", "Sanitari", "Cultural", "Religios")
pred_equip = c("Estacionament", "Edificis_publics", "Esportiu", "NC")
pred_div = c("Agrari", "Altres_usos", "Sense_us_definit")
pred_mixte = c("Residencial", "Comercial", "Industria", "Oficines", "Oci_i_hostaleria")
pred_speed = c("aggregated_speed")

list_pred <- list(
  economic = pred_ec,
  residentiel = pred_res,
  equipements = pred_equip,
  divers = pred_div,
  mixte = pred_mixte,
  speed = pred_speed
)

for (name in names(list_pred)) {
  print(name)
  preds <- list_pred[[name]]
  preds_std <- paste0(preds, "_std")
  
  # Standardize predictors
  for (var in preds) {
    shp_sp@data[[paste0(var, "_std")]] <- scale(shp_sp@data[[var]])
  }
  
  # Build formula
  form <- as.formula(paste("aggregated_volume_std ~", paste(preds_std, collapse = " + ")))
  
  # Bandwidth selection and GWR
  bw <- gwr.sel(form, data = shp_sp, coords = coordinates(shp_sp), adapt = TRUE)
  gwr_res <- gwr(form, data = shp_sp, coords = coordinates(shp_sp), adapt = bw, hatmatrix = TRUE, se.fit = TRUE)
  
  # Convert to sf and add codi_emo
  gwr_sf <- st_as_sf(gwr_res$SDF)
  gwr_sf$codi_emo <- shp_sp@data$codi_emo
  
  # Join coefficients with original shapefile
  shp_gwr <- left_join(shp, st_drop_geometry(gwr_sf), by = "codi_emo")
  
  # Visualize coefficients for each predictor in the current lot
  for (var in preds) {
    coef_name <- paste0(var, "_std")
    print(
      tm_shape(shp_gwr) +
        tm_polygons(coef_name, palette = "viridis", style = "cont",
                    title = paste0("Coeff. ", var, " (GWR)")) +
        tm_layout(legend.outside = TRUE)
    )
  }
}