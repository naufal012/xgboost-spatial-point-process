# ============================================================
# BCI Preprocessing for XGBoostPP
# ============================================================
#
# Author  : Your Name
# Purpose :
#   Generate Poisson and Logistic quadrature datasets
#   from BCI tree species data.
#
# Input:
#   - bci.tree1.rdata
#   - bci.covars.Rda
#
# Output:
#   - CSV files for each species
#
# Example Outputs:
#   - beinonscale.csv
#   - beiloginonscale2.csv
#   - rynononscale.csv
#   - rynologinonscale2.csv
#
# ============================================================

# ============================================================
# LOAD REQUIRED LIBRARIES
# ============================================================

required_packages <- c(
  "spatstat.data",
  "spatstat.geom",
  "spatstat.explore",
  "spatstat",
  "raster",
  "dplyr"
)

for (pkg in required_packages) {
  
  if (!require(pkg, character.only = TRUE)) {
    
    install.packages(pkg)
    
    library(pkg, character.only = TRUE)
    
  }
  
}

# ============================================================
# LOAD DATA
# ============================================================

# Example:
# load("data/bci.tree1.rdata")
# load("data/bci.covars.Rda")

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

dir.create("data", showWarnings = FALSE)

dir.create("data/processed", showWarnings = FALSE)

# ============================================================
# SPECIES CONFIGURATION
# ============================================================

species_configs <- list(
  
  bei = list(
    species_code = "beilpe",
    poisson_output = "data/processed/beinonscale.csv",
    logistic_output = "data/processed/beiloginonscale2.csv"
  ),
  
  ryno = list(
    species_code = "ade1tr",
    poisson_output = "data/processed/rynononscale.csv",
    logistic_output = "data/processed/rynologinonscale2.csv"
  )
  
)

# ============================================================
# CREATE COVARIATE RASTERS
# ============================================================

cat("\n====================================================\n")
cat("Preparing Covariate Rasters\n")
cat("====================================================\n")

covariate_rasters <- list(
  
  Cu = raster(as.im(bci.covars$Cu)),
  
  grad = raster(as.im(bci.covars$grad)),
  
  elev = raster(as.im(bei.extra$elev)),
  
  Nmin = raster(as.im(bci.covars$Nmin)),
  
  P = raster(as.im(bci.covars$P)),
  
  pH = raster(as.im(bci.covars$pH)),
  
  solar = raster(as.im(bci.covars$solar)),
  
  twi = raster(as.im(bci.covars$twi))
  
)

# ============================================================
# FUNCTION:
# CREATE QUADRATURE DATASET
# ============================================================

create_quadrature_dataset <- function(
    quadrature_scheme,
    covariate_rasters
) {
  
  # ----------------------------------------------------------
  # Event coordinates
  # ----------------------------------------------------------
  
  event_coordinates <- data.frame(
    x = quadrature_scheme$data$x,
    y = quadrature_scheme$data$y
  )
  
  # ----------------------------------------------------------
  # Dummy coordinates
  # ----------------------------------------------------------
  
  dummy_coordinates <- data.frame(
    x = quadrature_scheme$dummy$x,
    y = quadrature_scheme$dummy$y
  )
  
  # ----------------------------------------------------------
  # Quadrature weights
  # ----------------------------------------------------------
  
  quadrature_weights <- quadrature_scheme$w
  
  label_vector <- is.data(quadrature_scheme)
  
  weight_event <- quadrature_weights[label_vector]
  
  weight_dummy <- quadrature_weights[!label_vector]
  
  # ----------------------------------------------------------
  # Extract covariates for event points
  # ----------------------------------------------------------
  
  event_covariates <- lapply(
    
    covariate_rasters,
    
    function(raster_layer) {
      
      raster::extract(
        raster_layer,
        event_coordinates
      )
      
    }
    
  )
  
  # ----------------------------------------------------------
  # Extract covariates for dummy points
  # ----------------------------------------------------------
  
  dummy_covariates <- lapply(
    
    covariate_rasters,
    
    function(raster_layer) {
      
      raster::extract(
        raster_layer,
        dummy_coordinates
      )
      
    }
    
  )
  
  # ----------------------------------------------------------
  # Event dataset
  # ----------------------------------------------------------
  
  event_dataset <- data.frame(
    
    event_coordinates,
    
    event_covariates,
    
    label = 1,
    
    vol = weight_event,
    
    int = 1
    
  )
  
  # ----------------------------------------------------------
  # Dummy dataset
  # ----------------------------------------------------------
  
  dummy_dataset <- data.frame(
    
    dummy_coordinates,
    
    dummy_covariates,
    
    label = -1,
    
    vol = weight_dummy,
    
    int = 0
    
  )
  
  # ----------------------------------------------------------
  # Remove missing values
  # ----------------------------------------------------------
  
  dummy_dataset <- na.omit(dummy_dataset)
  
  # ----------------------------------------------------------
  # Combine event and dummy datasets
  # ----------------------------------------------------------
  
  final_dataset <- rbind(
    event_dataset,
    dummy_dataset
  )
  
  return(final_dataset)
  
}

# ============================================================
# MAIN LOOP
# ============================================================

for (dataset_name in names(species_configs)) {
  
  cat("\n====================================================\n")
  cat("Processing Dataset :", dataset_name, "\n")
  cat("====================================================\n")
  
  config <- species_configs[[dataset_name]]
  
  # ----------------------------------------------------------
  # Filter species data
  # ----------------------------------------------------------
  
  species_data <- subset(
    bci.tree1,
    sp == config$species_code
  )
  
  species_data <- species_data[, c("gx", "gy")]
  
  species_data <- na.omit(species_data)
  
  species_data <- species_data[
    !duplicated(species_data),
  ]
  
  # ----------------------------------------------------------
  # Create point pattern
  # ----------------------------------------------------------
  
  species_ppp <- ppp(
    x = species_data$gx,
    y = species_data$gy,
    window = bei$window
  )
  
  cat("Number of points :", species_ppp$n, "\n")
  
  # ----------------------------------------------------------
  # Plot point pattern
  # ----------------------------------------------------------
  
  plot(
    species_ppp,
    main = paste(
      "Spatial Point Pattern -",
      dataset_name
    ),
    pch = 16,
    cex = 0.4
  )
  
  # ----------------------------------------------------------
  # Generate Poisson quadrature scheme
  # ----------------------------------------------------------
  
  quadrature_poisson <- quadscheme(
    unmark.ppp(species_ppp)
  )
  
  # ----------------------------------------------------------
  # Generate Logistic quadrature scheme
  # ----------------------------------------------------------
  
  quadrature_logistic <- quadscheme.logi(
    unmark.ppp(species_ppp)
  )
  
  # ----------------------------------------------------------
  # Plot quadrature schemes
  # ----------------------------------------------------------
  
  plot(
    quadrature_poisson,
    main = paste(
      "Poisson Quadrature -",
      dataset_name
    )
  )
  
  plot(
    quadrature_logistic,
    main = paste(
      "Logistic Quadrature -",
      dataset_name
    )
  )
  
  # ----------------------------------------------------------
  # Create Poisson dataset
  # ----------------------------------------------------------
  
  poisson_dataset <- create_quadrature_dataset(
    quadrature_scheme = quadrature_poisson,
    covariate_rasters = covariate_rasters
  )
  
  # ----------------------------------------------------------
  # Create Logistic dataset
  # ----------------------------------------------------------
  
  logistic_dataset <- create_quadrature_dataset(
    quadrature_scheme = quadrature_logistic,
    covariate_rasters = covariate_rasters
  )
  
  # ----------------------------------------------------------
  # Save datasets
  # ----------------------------------------------------------
  
  write.csv(
    poisson_dataset,
    config$poisson_output,
    row.names = FALSE
  )
  
  write.csv(
    logistic_dataset,
    config$logistic_output,
    row.names = FALSE
  )
  
  # ----------------------------------------------------------
  # Summary
  # ----------------------------------------------------------
  
  cat("\nSaved Files:\n")
  
  cat(
    "  -",
    config$poisson_output,
    "\n"
  )
  
  cat(
    "  -",
    config$logistic_output,
    "\n"
  )
  
  cat(
    "Poisson Rows :",
    nrow(poisson_dataset),
    "\n"
  )
  
  cat(
    "Logistic Rows:",
    nrow(logistic_dataset),
    "\n"
  )
  
}

# ============================================================
# OPTIONAL:
# DISPLAY AVAILABLE SPECIES
# ============================================================

cat("\n====================================================\n")
cat("Available Species in bci.tree1\n")
cat("====================================================\n")

print(sort(unique(bci.tree1$sp)))

# ============================================================
# OPTIONAL:
# SPECIES FREQUENCY TABLE
# ============================================================

species_frequency <- sort(
  table(bci.tree1$sp),
  decreasing = TRUE
)

print(species_frequency)

# ============================================================
# COMPLETED
# ============================================================

cat("\n====================================================\n")
cat("BCI Preprocessing Completed Successfully\n")
cat("====================================================\n")