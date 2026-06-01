# ============================================================
# XGBoostPP Universal Pipeline
# ============================================================
#
# Author  : Your Name
# Purpose : Universal XGBoostPP workflow for multiple datasets
#
# Supported Datasets:
#   1. BCI Bei
#   2. BCI Ryno
#   3. Crime
#   4. Traffic Accident
#
# ============================================================
# RETICULATE SETUP
# ============================================================
#
# IMPORTANT:
# Configure your Conda environment before running this script.
#
# Recommended:
#   reticulate::use_condaenv("xgb-env", required = TRUE)
#
# Documentation:
#   https://rstudio.github.io/reticulate/
#
# ============================================================

# ============================================================
# LOAD LIBRARIES
# ============================================================

required_packages <- c(
  "reticulate",
  "dplyr",
  "spatstat.geom",
  "spatstat.explore",
  "spatstat.linnet",
  "spatstat",
  "xgboost",
  "lightgbm",
  "ggplot2",
  "viridis",
  "RColorBrewer",
  "tictoc"
)

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# ============================================================
# RETICULATE CONFIGURATION
# ============================================================

library(reticulate)

use_condaenv("xgb-env", required = TRUE)

cat("\n====================================================\n")
cat("Python Configuration\n")
cat("====================================================\n")

py_config()

# ============================================================
# RANDOM SEED
# ============================================================

set.seed(1234)

# ============================================================
# LOAD PYTHON MODULES
# ============================================================

np <- import("numpy", convert = TRUE)
xgb_py <- import("xgboost", convert = TRUE)

# ============================================================
# SOURCE PYTHON SCRIPT
# ============================================================

python_script <- "python/xgbpp-revised1.py"

if (!file.exists(python_script)) {
  stop("Python script not found.")
}

source_python(python_script)

# ============================================================
# DATASET CONFIGURATION
# ============================================================

dataset_name <- "crime"

# Available options:
# "bei_poisson"
# "bei_logistic"
# "ryno_poisson"
# "ryno_logistic"
# "crime_poisson"
# "crime_logistic"
# "accident_poisson"
# "accident_logistic"

dataset_configs <- list(
  
  bei_poisson = list(
    file = "data/bei/beinonscale.csv",
    covariates = c(
      "Cu","grad","elev","Nmin",
      "P","pH","solar","twi"
    ),
    window_type = "bei"
  ),
  
  bei_logistic = list(
    file = "data/bei/beiloginonscale2.csv",
    covariates = c(
      "Cu","grad","elev","Nmin",
      "P","pH","solar","twi"
    ),
    window_type = "bei"
  ),
  
  ryno_poisson = list(
    file = "data/ryno/rynononscale.csv",
    covariates = c(
      "Cu","grad","elev","Nmin",
      "P","pH","solar","twi"
    ),
    window_type = "bei"
  ),
  
  ryno_logistic = list(
    file = "data/ryno/rynologinonscale2.csv",
    covariates = c(
      "Cu","grad","elev","Nmin",
      "P","pH","solar","twi"
    ),
    window_type = "bei"
  ),
  
  crime_poisson = list(
    file = "data/crime/crimenonscale.csv",
    covariates = c(
      "transport","injuries","construction",
      "water","pharmacies","schools",
      "health","parks","markets","libraries"
    ),
    window_type = "crime"
  ),
  
  crime_logistic = list(
    file = "data/crime/crimeloginonscale.csv",
    covariates = c(
      "transport","injuries","construction",
      "water","pharmacies","schools",
      "health","parks","markets","libraries"
    ),
    window_type = "crime"
  ),
  
  accident_poisson = list(
    file = "data/accident/linnetdata.csv",
    covariates = c(
      "Jarak_Lampu",
      "n_Lajur",
      "n_Jalur",
      "Kerb_R",
      "Kerb_L"
    ),
    window_type = "linnet"
  ),
  
  accident_logistic = list(
    file = "data/accident/linnetdatalogi.csv",
    covariates = c(
      "Jarak_Lampu",
      "n_Lajur",
      "n_Jalur",
      "Kerb_R",
      "Kerb_L"
    ),
    window_type = "linnet"
  )
)

config <- dataset_configs[[dataset_name]]

# ============================================================
# LOAD REQUIRED SPATIAL OBJECTS
# ============================================================

# Example:
# load("data/CrimeDataKennedy.RData")
# load("data/bei.RData")
# load("data/nganjuk_ln.RData")

# ============================================================
# LOAD DATASET
# ============================================================

data_df <- read.csv(config$file)

feature_columns <- config$covariates

# ============================================================
# SPLIT EVENT AND DUMMY POINTS
# ============================================================

event_data <- data_df %>%
  filter(label == 1)

dummy_data <- data_df %>%
  filter(label == -1)

# ============================================================
# PREPARE FEATURES
# ============================================================

feature_matrix <- data_df[, feature_columns]

labels <- data_df$label

quadrature_weight <- data_df$vol

# ============================================================
# SCALE FACTOR
# ============================================================

scale_factor <- min(
  max(data_df$y) - min(data_df$y),
  max(data_df$x) - min(data_df$x)
)

# ============================================================
# SELECT OBSERVATION WINDOW
# ============================================================

if (config$window_type == "bei") {
  
  observation_window <- bei$window
  
} else if (config$window_type == "crime") {
  
  observation_window <- CrimesPP$window
  
} else if (config$window_type == "linnet") {
  
  observation_window <- as.owin(nganjuk_ln)
  
}

# ============================================================
# CREATE EVENT POINT PATTERN
# ============================================================

event_ppp <- ppp(
  x = event_data$x,
  y = event_data$y,
  window = observation_window,
  check = FALSE
)

event_ppp <- rescale(event_ppp, s = scale_factor)

# ============================================================
# COMPUTE K-INHOM
# ============================================================

k_function <- Kinhom(
  event_ppp,
  correction = "translation"
)

# ============================================================
# COMPUTE NEAREST NEIGHBOR DISTANCE
# ============================================================

median_distance <- median(
  apply(
    as.matrix(
      dist(scale(event_data[, c("x", "y")]))
    ),
    1,
    function(row) min(row[row > 0])
  )
)

# ============================================================
# COMPUTE F PRIME
# ============================================================

F_prime <- with(
  k_function[
    which.min(abs(k_function$r - median_distance)),
  ],
  trans - theo
)

# Optional:
F_prime <- 0

lambda_values <- 0

# ============================================================
# CONVERT DATA TO PYTHON
# ============================================================

X_py <- np$array(
  as.matrix(feature_matrix),
  dtype = "float64"
)

y_py <- np$array(
  as.numeric(labels),
  dtype = "float64"
)

vol_py <- np$array(
  as.numeric(quadrature_weight) / (scale_factor^2),
  dtype = "float64"
)

F_prime_py <- np$array(
  as.numeric(F_prime),
  dtype = "float64"
)

lambda_py <- np$array(
  as.numeric(lambda_values),
  dtype = "float64"
)

# ============================================================
# CREATE DMatrix
# ============================================================

dtrain_py <- xgb_py$DMatrix(
  data = X_py,
  label = y_py
)

dpred_py <- xgb_py$DMatrix(
  data = as.matrix(feature_matrix)
)

# ============================================================
# LOG-LIKELIHOOD FUNCTIONS
# ============================================================

compute_poisson_loglik <- function(
    model,
    dtrain_py,
    dpred_py,
    vol_py,
    scale_factor = 500
) {
  
  all_preds <- model$predict(dpred_py)
  
  labels_py <- np$ravel(
    dtrain_py$get_label()
  )
  
  event_mask <- labels_py > 0
  
  all_preds_clipped <- np$clip(
    all_preds,
    -50,
    50
  )
  
  y_pred_event <- all_preds_clipped[event_mask]
  
  left <- sum(
    y_pred_event - log(scale_factor^2)
  )
  
  right <- sum(
    exp(all_preds_clipped) * vol_py
  )
  
  log_likelihood <- left - right
  
  cat(
    "Poisson log-likelihood:",
    as.numeric(log_likelihood),
    "\n"
  )
  
  return(as.numeric(log_likelihood))
}

compute_logistic_loglik <- function(
    model,
    dtrain_py,
    dpred_py,
    vol_py
) {
  
  all_preds <- model$predict(dpred_py)
  
  labels_py <- np$ravel(
    dtrain_py$get_label()
  )
  
  event_mask <- labels_py > 0
  
  dummy_mask <- labels_py < 0
  
  y_pred_event <- np$clip(
    all_preds[event_mask],
    -50,
    50
  )
  
  y_pred_dummy <- np$clip(
    all_preds[dummy_mask],
    -50,
    50
  )
  
  vol_event <- vol_py[event_mask]
  
  vol_dummy <- vol_py[dummy_mask]
  
  delta_event <- 1.0 / vol_event
  
  delta_dummy <- 1.0 / vol_dummy
  
  left_val <- sum(
    log(
      exp(y_pred_event) /
        (exp(y_pred_event) + delta_event)
    )
  )
  
  right_val <- sum(
    log(
      delta_dummy /
        (exp(y_pred_dummy) + delta_dummy)
    )
  )
  
  log_likelihood <- left_val + right_val
  
  cat(
    "Logistic log-likelihood:",
    as.numeric(log_likelihood),
    "\n"
  )
  
  return(as.numeric(log_likelihood))
}

# ============================================================
# CREATE XGBOOST PARAMETERS
# ============================================================

create_xgb_params <- function(
    eta = 0.01,
    lambda = 10,
    max_depth = 8
) {
  
  reticulate::dict(
    booster = "gbtree",
    subsample = 0.8,
    colsample_bytree = 1/3,
    nthread = as.integer(-1),
    tree_method = "exact",
    verbosity = as.integer(0),
    eta = eta,
    alpha = 0,
    lambda = lambda,
    max_depth = as.integer(max_depth)
  )
}

# ============================================================
# GRID SEARCH CONFIGURATION
# ============================================================

etas <- c(0.1, 0.01, 0.001)

lambdas <- c(0, 10, 30, 50)

loss_type <- "weighted_logistic"

# Available:
# "poisson"
# "weighted_poisson"
# "logistic"
# "weighted_logistic"

# ============================================================
# GRID SEARCH
# ============================================================

tuning_results <- data.frame(
  eta = numeric(),
  lambda = numeric(),
  poisson_loglik = numeric(),
  logistic_loglik = numeric(),
  stringsAsFactors = FALSE
)

tuple <- reticulate::tuple

for (eta_value in etas) {
  
  for (lambda_value in lambdas) {
    
    cat("\n====================================================\n")
    cat(
      "Testing eta =",
      eta_value,
      ", lambda =",
      lambda_value,
      "\n"
    )
    cat("====================================================\n")
    
    params_py <- create_xgb_params(
      eta = eta_value,
      lambda = lambda_value
    )
    
    xgbpp_model <- xgbpp_py(
      dtrain = dtrain_py,
      vol = vol_py,
      params = params_py,
      loss = loss_type,
      F_prime = F_prime_py,
      lambdau = lambda_py,
      evals = list(
        tuple(dtrain_py, "train")
      ),
      num_boost_round = as.integer(5000),
      early_stopping_rounds = as.integer(50),
      verbose_eval = FALSE
    )
    
    poisson_loglik <- compute_poisson_loglik(
      xgbpp_model,
      dtrain_py,
      dpred_py,
      vol_py,
      scale_factor
    )
    
    logistic_loglik <- compute_logistic_loglik(
      xgbpp_model,
      dtrain_py,
      dpred_py,
      vol_py
    )
    
    tuning_results <- rbind(
      tuning_results,
      data.frame(
        eta = eta_value,
        lambda = lambda_value,
        poisson_loglik = poisson_loglik,
        logistic_loglik = logistic_loglik
      )
    )
  }
}

# ============================================================
# BEST PARAMETERS
# ============================================================

best_result <- tuning_results[
  which.max(tuning_results$logistic_loglik),
]

cat("\n====================================================\n")
cat("Best Parameter Combination\n")
cat("====================================================\n")

print(best_result)

# ============================================================
# FINAL MODEL TRAINING
# ============================================================

final_params <- create_xgb_params(
  eta = best_result$eta,
  lambda = best_result$lambda
)

cat("\n====================================================\n")
cat("Training Final Model\n")
cat("====================================================\n")

tic()

final_model <- xgbpp_py(
  dtrain = dtrain_py,
  vol = vol_py,
  params = final_params,
  loss = loss_type,
  F_prime = F_prime_py,
  lambdau = lambda_py,
  evals = list(
    tuple(dtrain_py, "train")
  ),
  num_boost_round = as.integer(5000),
  early_stopping_rounds = as.integer(50),
  verbose_eval = FALSE
)

runtime <- toc(quiet = TRUE)

# ============================================================
# FINAL EVALUATION
# ============================================================

final_poisson_loglik <- compute_poisson_loglik(
  final_model,
  dtrain_py,
  dpred_py,
  vol_py,
  scale_factor
)

final_logistic_loglik <- compute_logistic_loglik(
  final_model,
  dtrain_py,
  dpred_py,
  vol_py
)

# ============================================================
# SAVE GRID SEARCH RESULTS
# ============================================================

dir.create("output", showWarnings = FALSE)
dir.create("output/tables", showWarnings = FALSE)
dir.create("output/models", showWarnings = FALSE)
dir.create("output/figures", showWarnings = FALSE)

write.csv(
  tuning_results,
  file.path(
    "output/tables",
    paste0(dataset_name, "_grid_search.csv")
  ),
  row.names = FALSE
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df <- py$get_feature_importance_xgbpp(
  final_model,
  feature_names = feature_columns,
  importance_type = "gain"
)

importance_df_R <- py_to_r(importance_df)

importance_df_R$importance <- (
  importance_df_R$importance /
    sum(importance_df_R$importance)
)

# ============================================================
# FEATURE IMPORTANCE PLOT
# ============================================================

importance_plot <- ggplot(
  head(importance_df_R, 10),
  aes(
    x = reorder(feature, importance),
    y = importance
  )
) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = paste(
      "Feature Importance -",
      dataset_name
    ),
    x = "Feature",
    y = "Importance"
  )

print(importance_plot)

ggsave(
  filename = file.path(
    "output/figures",
    paste0(dataset_name, "_importance.png")
  ),
  plot = importance_plot,
  width = 7,
  height = 5
)

# ============================================================
# PREDICTION
# ============================================================

all_predictions <- final_model$predict(dpred_py)

labels_py <- np$ravel(
  dtrain_py$get_label()
)

dummy_mask <- labels_py < 0

predicted_dummy <- all_predictions[dummy_mask]

predicted_dummy <- log(
  exp(predicted_dummy) /
    (scale_factor^2)
)

# ============================================================
# SPATIAL VISUALIZATION
# ============================================================

if (config$window_type != "linnet") {
  
  dummy_ppp <- ppp(
    x = data_df$x[data_df$label == -1],
    y = data_df$y[data_df$label == -1],
    window = observation_window,
    marks = as.vector(predicted_dummy)
  )
  
  intensity_map <- Smooth(dummy_ppp)
  
  png(
    filename = file.path(
      "output/figures",
      paste0(dataset_name, "_intensity.png")
    ),
    width = 900,
    height = 700
  )
  
  plot(
    intensity_map,
    col = viridis::viridis(
      100,
      option = "plasma"
    ),
    ribbon = TRUE,
    ribbon.pos = "bottom",
    asp = 1,
    main = paste(
      "Estimated Intensity -",
      dataset_name
    )
  )
  
  points(
    event_data$x,
    event_data$y,
    pch = 19,
    cex = 0.2,
    col = "black"
  )
  
  dev.off()
  
}

# ============================================================
# SAVE MODEL
# ============================================================

saveRDS(
  final_model,
  file.path(
    "output/models",
    paste0(dataset_name, "_xgbpp_model.rds")
  )
)

# ============================================================
# SAVE FEATURE IMPORTANCE TABLE
# ============================================================

write.csv(
  importance_df_R,
  file.path(
    "output/tables",
    paste0(dataset_name, "_feature_importance.csv")
  ),
  row.names = FALSE
)

# ============================================================
# FINAL SUMMARY
# ============================================================

cat("\n====================================================\n")
cat("Analysis Completed Successfully\n")
cat("====================================================\n")

cat("Dataset          :", dataset_name, "\n")
cat("Loss Function    :", loss_type, "\n")
cat("Poisson LogLik   :", final_poisson_loglik, "\n")
cat("Logistic LogLik  :", final_logistic_loglik, "\n")
cat("Runtime (sec)    :", runtime$toc - runtime$tic, "\n")

cat("\nOutputs saved in:\n")
cat("  output/models/\n")
cat("  output/tables/\n")
cat("  output/figures/\n")