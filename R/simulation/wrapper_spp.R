#' @title Menjalankan Analisis Model ML (Versi Final Fleksibel)
#' @description Fungsi ini dapat menjalankan analisis dengan parameter tetap ("fixed")
#' atau dengan tuning ("tuned"). Ia juga menangani perbedaan input antara
#' LightGBM dan XGBoost.
#'
#' @param analysis_type String, "fixed" atau "tuned". Menentukan skenario analisis.

run_analysis <- function(model_type, loss_type, sim_data, sim_intensity, sim_points, 
                         base_output_dir, run_number, base_params, analysis_type = "fixed",
                         scale_factor = scale_factor) {
  
  # --- 1. Setup Awal ---
  start_time <- Sys.time()
  model_name_descriptive <- paste0(model_type, "_", analysis_type)
  cat(sprintf("\n--- Menganalisis: Model = %s, Loss = %s ---\n", toupper(model_name_descriptive), toupper(loss_type)))
  
  output_folder <- file.path(base_output_dir, model_name_descriptive, loss_type)
  viz_folder <- file.path(output_folder, "visualizations")
  if (!dir.exists(viz_folder)) dir.create(viz_folder, recursive = TRUE)
  
  # --- 2. Persiapan Data ---
  sim_data$label <- ifelse(sim_data$label == 1, 1, -1)
  features_cols <- setdiff(names(sim_data), c("x", "y", "label", "vol"))
  X <- sim_data[, features_cols]; y <- sim_data$label; vol <- sim_data$vol
  
  # --- 3. Menghitung F_prime (jika perlu) ---
  F_prime <- 0
  if (loss_type == "weighted_poisson") {
    k_function <- Kinhom(sim_points, lambda = sim_intensity, correction = 'translation')
    rata_rata_jarak <- median(nndist(sim_points))
    F_prime <- with(k_function[which.min(abs(k_function$r - rata_rata_jarak)), ], trans - theo)
    if (is.na(F_prime) || is.infinite(F_prime) || F_prime < 0) F_prime <- 0
  }
  
  # --- 4. Memanggil & Melatih Model ---
  if (analysis_type == "fixed") {
    if (model_type == "lgb") {
      final_model <- train_lgbpp_fixed(X, y, vol, loss_type, F_prime, base_params = base_params)
    } else { # xgb
      final_model <- train_xgbpp_fixed(X, y, vol, loss_type, F_prime, base_params = base_params)
    }
    # Hitung metrik secara manual
    X_matrix <- X
    all_preds <- if (model_type == "xgb") final_model$predict(xgb$DMatrix(X_matrix)) else final_model$predict(as.matrix(X_matrix))
    delta <- 1/vol
    log_likelihood <- sum(all_preds[y == 1]) - sum(exp(all_preds) * vol)
    left_scaled_log_likelihood <- sum(all_preds[y == 1] - log((scale_factor)^2))
    right_scaled_log_likelihood <- sum(exp(all_preds) * vol)
    scaled_log_likelihood <- left_scaled_log_likelihood - right_scaled_log_likelihood
    
    # Baddeley
    #logistic_log_likelihood <- sum(log(exp(all_preds[y==1])/(exp(all_preds[y==1]) + delta[y==1]))) + sum(log(delta[y == -1]/(exp(all_preds[y == -1]) + delta[y == -1])))
    
    left_logistic_log_likelihood <- sum((log(exp(all_preds[y == 1])/(delta[y == 1] + exp(all_preds[y == 1])))))
    right_logistic_log_likelihood <- sum((log(delta[y == -1]/(delta[y == -1] + exp(all_preds[y == -1])))))
    logistic_log_likelihood <- left_logistic_log_likelihood + right_logistic_log_likelihood
    num_events <- sum(exp(all_preds) * vol)
    
  } else if (analysis_type == "tuned") {
    cat(sprintf("--- Memanggil TUNER Python untuk %s...\n", toupper(model_type)))
    if (model_type == "lgb") {
      tuning_results <- tune_lgbpp(X_df = X, y_series = y, vol_series = vol, loss = loss_type, F_prime = F_prime, constrain_events = TRUE, n_trials=20L)
    } else { # xgb
      tuning_results <- tune_xgbpp(X_df = X, y_series = y, vol_series = vol, loss = loss_type, F_prime = F_prime, constrain_events = TRUE, n_trials=20L)
    }
    final_model <- tuning_results$final_model
    log_likelihood <- tuning_results$best_log_likelihood
    left_scaled_log_likelihood <- 0
    right_scaled_log_likelihood <- 0
    scaled_log_likelihood <- 0
    logistic_log_likelihood <- 0
    left_logistic_log_likelihood <- 0
    right_logistic_log_likelihood <- 0
    num_events <- tuning_results$num_events
  } else {
    stop("analysis_type tidak valid. Gunakan 'fixed' atau 'tuned'.")
  }
  
  # --- 5. Membuat Peta Intensitas & Hitung Metrik Tambahan ---
  
  # Membuat peta intensitas prediksi ML
  dummy_ppp <- ppp(x = sim_data$x[sim_data$label == -1], y = sim_data$y[sim_data$label == -1], window = Window(sim_points))
  dummy_data_df <- sim_data[sim_data$label == -1, features_cols]
  prediksi_dummy_ml <- if (model_type == "xgb") final_model$predict(xgb$DMatrix(dummy_data_df)) else final_model$predict(as.matrix(dummy_data_df))
  
  num_events_true <- npoints(sim_points)
  
  marks(dummy_ppp) <- as.vector(exp(prediksi_dummy_ml))
  intensity_map_ml <- Smooth(dummy_ppp)
  
  # Metrik Baru: MISE (Mean Integrated Squared Error)
  intensity_map_ml_compat <- as.im(intensity_map_ml, W = sim_points$window)
  squared_error_im <- (sim_intensity - intensity_map_ml_compat)^2
  squared_error_im_scaled <- (sim_intensity/(scale_factor)^2 - intensity_map_ml_compat/(scale_factor)^2)^2
  mise <- integral(squared_error_im)/area.owin(Window(sim_points))
  mise_scaled <- integral(squared_error_im_scaled)/area.owin(Window(sim_points))
  
  # True Log-Lik
  true_log_likelihood <- sum(log(sim_intensity[sim_points])) - integral(sim_intensity)
  left_scaled_true_log_likelihood <- sum(log(sim_intensity[sim_points]/(scale_factor)^2))
  right_scaled_true_log_likelihood <- integral(sim_intensity)
  scaled_true_log_likelihood <- left_scaled_true_log_likelihood - right_scaled_true_log_likelihood
  
  ### Logistic
  #true_rho_all <- sim_intensity[ppp(sim_data$x, sim_data$y, window = Window(sim_points))]
  #delta <- 1 / sim_data$vol
  #loglik_events_part <- log(true_rho_all[y == 1] / (true_rho_all[y == 1] + delta[y == 1]))
  #loglik_dummy_part <- log(delta[y == -1] / (true_rho_all[y == -1] + delta[y == -1]))
  #true_logistic_log_likelihood <- sum(loglik_events_part) + sum(loglik_dummy_part)
  
  true_rho_all <- sim_intensity[ppp(sim_data$x, sim_data$y, window = Window(sim_points))]
  delta <- 1 / sim_data$vol
  vol <- sim_data$vol
  
  left_true_logistic_log_likelihood <- sum(log(true_rho_all[y == 1] / (delta[y == 1] + true_rho_all[y == 1])))
  right_true_logistic_log_likelihood <- sum((delta[y == -1] * log((true_rho_all[y == -1] + delta[y == -1]) / delta[y == -1])) * vol[y == -1])
  true_logistic_log_likelihood <-  left_true_logistic_log_likelihood - right_true_logistic_log_likelihood
  
  # --- 6. Membuat Laporan ---
  computation_time <- Sys.time() - start_time
  results_summary <- data.frame(
    Parameter = c("True Log-Likelihood", "Scaled True Log-Likelihood", "Left Scaled True Log-Likelihood", "Right Scaled True Log-Likelihood",
                  "True Logistic Log-Likelihood", "Left True Logistic Log-Likelihood", "Right True Logistic Log-Likelihood", "Log-Likelihood",
                  "Scaled Log-Likelihood", "Left Scaled Log-Likelihood", "Right Scaled Log-Likelihood", "Left Logistic Log-Likelihood",
                  "Right Logistic Log-Likelihood", "Logistic Log-Likelihood", "Predicted Events", "True Events", "MISE",
                  "Scaled MISE", "Computation Time (mins)"),
    Value = c(round(true_log_likelihood, 4), round(scaled_true_log_likelihood, 4), round(left_scaled_true_log_likelihood, 4),round(right_scaled_true_log_likelihood, 4),
              round(true_logistic_log_likelihood,4), round(left_true_logistic_log_likelihood, 4), round(right_true_logistic_log_likelihood, 4), round(log_likelihood, 4),
              round(scaled_log_likelihood, 4), round(left_scaled_log_likelihood, 4), round(right_scaled_log_likelihood, 4), round(left_logistic_log_likelihood, 4),
              round(right_logistic_log_likelihood, 4), round(logistic_log_likelihood, 4), round(num_events, 4), num_events_true, round(mise, 4),
              round(mise_scaled,8), round(as.numeric(computation_time, units = "mins"), 2))
  )
  wb <- createWorkbook(); addWorksheet(wb, "Summary"); writeData(wb, "Summary", results_summary)
  saveWorkbook(wb, file.path(output_folder, "summary_results.xlsx"), overwrite = TRUE)
  
  results_df <- sim_data
  
  results_df$log_predicted_intensity <- as.vector(exp(all_preds))
  results_df$predicted_intensity <- as.vector(exp(all_preds))
  results_df$pred <- results_df$predicted_intensity * results_df$vol
  
  output_data_path <- file.path(output_folder, "results_dataframe.csv")
  write.csv(results_df, output_data_path, row.names = FALSE)
  
  # --- 6. Membuat Visualisasi ---
  # --- PLOT 1: Intensitas simulasi ASLI (DENGAN overlay titik) ---
  sim_plot_title <- sprintf("Simulated Intensity & Points - Run %d", run_number)
  sim_file_name <- sprintf("simulated_intensity_run_%d.png", run_number)
  png(file.path(output_folder, sim_file_name), width = 800, height = 800, res = 120)
  plot(sim_intensity, main = sim_plot_title, col = viridis(100))
  points(sim_points, pch = 1, cex = 0.5, col = "white")
  dev.off()
  
  # --- TAMBAHAN: PLOT INTENSITAS SIMULASI (TANPA overlay titik) ---
  sim_plot_title_no_points <- sprintf("Simulated Intensity ONLY - Run %d", run_number)
  sim_file_name_no_points <- sprintf("simulated_intensity_only_run_%d.png", run_number)
  png(file.path(output_folder, sim_file_name_no_points), width = 800, height = 800, res = 120)
  # Perhatikan: baris 'points(...)' dihilangkan di sini
  plot(sim_intensity, main = sim_plot_title_no_points, col = viridis(100))
  dev.off()
  
  # --- TAMBAHAN: PLOT INTENSITAS SIMULASI (TANPA overlay titik) ---
  sim_plot_title_no_points <- sprintf("Simulated Intensity ONLY BACKED - Run %d", run_number)
  sim_file_name_no_points <- sprintf("simulated_intensity_only_backed_run_%d.png", run_number)
  png(file.path(output_folder, sim_file_name_no_points), width = 800, height = 800, res = 120)
  # Perhatikan: baris 'points(...)' dihilangkan di sini
  plot(sim_intensity/(scale_factor)^2, main = sim_plot_title_no_points, col = viridis(100))
  dev.off()
  
  # --------------------------------------------------------------------------
  
  # Menghitung prediksi intensitas dari model ML
  dummy_ppp <- ppp(x = sim_data$x[sim_data$label == -1], y = sim_data$y[sim_data$label == -1], window = Window(sim_points))
  dummy_data_matrix <- sim_data[sim_data$label == -1, features_cols]
  prediksi_dummy_ml <- if (model_type == "xgb") final_model$predict(xgb$DMatrix(dummy_data_matrix)) else final_model$predict(as.matrix(dummy_data_matrix))
  
  marks(dummy_ppp) <- as.vector(exp(prediksi_dummy_ml))
  intensity_map_ml <- Smooth(dummy_ppp)
  
  # --- PLOT 2: Intensitas prediksi ML (DENGAN overlay titik) ---
  model_name <- toupper(model_name_descriptive)
  loss_name <- tools::toTitleCase(gsub("_", " ", loss_type))
  ml_plot_title <- sprintf("Intensity %s %s & Points - Run %d", model_name, loss_name, run_number)
  ml_file_name <- sprintf("intensity_%s_%s_run_%d.png", tolower(model_name_descriptive), tolower(gsub("_", "", loss_type)), run_number)
  png(file.path(viz_folder, ml_file_name), width = 800, height = 800, res = 120)
  plot(intensity_map_ml, main = ml_plot_title, col = viridis(100))
  plot(sim_points, pch = 1, cex = 0.5, col = "white", add = TRUE)
  dev.off()
  
  # --- TAMBAHAN: PLOT INTENSITAS PREDIKSI ML (TANPA overlay titik) ---
  ml_plot_title_no_points <- sprintf("Intensity %s %s ONLY - Run %d", model_name, loss_name, run_number)
  ml_file_name_no_points <- sprintf("intensity_only_%s_%s_run_%d.png", tolower(model_name_descriptive), tolower(gsub("_", "", loss_type)), run_number)
  png(file.path(viz_folder, ml_file_name_no_points), width = 800, height = 800, res = 120)
  # Perhatikan: baris kedua 'plot(sim_points, ...)' dihilangkan di sini
  plot(intensity_map_ml, main = ml_plot_title_no_points, col = viridis(100))
  dev.off()
  
  # --- TAMBAHAN: PLOT INTENSITAS PREDIKSI ML (TANPA overlay titik) ---
  ml_plot_title_no_points <- sprintf("Intensity %s %s ONLY BACKED - Run %d", model_name, loss_name, run_number)
  ml_file_name_no_points <- sprintf("intensity_only_backed_%s_%s_run_%d.png", tolower(model_name_descriptive), tolower(gsub("_", "", loss_type)), run_number)
  png(file.path(viz_folder, ml_file_name_no_points), width = 800, height = 800, res = 120)
  # Perhatikan: baris kedua 'plot(sim_points, ...)' dihilangkan di sini
  plot(intensity_map_ml/(scale_factor)^2, main = ml_plot_title_no_points, col = viridis(100))
  dev.off()
  
  cat(sprintf("--- Analisis Selesai: %s - %s ---\n", toupper(model_name_descriptive), toupper(loss_type)))
  return(results_summary)
}