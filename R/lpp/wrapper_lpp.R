#' @title Menjalankan Analisis Model ML untuk LPP (Hanya Poisson LogLik)
#' @description Fungsi ini menjalankan analisis "fixed" atau "tuned" untuk data
#' simulasi pada jaringan linear (LPP).
#'
#' @param L Objek 'linnet' yang menjadi dasar simulasi.
#' @param analysis_type String, "fixed" atau "tuned".
#'
run_analysis_LPP <- function(L, model_type, loss_type, sim_data, sim_intensity_linim, sim_points_lpp,
                             base_output_dir, run_number, base_params, analysis_type = "fixed") {
  
  # --- 1. Setup Awal ---
  start_time <- Sys.time()
  model_name_descriptive <- paste0(model_type, "_", analysis_type)
  cat(sprintf("\n--- Menganalisis (LPP): Model = %s, Loss = %s ---\n", toupper(model_name_descriptive), toupper(loss_type)))
  
  output_folder <- file.path(base_output_dir, loss_type)
  viz_folder <- file.path(output_folder, "visualizations")
  if (!dir.exists(viz_folder)) dir.create(viz_folder, recursive = TRUE)
  
  # --- 2. Persiapan Data ---
  sim_data$label <- ifelse(sim_data$label == 1, 1, -1)
  features_cols <- setdiff(names(sim_data), c("x", "y", "label", "vol"))
  sf <- min(max(sim_data$y) - min(sim_data$y),max(sim_data$x) - min(sim_data$x))
  sim_data$vol2 <- sim_data$vol
  sim_data$vol <- sim_data$vol/(sf*sf)
  X <- sim_data[, features_cols]; y <- sim_data$label; vol <- sim_data$vol
  
  # --- 3. Menghitung F_prime ---
  F_prime <- 0
  if (loss_type == "weighted_poisson") {
    k_function <- Knet(sim_points_lpp, lambda = sim_intensity_linim, correction = 'translation')
    rata_rata_jarak <- median(nndist(sim_points_lpp)) 
    
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
    
    X_matrix <- as.matrix(X)
    all_preds <- if (model_type == "xgb") final_model$predict(xgb$DMatrix(X_matrix)) else final_model$predict(X_matrix)
    
    delta <- 1/vol
    # log_likelihood <- sum(all_preds[y == 1]) - sum(exp(all_preds) * vol)
    left_scaled_log_likelihood <- sum(all_preds[y == 1] - log((sf)^2))
    right_scaled_log_likelihood <- sum(exp(all_preds) * vol)
    log_likelihood <- left_scaled_log_likelihood - right_scaled_log_likelihood
    
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
    num_events <- tuning_results$num_events
    
    X_matrix <- as.matrix(X)
    all_preds <- if (model_type == "xgb") final_model$predict(xgb$DMatrix(X_matrix)) else final_model$predict(X_matrix)
    
  } else {
    stop("analysis_type tidak valid. Gunakan 'fixed' atau 'tuned'.")
  }
  
  # --- 5. Membuat Peta Intensitas & Hitung Metrik Tambahan ---
  dummy_points_lpp <- lpp(X = sim_data[sim_data$label == -1, c("x", "y")], L = L)
  dummy_data_matrix <- as.matrix(sim_data[sim_data$label == -1, features_cols])
  prediksi_dummy_ml <- if (model_type == "xgb") final_model$predict(xgb$DMatrix(dummy_data_matrix)) else final_model$predict(dummy_data_matrix)
  
  marks(dummy_points_lpp) <- as.vector(exp(prediksi_dummy_ml)/(sf*sf))
  
  W <- as.owin(L)
  
  pp_quad <- ppp(x = sim_data$x[sim_data$label == -1],
                 y = sim_data$y[sim_data$label == -1],
                 window = W,
                 marks = as.vector(exp(prediksi_dummy_ml)/(sf*sf)))
  im_pred <- Smooth(pp_quad)
  
  #im_pred <- pixellate(pp_quad, weights = marks(pp_quad), dims = c(300,300))
  
  intensity_map_ml_linim <- linim(L, im_pred)   # sekarang Z adalah im -> seharusnya lolos is.im check
  
  num_events_true <- npoints(sim_points_lpp)
  
  squared_error_linim <- (sim_intensity_linim - intensity_map_ml_linim)^2
  total_length_network <- sum(lengths(L))
  mise <- integral(squared_error_linim) / total_length_network
  
  # True Log-Likelihood (Poisson)
  true_log_likelihood <- sum(log(sim_intensity_linim[sim_points_lpp])) - integral(sim_intensity_linim)
  
  # --- PERHITUNGAN TRUE LOGISTIC LOG-LIKELIHOOD DIHAPUS ---
  
  # --- 6. Membuat Laporan (REVISI: Logistic LogLik Dihapus) ---
  computation_time <- Sys.time() - start_time
  results_summary <- data.frame(
    Parameter = c("left Log-Likelihood",
                  "Right Log-Likelihood",
                  "Log-Likelihood",
                  "Left Logistic Log-Lik",
                  "Right Logistic Log-Lik",
                  "Logistic Log-Lik",
                  "Predicted Events", "True Events", "MISE (Network)",
                  "Computation Time (mins)"),
    Value = c(round(left_scaled_log_likelihood, 4),
              round(right_scaled_log_likelihood, 4),
              round(log_likelihood, 4),
              round(left_logistic_log_likelihood, 4),
              round(logistic_log_likelihood, 4),
              round(logistic_log_likelihood, 4),
              round(num_events, 4), num_events_true, round(mise, 8),
              round(as.numeric(computation_time, units = "mins"), 2))
  )
  wb <- createWorkbook(); addWorksheet(wb, "Summary"); writeData(wb, "Summary", results_summary)
  saveWorkbook(wb, file.path(output_folder, "summary_results.xlsx"), overwrite = TRUE)
  
  results_df <- sim_data
  results_df$log_predicted_intensity <- as.vector(log(exp(all_preds)/(sf*sf))) 
  results_df$predicted_intensity <- as.vector(exp(all_preds)/(sf*sf)) 
  results_df$pred <- results_df$predicted_intensity * results_df$vol * (sf*sf)
  
  output_data_path <- file.path(output_folder, "results_dataframe.csv")
  write.csv(results_df, output_data_path, row.names = FALSE)
  
  # --- 7. Membuat Visualisasi (LPP) ---
  
  # 1. Intensity + Points
  sim_plot_title <- sprintf("Simulated Intensity (linim) & Points - Run %d", run_number)
  sim_file_name <- sprintf("simulated_intensity_run_%d.png", run_number)
  png(file.path(output_folder, sim_file_name), width = 800, height = 800, res = 120)
  
  plot(sim_intensity_linim, main = sim_plot_title, col = viridis(100))
  plot(sim_points_lpp, pch = 1, cex = 0.5, col = "white", add = TRUE)
  
  dev.off()
  
  # 2. Points Only
  sim_plot_title_lpp <- sprintf("Simulated Points - Run %d", run_number)
  sim_file_name_lpp <- sprintf("simulated_points_run_%d.png", run_number)
  png(file.path(output_folder, sim_file_name_lpp), width = 800, height = 800, res = 120)
  
  # Note: add = TRUE is REMOVED here because it is a new PNG file
  plot(sim_points_lpp, pch = 1, cex = 0.5, col = "black", main = sim_plot_title_lpp)
  
  dev.off()
  
  # 3. Intensity Only
  sim_plot_title_no_points <- sprintf("Simulated Intensity ONLY (linim) - Run %d", run_number)
  sim_file_name_no_points <- sprintf("simulated_intensity_only_run_%d.png", run_number)
  png(file.path(output_folder, sim_file_name_no_points), width = 800, height = 800, res = 120)
  
  plot(sim_intensity_linim, main = sim_plot_title_no_points, col = viridis(100))
  
  dev.off()

  model_name <- toupper(model_name_descriptive)
  loss_name <- tools::toTitleCase(gsub("_", " ", loss_type))
  ml_plot_title <- sprintf("Intensity %s %s (linim) & Points - Run %d", model_name, loss_name, run_number)
  ml_file_name <- sprintf("intensity_%s_%s_run_%d.png", tolower(model_name_descriptive), tolower(gsub("_", "", loss_type)), run_number)
  png(file.path(viz_folder, ml_file_name), width = 800, height = 800, res = 120)
  plot(intensity_map_ml_linim, main = ml_plot_title, col = viridis(100))
  plot(sim_points_lpp, pch = 1, cex = 0.5, col = "white", add = TRUE)
  dev.off()

  ml_plot_title_no_points <- sprintf("Intensity %s %s ONLY (linim) - Run %d", model_name, loss_name, run_number)
  ml_file_name_no_points <- sprintf("intensity_only_%s_%s_run_%d.png", tolower(model_name_descriptive), tolower(gsub("_", "", loss_type)), run_number)
  png(file.path(viz_folder, ml_file_name_no_points), width = 800, height = 800, res = 120)
  # Perhatikan: baris kedua 'plot(sim_points, ...)' dihilangkan di sini
  plot(intensity_map_ml_linim, main = ml_plot_title_no_points, col = viridis(100))
  dev.off()
  
  cat(sprintf("--- Analisis (LPP) Selesai: %s - %s ---\n", toupper(model_name_descriptive), toupper(loss_type)))
  return(results_summary)
}