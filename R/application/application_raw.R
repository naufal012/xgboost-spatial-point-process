library(viridis)
library(RColorBrewer)
library(spatstat.linnet)

# Bersihkan Python yang sudah pernah diinisialisasi (kalau ada)
if ("package:reticulate" %in% search()) {
  detach("package:reticulate", unload = TRUE, character.only = TRUE)
}

if ("reticulate" %in% loadedNamespaces()) {
  unloadNamespace("reticulate")
}


# Muat ulang reticulate
library(reticulate)

# 1. Tentukan nama environment Anda
conda_env_name <- "xgb-env"

# 2. Path ke conda.exe (sesuaikan dengan instalasi Anda)
conda_executable_path <- "C:/Users/Akmal/anaconda3/Scripts/conda.exe"

# 3. Pastikan langsung pakai condaenv sebelum Python lain ter-load
tryCatch({
  use_condaenv(
    condaenv = conda_env_name,
    conda = conda_executable_path,
    required = TRUE
  )
}, error = function(e) {
  stop(paste(
    "Gagal terhubung ke Conda. Pastikan:\n",
    "1. Nama environment '", conda_env_name, "' sudah benar.\n",
    "2. Path ke conda.exe '", conda_executable_path, "' sudah benar.",
    sep = ""
  ))
})

# 4. Verifikasi konfigurasi Python
cat("--- Konfigurasi Python yang digunakan oleh Reticulate ---\n")
py_config()


# =============================================================
# BAGIAN 0: SETUP R DAN RETICULATE
# =============================================================

# Install dan muat pustaka yang diperlukan
if (!require("reticulate")) install.packages("reticulate")
if (!require("dplyr")) install.packages("dplyr")

library(reticulate)
library(dplyr)
library(spatstat.explore)
#reticulate::py_install("xgboost")
xgb_py <- reticulate::import("xgboost")
#reticulate::py_install("lightgbm")
#lgb_py <- reticulate::import("lightgbm")

# --- PENTING: Arahkan reticulate ke Conda Environment Anda ---
# Ini adalah cara yang paling andal untuk menghindari error.

# 1. Tentukan nama environment Anda
conda_env_name <- "xgb-env" 

# 2. Tentukan path ke file eksekusi conda.exe di dalam instalasi Anaconda Anda
# Path ini biasanya benar untuk instalasi standar di Windows.
conda_executable_path <- "C:/Users/Akmal/anaconda3/Scripts/conda.exe"

tryCatch({
  # Gunakan use_condaenv dengan path eksplisit ke conda.exe
  use_condaenv(
    condaenv = conda_env_name,
    conda = conda_executable_path,
    required = TRUE
  )
}, error = function(e) {
  stop(paste("Gagal terhubung ke Conda. Pastikan:\n",
             "1. Nama environment '", conda_env_name, "' sudah benar.\n",
             "2. Path ke conda.exe '", conda_executable_path, "' sudah benar.", sep=""))
})


# Verifikasi konfigurasi Python untuk memastikan sudah benar
cat("--- Konfigurasi Python yang digunakan oleh Reticulate ---\n")
py_config()

setwd("C:/Users/Akmal/Documents/ITS/TA/xgbnew") # <-- GANTI PATH INI
gempapp = load("datatesistabita/gempa_pp_tesis.Rda")
gempaqs = load("datatesistabita/gempa_qs_tesis.Rda")
load("CrimeDataKennedy.RData")

library(spatstat.geom)


#df <- read.csv("beiloginonscale2.csv")
#df <- read.csv("crimenonscale.csv")
df <- read.csv("linnetdatalogi.csv")
#df <- read.csv("gempanonscale10thn.csv")

#features_cols <- c("transport","injuries","construction","water","pharmacies","schools","health","parks","markets","libraries")
#features_cols <- c("Cu","grad" ,"elev" ,"Nmin" ,"P","pH","solar", "twi")
#features_cols <- c("depth","dip","sesar","strike","megathrust","volcano")
features_cols <- c('Jarak_Lampu', 'n_Lajur', 'n_Jalur', 'Kerb_R', 'Kerb_L')

#plot(CrimesPP,pch=19, cex=0.35)


truedata <- df %>%
  filter(label == 1)

dummydata <- df %>%
  filter(label == -1)

# Memisahkan data
X <- df[, features_cols]
y <- df$label
vol <- df$vol
sf <- min(max(df$y) - min(df$y),max(df$x) - min(df$x));sf
#win_left <- bei$window
#win_left  <- spatstat.geom::shift(gempaSM$window, vec = c(-11.3, 0))
win_left <- CrimesPP$window
xs <- ppp(x = truedata$x,
          y = truedata$y,
          window = win_left, check = FALSE)
plot(xs,pch=19,cex=0.35,main="Crime")

xs = rescale(xs,s=sf)
library(spatstat)

#lambda_est <- density.ppp(xs, sigma = bw.diggle(xs))
#lambda_values <- lambda_est[ xs ]
#sigma=0.4

#str(full_df)
k_function <- Kinhom(xs,
                     #r=seq(0,sigma,0.005),
                     #lambda = as.vector(lambda_values),
                     correction = 'translation')
rata_rata_jarak <- median(apply(
   as.matrix(dist(scale(truedata[, c("x", "y")]))),
   1,
   function(baris) min(baris[baris > 0])
 ))
plot(k_function)
#plot(k_function$trans-k_function$theo)
#k_function
#rata_rata_jarak <- median(nndist(truedata[, c("x", "y")]))
F_prime <- with(k_function[which.min(abs(k_function$r - rata_rata_jarak)), ], trans - theo);F_prime
#F_prime <- k_function$trans[2] - k_function$theo[2];F_prime

# 
datappp <- ppp(x = df$x,
               y = df$y,
               window = win_left, check = FALSE)
datappp = rescale(datappp,s=sf)

dens <- density.ppp(datappp,sigma=bw.diggle(datappp), at='points')
lambda_values <- dens
#lambda_values <- dens[datappp]
F_prime <- lambda_values*F_prime
# we <- 1/1+F_prime

range(F_prime)
F_prime=0
lambda_values=0
py_script_path <- "xgbpp-revised1.py" 
if (!file.exists(py_script_path)) {
  stop(paste("File Python '", py_script_path, "' tidak ditemukan.", sep=""))
}
source_python(py_script_path)

library(xgboost)
library(lightgbm)
library(tictoc)
tuple <- reticulate::tuple
# Import numpy dari Python
np <- import("numpy", convert = TRUE)
xgb_py <- import("xgboost", convert = TRUE)
#lgb_py <- import("lightgbm", convert = TRUE)

X_py   <- np$array(as.matrix(X), dtype = "float64")
y_py   <- np$array(as.numeric(y), dtype = "float64")
vol_py <- np$array(as.numeric(vol)/(sf*sf), dtype = "float64")
F_prime_py <- np$array(as.numeric(F_prime), dtype = "float64")
lambda_py <- np$array(as.numeric(lambda_values), dtype = "float64")

dtrain_py <- xgb_py$DMatrix(data = X_py, label = y_py)
dpred_py  <- xgb_py$DMatrix(data = as.matrix(X))

compute_poisson_loglik <- function(model, dtrain_py, dpred_py, vol_py, scale_factor = 500) {
  # Pastikan numpy sudah di-import
  np <- import("numpy", convert = TRUE)
  
  # --- Ambil prediksi dari model Python ---
  all_preds <- model$predict(dpred_py)
  
  # --- Ambil label dari DMatrix (Python) ---
  labels_py <- np$ravel(dtrain_py$get_label())
  
  # --- Buat mask event dan dummy ---
  event_mask <- labels_py > 0
  dummy_mask <- labels_py < 0
  
  # --- Clipping nilai prediksi ---
  all_preds_clipped <- np$clip(all_preds, -50, 50)
  
  y_pred_event <- all_preds_clipped[event_mask]
  y_pred_dummy <- all_preds_clipped[dummy_mask]
  
  #y_pred_event <- all_preds[event_mask]
  #y_pred_dummy <- all_preds[dummy_mask]
  
  # --- Hitung komponen log-likelihood ---
  # left = sum(log(exp(f_event)/(scale_factor^2)))
  #left <- sum(log(np$exp(y_pred_event)/(scale_factor^2)))
  left <- sum(y_pred_event - log((scale_factor)^2))
  #left <- np$sum(y_pred_event) - length(y_pred_event)*2*log(scale_factor)
  
  # right = sum(exp(f_dummy) * vol_dummy)
  right <- sum(exp(all_preds_clipped) * vol_py)
  #right <- sum(exp(all_preds) * vol_py)
  
  # --- Log-likelihood akhir ---
  log_likelihood <- left - right
  
  cat("Poisson log-likelihood:", as.numeric(log_likelihood), "\n")
  return(as.numeric(log_likelihood))
}

compute_logistic_loglik <- function(model, dtrain_py, dpred_py, vol_py) {
  np <- import("numpy", convert = TRUE)
  
  all_preds <- model$predict(dpred_py)
  labels_py <- np$ravel(dtrain_py$get_label())
  
  event_mask <- labels_py > 0
  dummy_mask <- labels_py < 0
  
  y_pred_event <- np$clip(all_preds[event_mask], -50, 50)
  y_pred_dummy <- np$clip(all_preds[dummy_mask], -50, 50)
  vol_event <- vol_py[event_mask]
  vol_dummy <- vol_py[dummy_mask]
  
  delta_event <- 1.0 / vol_event
  delta_dummy <- 1.0 / vol_dummy
  
  left_val <- sum(log(exp(y_pred_event)/(exp(y_pred_event)+delta_event)))
  right_val <- sum(log(delta_dummy/ (exp(y_pred_dummy) + delta_dummy)))
  
  log_likelihood <- left_val + right_val
  
  cat("Logistic log-likelihood:", as.numeric(log_likelihood), "\n")
  
  return(as.numeric(log_likelihood))
}


# --- Grid Search Manual ---
etas <- c(0.1, 0.01, 0.001)
lambdas <- c(0,10,30,50)
loss <- "poisson"
loss <- "weighted_poisson"
loss <- "logistic"
loss <- "weighted_logistic"

# Simpan hasil loglik
results <- data.frame(
  eta = numeric(),
  lambda = numeric(),
  ploglik = numeric(),
  lloglik = numeric(),
  stringsAsFactors = FALSE
)

# Loop kombinasi parameter
for (eta_val in etas) {
  for (lambda_val in lambdas) {
    cat("=== Testing eta =", eta_val, ", lambda =", lambda_val, "===\n")
    
    # Definisikan parameter ke Python dict
    final_params_py <- dict(
      booster = "gbtree",
      subsample = 0.8,
      colsample_bytree = 1/3,
      nthread = as.integer(-1),
      tree_method = "exact",
      verbosity = as.integer(0),
      eta = eta_val,
      alpha = 0,
      lambda = lambda_val,
      max_depth = as.integer(8)
    )
    
    # Konversi F_prime
    #F_prime_py <- np$float64(F_prime)
    F_prime_py <- F_prime_py
    # Jalankan model
    model <- xgbpp_py(
      dtrain = dtrain_py,
      vol = vol_py,
      params = final_params_py,
      loss = loss,
      F_prime = F_prime_py,
      lambdau = lambda_py,
      evals = list(tuple(dtrain_py, 'train')),
      num_boost_round = as.integer(5000),
      early_stopping_rounds = as.integer(50),
      verbose_eval = FALSE
    )
    
    # Hitung log-likelihood
    ploglik <- compute_poisson_loglik(model, dtrain_py, dpred_py, vol_py, scale_factor = sf)
    lloglik <- compute_logistic_loglik(model, dtrain_py, dpred_py, vol_py)
    # Simpan hasil
    results <- rbind(results, data.frame(
      eta = eta_val,
      lambda = lambda_val,
      ploglik = ploglik,
      lloglik = lloglik
    ))
  }
}
# Cari kombinasi terbaik
best_result <- results[which.max(results$ploglik), ]
cat("\n=== Best combination ===\n")
print(best_result)
# Cari kombinasi terbaik
best_result <- results[which.max(results$lloglik), ]
cat("\n=== Best combination ===\n")
print(best_result)

loss <- "poisson"
loss <- "weighted_poisson"
loss <- "logistic"
loss <- "weighted_logistic"

# Buat parameter dictionary Python (bukan list R!)
final_params_py <- dict(
  booster = "gbtree",
  subsample = 0.8,
  colsample_bytree = 1/3,
  nthread = as.integer(-1),
  tree_method = "exact",
  verbosity = as.integer(0),
  eta = 0.001,
  alpha = 0,
  lambda = 50,
  max_depth = as.integer(8)
)

# Konversi F_prime ke Python float  
F_prime_py <- F_prime_py

# Jalankan model (langsung di Python)
tic()
model <- xgbpp_py(
  dtrain = dtrain_py,
  vol = vol_py,
  params = final_params_py,
  loss = loss,
  F_prime = F_prime_py,
  lambdau = lambda_py,
  evals = list(tuple(dtrain_py, 'train')),
  #evals = list(list(dtrain, 'train')),
  num_boost_round = as.integer(5000),
  early_stopping_rounds = as.integer(50),
  verbose_eval=FALSE
)
toc()

compute_poisson_loglik(model, dtrain_py, dpred_py, vol_py, scale_factor = sf)
compute_logistic_loglik(model, dtrain_py, dpred_py, vol_py)

# --- Ambil prediksi dari model Python ---
all_preds <- model$predict(dpred_py)

# --- Ambil label dari DMatrix (Python) ---
labels_py <- np$ravel(dtrain_py$get_label())

# --- Buat mask event dan dummy ---
event_mask <- labels_py > 0
dummy_mask <- labels_py < 0

# --- Clipping nilai prediksi ---
all_preds_clipped <- np$clip(all_preds, -50, 50)

y_pred_event <- all_preds_clipped[event_mask]
y_pred_dummy <- all_preds_clipped[dummy_mask]
#we <- 1/(1+y_pred_event*F_prime)
#prediksi_dummy <- log(exp(y_pred_dummy)/(sf*sf))
prediksi_dummy <- log(exp(y_pred_dummy)/(sf*sf))
# dummy_ppp <- ppp(x = truedata$x, 
#                  y = truedata$y, 
#                  window = win_left, check = FALSE)
# dummy_ppp <- ppp(x = df$x,
#                  y = df$y,
#                  window = win_left, check = FALSE)

########## LINNET ###################

ddfx = df$x[df$label == -1]
ddfy = df$y[df$label == -1]
ddf = data.frame(x=ddfx,y=ddfy)

dummy_ppp <- lpp(X = ddf, L = nganjuk_ln)
plot(unmark(dummy_ppp),pch=19,cex=0.2,main="Traffic Accident")#

#marks(dummy_ppp) <- data.frame(prediction = as.vector(prediksi_dummy))
marks(dummy_ppp) <- data.frame(prediction = as.vector(1))
plot(dummy_ppp,
     which.marks = "prediction",
     main = "Predicted Value at Dummy Points")

W <- as.owin(nganjuk_ln)
# 
# pp_quad <- ppp(x = df$x[df$label == -1],
#                y = df$y[df$label == -1],
#                window = W,
#                marks = as.vector(exp(prediksi_dummy)))
# 
# im_pred <- pixellate(pp_quad, weights = marks(pp_quad), dims = c(300,300))
# 
# intensity_map_ml_linim <- linim(nganjuk_ln, im_pred)   # sekarang Z adalah im -> seharusnya lolos is.im check
# 
# plot(intensity_map_ml_linim)

prediction_surface2 <- Smooth(dummy_ppp,
                             sigma = 0.1,
                             what = "marks",
                             dx = 0.01,
                             #iterMax = 3e+06,
                             finespacing = FALSE,
                             leaveout = FALSE)

# The result is a 'linim' (a pixel image on the network)
plot(prediction_surface2,
     main = "Smoothed Prediction Surface (linim)")

plot(truedata$x, truedata$y,
     type = "n",       # membuat kanvas kosong
     xlab = "X",
     ylab = "Y")

plot(W)
points(truedata$x, truedata$y, pch=20, cex=0.35, col="black")

########## SPP ###################
# 
# Smoothing (Intensity Estimation) from prediction
marks(dummy_ppp) <- data.frame(prediction = as.vector(prediksi_dummy))
intensity_map <- Smooth(dummy_ppp)
range(intensity_map)
plot(intensity_map,                              # first layer
     col  = viridis::viridis(100, option = 'plasma'),               # any palette
     ribbon = TRUE,                              # colour-bar
     ribbon.pos = "bottom",
     asp  = 1,                                   # enforces 1:1
     main = " ",
     zlim   = c(-17, 2))

points(truedata$x, truedata$y, pch=19, cex=0.2, col="black")

importance_df <- py$get_feature_importance_xgbpp(
  model,
  #feature_names = c("depth","dip","fault","strike","megathrust","volcano"),
  feature_names = features_cols,  # optional
  importance_type = "gain"
)

# Convert to R data.frame
importance_df_R <- py_to_r(importance_df)
importance_df_R$importance <- importance_df_R$importance / sum(importance_df_R$importance)
importance_df_R$feature <- recode(
  importance_df_R$feature,
  "n_Lajur" = "lane",
  "Jarak_Lampu" = "traffic light",
  "Kerb_L" = "curb-left",
  "Kerb_R" = "curb-right",
  "n_Jalur" = "highway"
)


library(ggplot2)
ggplot(head(importance_df_R, 10), aes(x = reorder(feature, importance), y = importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  #scale_y_continuous(limits = c(0, 0.2)) +
  labs(title = "", x = "Feature", y = "Importance")

