standardize_im <- function(img) {
  values <- as.vector(img$v)          # Extract pixel values as vector
  mean_val <- mean(values)
  sd_val <- sd(values)
  # Standardize pixel values
  standardized_values <- (values - mean_val) / sd_val
  dims <- dim(img)
  # Create new image with standardized values but same spatial domain
  im(matrix(standardized_values, nrow=dims[1], ncol=dims[2]),
     xcol = img$xcol,
     yrow = img$yrow,
     xrange = img$xrange,
     yrange = img$yrange,
     unitname = img$unitname)
}

simulate_strauss_process <- function(covariate_names, intercept, coefficients, bci_covars, bei_window, r_inhibit, gamma_inhibit, scale_factor=500, n_points=300) {
  library(spatstat)
  library(spatstat.random)
  library(spatstat.geom)
  
  # 1. Extract and rescale covariates
  cov_images <- lapply(covariate_names, function(name) {
    img <- bci_covars[[name]]
    img_rescaled <- rescale(img, s = scale_factor, unitname = c("km", "kms"))
    standardize_im(img_rescaled)
  })
  names(cov_images) <- covariate_names
  # Create a constant image with the intercept
  zero_matrix <- matrix(0,
                        nrow = length(cov_images[[1]]$yrow),
                        ncol = length(cov_images[[1]]$xcol))
  
  logLambda <- im(zero_matrix,
                  xcol = cov_images[[1]]$xcol,
                  yrow = cov_images[[1]]$yrow,
                  xrange = cov_images[[1]]$xrange,
                  yrange = cov_images[[1]]$yrange,
                  unitname = cov_images[[1]]$unitname)
  
  # Sum covariates times their coefficients
  for (i in seq_along(covariate_names)) {
    logLambda <- logLambda + coefficients[i] * cov_images[[i]]
  }
  
  win <- with(bei_window, owin(xrange = xrange / scale_factor, yrange = yrange / scale_factor))
  Lambda <- eval.im(exp(logLambda))
  beta0 <- log(n_points/(integral.im(Lambda,win)))
  
  mh_model <- rmhmodel(
    cif = "strauss",
    par = list(beta = exp(beta0), gamma = gamma_inhibit, r = r_inhibit),
    trend = Lambda,
    w = win
  )
  
  mh_start <- list(n.start = n_points)
  mh_control <- list(nrep = 1e6)
  
  sim_points <- rmh(model = mh_model, start = mh_start, control = mh_control)
  
  # 7. Extract covariate values at simulated points
  cov_values <- lapply(cov_images, function(img) img[sim_points])
  
  # 8. Build dataframe of x, y and covariates
  df <- data.frame(
    x = sim_points$x,
    y = sim_points$y,
    cov_values
  )
  names(df)[-(1:2)] <- covariate_names  # name covariate columns
  
  return(list(
    sim_points = sim_points,
    intensity = Lambda,
    covariates = cov_images,
    sim_data = df,
    pp = sim_points
  ))
}

simulate_poisson_process <- function(covariate_names, coefficients, bci_covars, bei_window, n_points=300,
                                     scale_factor = 500, intensity_form = 'linear') {
  library(spatstat)
  library(spatstat.random)
  library(spatstat.geom)
  
  # 1. Extract covariates directly (no scaling or standardization)
  ## CHANGED: This is now much simpler
  cov_images <- lapply(covariate_names, function(name) {
    img <- bci_covars[[name]]
    img_rescaled <- rescale(img, s = scale_factor, unitname = c("km", "kms"))
    standardize_im(img_rescaled)
  })
  names(cov_images) <- covariate_names
  
  # 2. Build the log-intensity from RAW covariate values
  ## REMOVED: No more standardized images needed
  zero_matrix <- matrix(0, nrow = nrow(cov_images[[1]]), ncol = ncol(cov_images[[1]]))
  logLambda <- im(zero_matrix, xcol = cov_images[[1]]$xcol, yrow = cov_images[[1]]$yrow)
  
  if (intensity_form == 'linear') {
    # Simple log-linear model: sum of b_i * z_i
    for (i in seq_along(covariate_names)) {
      logLambda <- logLambda + coefficients[i] * cov_images[[i]]
    }
    
  } else if (intensity_form == 'complex') {
    # --- FINAL FLEXIBLE SECTION STARTS HERE ---
    
    if (length(coefficients) != length(covariate_names)) {
      stop("The number of coefficients must match the number of covariates.")
    }
    
    # Loop through each covariate one by one
    for (i in 1:length(covariate_names)) {
      
      # Determine which step of the 4-part cycle we are in
      # (i-1) %% 4 results in 0, 1, 2, 3, 0, 1, 2, 3, ...
      cycle_step <- (i - 1) %% 4
      
      if (cycle_step == 0) {
        # Step 1: Original form (for the 1st, 5th, 9th, ... covariate)
        term <- coefficients[i] * cov_images[[i]]
        
      } else if (cycle_step == 1) {
        # Step 2: Interaction with previous (for the 2nd, 6th, 10th, ... covariate)
        term <- coefficients[i] * cov_images[[i-1]] * cov_images[[i]]
        
      } else if (cycle_step == 2) {
        # Step 3: Exponential form (for the 3rd, 7th, 11th, ... covariate)
        term <- coefficients[i] * exp(cov_images[[i]])
        
      } else { # cycle_step == 3
        # Step 4: Sine form (for the 4th, 8th, 12th, ... covariate)
        term <- coefficients[i] * sin(cov_images[[i]])
      }
      
      # Add the calculated term to the total log-intensity
      logLambda <- logLambda + term
    }
    # --- FINAL FLEXIBLE SECTION ENDS HERE ---
    
  } else if (intensity_form == 'complex_sparse'){
    if (length(covariate_names) < 2 || length(coefficients) < 2) {
      stop("Untuk 'complex_sparse', dibutuhkan minimal 2 kovariat dan 2 koefisien.")
    }
    # Formula: b1*z1 + b2*(z1*z2)
    logLambda <- coefficients[1] * (cov_images[[1]] + cov_images[[2]])/2 + coefficients[2] * sqrt(exp(cov_images[[1]] * cov_images[[2]]))
  } else {
    stop("Error: 'intensity_form' must be either 'linear' or 'complex'.")
  }
  
  # 3. Use the window directly
  ## CHANGED: No more division by scale_factor
  win <- rescale(bei_window, s = scale_factor, unitname = c("km", "kms")) 
  Lambda <- eval.im(exp(logLambda))
  beta0 <- log(n_points / (integral.im(Lambda, win)))
  Lambda <- Lambda * exp(beta0)
  
  # 4. Simulate the point process
  sim_points <- rpoispp(Lambda, win = win, forcewin = TRUE)
  
  # 5. Create the quadrature scheme
  qd_pois <- spatstat.geom::quadscheme(sim_points)
  x_all_pois <- c(qd_pois$data$x, qd_pois$dummy$x)
  y_all_pois <- c(qd_pois$data$y, qd_pois$dummy$y)
  all_pois <- ppp(x_all_pois, y_all_pois, window = win, check = FALSE)
  
  qd_logi <- spatstat.geom::quadscheme.logi(sim_points)
  x_all_logi <- c(qd_logi$data$x, qd_logi$dummy$x)
  y_all_logi <- c(qd_logi$data$y, qd_logi$dummy$y)
  all_logi <- ppp(x_all_logi, y_all_logi, window = win, check = FALSE)
  
  qd_logi_nd2 <- spatstat.geom::quadscheme.logi(sim_points,nd = sqrt(n_points))
  x_all_logi_nd2 <- c(qd_logi_nd2$data$x, qd_logi_nd2$dummy$x)
  y_all_logi_nd2 <- c(qd_logi_nd2$data$y, qd_logi_nd2$dummy$y)
  all_logi_nd2 <- ppp(x_all_logi_nd2, y_all_logi_nd2, window = win, check = FALSE)
  
  # 6. Extract raw covariate values at ALL points (data + dummy)
  cov_values_all_pois <- lapply(cov_images, function(img) img[all_pois])
  cov_values_all_logi <- lapply(cov_images, function(img) img[all_logi])
  cov_values_all_logi_nd2 <- lapply(cov_images, function(img) img[all_logi_nd2])
  
  # 7. Build the complete dataframe for model fitting 📊
  df_full_pois <- as.data.frame(c(
    list(x = x_all_pois, y = y_all_pois),
    cov_values_all_pois,
    list(
      label = as.integer(is.data(qd_pois)), # This function works correctly
      vol = w.quad(qd_pois)                  # This function also works correctly
    )
  ))
  
  df_full_logi <- as.data.frame(c(
    list(x = x_all_logi, y = y_all_logi),
    cov_values_all_logi,
    list(
      label = as.integer(is.data(qd_logi)), # This function works correctly
      vol = w.quad(qd_logi)                  # This function also works correctly
    )
  ))
  
  df_full_logi_nd2 <- as.data.frame(c(
    list(x = x_all_logi_nd2, y = y_all_logi_nd2),
    cov_values_all_logi_nd2,
    list(
      label = as.integer(is.data(qd_logi_nd2)), # This function works correctly
      vol = w.quad(qd_logi_nd2)                  # This function also works correctly
    )
  ))
  
  # 8. Return everything in a list
  return(list(
    alpha = beta0,
    sim_points = sim_points,
    intensity = Lambda,
    sim_data_full_pois = df_full_pois,
    sim_data_full_logi = df_full_logi,
    sim_data_full_logi_nd2 = df_full_logi_nd2,
    quad_scheme_pois = qd_pois,
    quad_scheme_logi = qd_logi
  ))
}

simulate_thomas_process <- function(covariate_names, coefficients, bci_covars, bei_window, n_points=300,
                                     scale_factor = 500, intensity_form = 'linear') {
  library(spatstat)
  library(spatstat.random)
  library(spatstat.geom)
  
  # 1. Extract covariates directly (no scaling or standardization)
  ## CHANGED: This is now much simpler
  cov_images <- lapply(covariate_names, function(name) {
    img <- bci_covars[[name]]
    img_rescaled <- rescale(img, s = scale_factor, unitname = c("km", "kms"))
    standardize_im(img_rescaled)
  })
  names(cov_images) <- covariate_names
  
  # 2. Build the log-intensity from RAW covariate values
  ## REMOVED: No more standardized images needed
  zero_matrix <- matrix(0, nrow = nrow(cov_images[[1]]), ncol = ncol(cov_images[[1]]))
  logLambda <- im(zero_matrix, xcol = cov_images[[1]]$xcol, yrow = cov_images[[1]]$yrow)
  
  if (intensity_form == 'linear') {
    # Simple log-linear model: sum of b_i * z_i
    for (i in seq_along(covariate_names)) {
      logLambda <- logLambda + coefficients[i] * cov_images[[i]]
    }
    
  } else if (intensity_form == 'complex') {
    # --- FINAL FLEXIBLE SECTION STARTS HERE ---
    
    if (length(coefficients) != length(covariate_names)) {
      stop("The number of coefficients must match the number of covariates.")
    }
    
    # Loop through each covariate one by one
    for (i in 1:length(covariate_names)) {
      
      # Determine which step of the 4-part cycle we are in
      # (i-1) %% 4 results in 0, 1, 2, 3, 0, 1, 2, 3, ...
      cycle_step <- (i - 1) %% 4
      
      if (cycle_step == 0) {
        # Step 1: Original form (for the 1st, 5th, 9th, ... covariate)
        term <- coefficients[i] * cov_images[[i]]
        
      } else if (cycle_step == 1) {
        # Step 2: Interaction with previous (for the 2nd, 6th, 10th, ... covariate)
        term <- coefficients[i] * cov_images[[i-1]] * cov_images[[i]]
        
      } else if (cycle_step == 2) {
        # Step 3: Exponential form (for the 3rd, 7th, 11th, ... covariate)
        term <- coefficients[i] * exp(cov_images[[i]])
        
      } else { # cycle_step == 3
        # Step 4: Sine form (for the 4th, 8th, 12th, ... covariate)
        term <- coefficients[i] * sin(cov_images[[i]])
      }
      
      # Add the calculated term to the total log-intensity
      logLambda <- logLambda + term
    }
    # --- FINAL FLEXIBLE SECTION ENDS HERE ---
    
  } else if (intensity_form == 'complex_sparse'){
    if (length(covariate_names) < 2 || length(coefficients) < 2) {
      stop("Untuk 'complex_sparse', dibutuhkan minimal 2 kovariat dan 2 koefisien.")
    }
    # Formula: b1*z1 + b2*(z1*z2)
    logLambda <- coefficients[1] * (cov_images[[1]] + cov_images[[2]])/2 + coefficients[2] * sqrt(exp(cov_images[[1]] * cov_images[[2]]))
  } else {
    stop("Error: 'intensity_form' must be either 'linear' or 'complex'.")
  }
  
  # 3. Use the window directly
  ## CHANGED: No more division by scale_factor
  win <- rescale(bei_window, s = scale_factor, unitname = c("km", "kms")) 
  Lambda <- eval.im(exp(logLambda))
  beta0 <- log(n_points / (integral.im(Lambda, win)))
  Lambda <- Lambda * exp(beta0)
  
  # 4. Simulate the point process
  #sim_points <- rpoispp(Lambda, win = win, forcewin = TRUE)
  #kappa = sum(win$yrange) / (10 * area.owin(win)); scale = 1e-6 * sum(win$xrange); mu <- Lambda/(kappa * 1e-6 * area.owin(win))
  kappa <- sum(win$yrange) * (5 * area.owin(win)); mu <- Lambda/kappa; scale <- 5e-2 * sum(win$xrange)
  sim_points <- rThomas(kappa, scale = scale, mu = mu, win = win, saveLambda = TRUE)
  
  # 5. Create the quadrature scheme
  qd_pois <- spatstat.geom::quadscheme(sim_points)
  x_all_pois <- c(qd_pois$data$x, qd_pois$dummy$x)
  y_all_pois <- c(qd_pois$data$y, qd_pois$dummy$y)
  all_pois <- ppp(x_all_pois, y_all_pois, window = win, check = FALSE)
  
  qd_logi <- spatstat.geom::quadscheme.logi(sim_points)
  x_all_logi <- c(qd_logi$data$x, qd_logi$dummy$x)
  y_all_logi <- c(qd_logi$data$y, qd_logi$dummy$y)
  all_logi <- ppp(x_all_logi, y_all_logi, window = win, check = FALSE)
  
  qd_logi_nd2 <- spatstat.geom::quadscheme.logi(sim_points,nd = sqrt(n_points))
  x_all_logi_nd2 <- c(qd_logi_nd2$data$x, qd_logi_nd2$dummy$x)
  y_all_logi_nd2 <- c(qd_logi_nd2$data$y, qd_logi_nd2$dummy$y)
  all_logi_nd2 <- ppp(x_all_logi_nd2, y_all_logi_nd2, window = win, check = FALSE)
  
  # 6. Extract raw covariate values at ALL points (data + dummy)
  cov_values_all_pois <- lapply(cov_images, function(img) img[all_pois])
  cov_values_all_logi <- lapply(cov_images, function(img) img[all_logi])
  cov_values_all_logi_nd2 <- lapply(cov_images, function(img) img[all_logi_nd2])
  
  # 7. Build the complete dataframe for model fitting 📊
  df_full_pois <- as.data.frame(c(
    list(x = x_all_pois, y = y_all_pois),
    cov_values_all_pois,
    list(
      label = as.integer(is.data(qd_pois)), # This function works correctly
      vol = w.quad(qd_pois)                  # This function also works correctly
    )
  ))
  
  df_full_logi <- as.data.frame(c(
    list(x = x_all_logi, y = y_all_logi),
    cov_values_all_logi,
    list(
      label = as.integer(is.data(qd_logi)), # This function works correctly
      vol = w.quad(qd_logi)                  # This function also works correctly
    )
  ))
  
  df_full_logi_nd2 <- as.data.frame(c(
    list(x = x_all_logi_nd2, y = y_all_logi_nd2),
    cov_values_all_logi_nd2,
    list(
      label = as.integer(is.data(qd_logi_nd2)), # This function works correctly
      vol = w.quad(qd_logi_nd2)                  # This function also works correctly
    )
  ))
  
  # 8. Return everything in a list
  return(list(
    alpha = beta0,
    sim_points = sim_points,
    intensity = attr(sim_points, "Lambda"),
    base_intensity = Lambda,
    sim_data_full_pois = df_full_pois,
    sim_data_full_logi = df_full_logi,
    sim_data_full_logi_nd2 = df_full_logi_nd2,
    quad_scheme_pois = qd_pois,
    quad_scheme_logi = qd_logi
  ))
}

simulate_lgc_process <- function(covariate_names, coefficients, bci_covars, bei_window, n_points=300,
                                    scale_factor = 500, intensity_form = 'linear') {
  library(spatstat)
  library(spatstat.random)
  library(spatstat.geom)
  
  # 1. Extract covariates directly (no scaling or standardization)
  ## CHANGED: This is now much simpler
  cov_images <- lapply(covariate_names, function(name) {
    img <- bci_covars[[name]]
    img_rescaled <- rescale(img, s = scale_factor, unitname = c("km", "kms"))
    standardize_im(img_rescaled)
  })
  names(cov_images) <- covariate_names
  
  # 2. Build the log-intensity from RAW covariate values
  ## REMOVED: No more standardized images needed
  zero_matrix <- matrix(0, nrow = nrow(cov_images[[1]]), ncol = ncol(cov_images[[1]]))
  logLambda <- im(zero_matrix, xcol = cov_images[[1]]$xcol, yrow = cov_images[[1]]$yrow)
  
  if (intensity_form == 'linear') {
    # Simple log-linear model: sum of b_i * z_i
    for (i in seq_along(covariate_names)) {
      logLambda <- logLambda + coefficients[i] * cov_images[[i]]
    }
    
  } else if (intensity_form == 'complex') {
    # --- FINAL FLEXIBLE SECTION STARTS HERE ---
    
    if (length(coefficients) != length(covariate_names)) {
      stop("The number of coefficients must match the number of covariates.")
    }
    
    # Loop through each covariate one by one
    for (i in 1:length(covariate_names)) {
      
      # Determine which step of the 4-part cycle we are in
      # (i-1) %% 4 results in 0, 1, 2, 3, 0, 1, 2, 3, ...
      cycle_step <- (i - 1) %% 4
      
      if (cycle_step == 0) {
        # Step 1: Original form (for the 1st, 5th, 9th, ... covariate)
        term <- coefficients[i] * cov_images[[i]]
        
      } else if (cycle_step == 1) {
        # Step 2: Interaction with previous (for the 2nd, 6th, 10th, ... covariate)
        term <- coefficients[i] * cov_images[[i-1]] * cov_images[[i]]
        
      } else if (cycle_step == 2) {
        # Step 3: Exponential form (for the 3rd, 7th, 11th, ... covariate)
        term <- coefficients[i] * exp(cov_images[[i]])
        
      } else { # cycle_step == 3
        # Step 4: Sine form (for the 4th, 8th, 12th, ... covariate)
        term <- coefficients[i] * sin(cov_images[[i]])
      }
      
      # Add the calculated term to the total log-intensity
      logLambda <- logLambda + term
    }
    # --- FINAL FLEXIBLE SECTION ENDS HERE ---
    
  } else if (intensity_form == 'complex_sparse'){
    if (length(covariate_names) < 2 || length(coefficients) < 2) {
      stop("Untuk 'complex_sparse', dibutuhkan minimal 2 kovariat dan 2 koefisien.")
    }
    # Formula: b1*z1 + b2*(z1*z2)
    logLambda <- coefficients[1] * (cov_images[[1]] + cov_images[[2]])/2 + coefficients[2] * sqrt(exp(cov_images[[1]] * cov_images[[2]]))
  } else {
    stop("Error: 'intensity_form' must be either 'linear' or 'complex'.")
  }
  
  # 3. Use the window directly
  ## CHANGED: No more division by scale_factor
  win <- rescale(bei_window, s = scale_factor, unitname = c("km", "kms")) 
  Lambda <- eval.im(exp(logLambda))
  beta0 <- log(n_points / (integral.im(Lambda, win)))
  Lambda <- Lambda * exp(beta0)
  
  # 4. Simulate the point process
  #sim_points <- rpoispp(Lambda, win = win, forcewin = TRUE)
  #kappa = sum(win$yrange) / (10 * area.owin(win)); scale = 1e-6 * sum(win$xrange)
  #sim_points <- rThomas(kappa, scale = scale, mu = Lambda/(kappa * 1e-6 * area.owin(win)), win = win)
  sim_points <- rLGCP(model = "gauss", mu = as.im(log(Lambda) - 1/2), win = win, var = 1, scale = 0.05,
                      nsim = 1, saveLambda = TRUE)
  
  # 5. Create the quadrature scheme
  qd_pois <- spatstat.geom::quadscheme(sim_points)
  x_all_pois <- c(qd_pois$data$x, qd_pois$dummy$x)
  y_all_pois <- c(qd_pois$data$y, qd_pois$dummy$y)
  all_pois <- ppp(x_all_pois, y_all_pois, window = win, check = FALSE)
  
  qd_logi <- spatstat.geom::quadscheme.logi(sim_points)
  x_all_logi <- c(qd_logi$data$x, qd_logi$dummy$x)
  y_all_logi <- c(qd_logi$data$y, qd_logi$dummy$y)
  all_logi <- ppp(x_all_logi, y_all_logi, window = win, check = FALSE)
  
  qd_logi_nd2 <- spatstat.geom::quadscheme.logi(sim_points,nd = sqrt(n_points))
  x_all_logi_nd2 <- c(qd_logi_nd2$data$x, qd_logi_nd2$dummy$x)
  y_all_logi_nd2 <- c(qd_logi_nd2$data$y, qd_logi_nd2$dummy$y)
  all_logi_nd2 <- ppp(x_all_logi_nd2, y_all_logi_nd2, window = win, check = FALSE)
  
  # 6. Extract raw covariate values at ALL points (data + dummy)
  cov_values_all_pois <- lapply(cov_images, function(img) img[all_pois])
  cov_values_all_logi <- lapply(cov_images, function(img) img[all_logi])
  cov_values_all_logi_nd2 <- lapply(cov_images, function(img) img[all_logi_nd2])
  
  # 7. Build the complete dataframe for model fitting 📊
  df_full_pois <- as.data.frame(c(
    list(x = x_all_pois, y = y_all_pois),
    cov_values_all_pois,
    list(
      label = as.integer(is.data(qd_pois)), # This function works correctly
      vol = w.quad(qd_pois)                  # This function also works correctly
    )
  ))
  
  df_full_logi <- as.data.frame(c(
    list(x = x_all_logi, y = y_all_logi),
    cov_values_all_logi,
    list(
      label = as.integer(is.data(qd_logi)), # This function works correctly
      vol = w.quad(qd_logi)                  # This function also works correctly
    )
  ))
  
  df_full_logi_nd2 <- as.data.frame(c(
    list(x = x_all_logi_nd2, y = y_all_logi_nd2),
    cov_values_all_logi_nd2,
    list(
      label = as.integer(is.data(qd_logi_nd2)), # This function works correctly
      vol = w.quad(qd_logi_nd2)                  # This function also works correctly
    )
  ))
  
  # 8. Return everything in a list
  return(list(
    alpha = beta0,
    sim_points = sim_points,
    intensity = attr(sim_points, "Lambda"),
    base_intensity = Lambda,
    sim_data_full_pois = df_full_pois,
    sim_data_full_logi = df_full_logi,
    sim_data_full_logi_nd2 = df_full_logi_nd2,
    quad_scheme_pois = qd_pois,
    quad_scheme_logi = qd_logi
  ))
}

build_dummy_dataframe <- function(dummy_coords, covariate_names, bci_covars, scale_factor = 500) {
  # dummy_coords: a data.frame or matrix with columns x,y giving dummy locations (unscaled or scaled)
  # covariate_names: vector of covariate image names
  # bci_covars: list of covariate images
  # scale_factor: factor used to scale coordinates (apply same scaling as for simulation)
  
  # Rescale covariates once (optional, to avoid repeating)
  cov_images <- lapply(covariate_names, function(name) {
    img <- bci_covars[[name]]
    rescale(img, s = scale_factor, unitname = c("km", "kms"))
  })
  names(cov_images) <- covariate_names
  
  # Create ppp object for dummy points (scaled coords)
  dummy_ppp <- ppp(x = dummy_coords$x, y = dummy_coords$y, window = as.owin(c(range(dummy_coords$x), range(dummy_coords$y))))
  
  # Extract covariate values at dummy points
  cov_values <- lapply(cov_images, function(img) img[dummy_ppp])
  
  # Build dataframe with covariates
  df_dummy <- data.frame(
    x = dummy_coords$x,
    y = dummy_coords$y,
    cov_values
  )
  names(df_dummy)[-(1:2)] <- covariate_names
  
  return(df_dummy)
}