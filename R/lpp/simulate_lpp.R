# Full combined code: robust linnet LPP simulator with scale support
# ---------------------------------------------------------------
# Required packages:
# install.packages(c("spatstat.geom","spatstat.linnet","spatstat.core","dplyr"))
library(spatstat.geom)    # im, as.im, rescale, as.owin, ppp, integral.im
library(spatstat.linnet)  # linnet, linim, rpoislpp, linequad
library(spatstat.explore)    # Smooth (if wanted)
library(dplyr)

# ---------- Helpers ----------
standardize_im_safe <- function(img) {
  if (!inherits(img, "im")) stop("standardize_im_safe: input must be an 'im' object")
  v <- img$v
  v[!is.finite(v)] <- NA
  vals <- as.vector(v)
  mean_val <- mean(vals, na.rm = TRUE)
  sd_val <- sd(vals, na.rm = TRUE)
  if (is.na(sd_val) || sd_val == 0) {
    warning("standardize_im_safe: zero or undefined sd -> centering only (sd forced = 1)")
    sd_val <- 1
  }
  standardized_values <- (vals - mean_val) / sd_val
  mat <- matrix(standardized_values, nrow = nrow(v), ncol = ncol(v))
  im(mat,
     xcol = img$xcol,
     yrow = img$yrow,
     xrange = img$xrange,
     yrange = img$yrange,
     unitname = img$unitname)
}

safe_exp_im <- function(logim, clip_lower = -50, clip_upper = 50) {
  if (!inherits(logim, "im")) stop("safe_exp_im: input must be 'im'")
  v <- logim$v
  v[!is.finite(v)] <- NA
  q_lo <- tryCatch(quantile(v, probs = 0.01, na.rm = TRUE), error = function(e) NA_real_)
  q_hi <- tryCatch(quantile(v, probs = 0.99, na.rm = TRUE), error = function(e) NA_real_)
  if (is.na(q_lo)) q_lo <- clip_lower/2
  if (is.na(q_hi)) q_hi <- clip_upper/2
  lower <- pmin(q_lo - 10, clip_lower)
  upper <- pmax(q_hi + 10, clip_upper)
  v_clipped <- v
  v_clipped[!is.na(v_clipped) & v_clipped < lower] <- lower
  v_clipped[!is.na(v_clipped) & v_clipped > upper] <- upper
  mat <- exp(v_clipped)
  mat[!is.finite(mat)] <- 0
  im(matrix(mat, nrow = nrow(logim$v), ncol = ncol(logim$v)),
     xcol = logim$xcol, yrow = logim$yrow,
     xrange = logim$xrange, yrange = logim$yrange,
     unitname = logim$unitname)
}

as.im_force <- function(x, W, dimyx = c(200,200)) {
  # x can be linfun, im, or function; W is owin
  if (inherits(x, "im")) {
    # resample to window W if windows mismatch
    if (!identical(x$xrange, W$xrange) || !identical(x$yrange, W$yrange)) {
      return(as.im(x, W = W, dimyx = dimyx))
    } else return(x)
  }
  return(as.im(x, W = W, dimyx = dimyx))
}

# ---------- Main function ----------
simulate_LPP_process <- function(L,
                                 covariates_list,
                                 coefficients,
                                 n_points = 2000,
                                 n_pix = 200,
                                 intensity_form = 'linear',     # 'linear' | 'complex' | 'complex_sparse'
                                 smooth_sigma = NULL,
                                 scale_factor = 1,
                                 standardize = TRUE,
                                 rescale_intensity = TRUE,
                                 return_scaled_network = TRUE,
                                 out_unitname = c("km","kms")) {
  # L: linnet
  # covariates_list: named list of 'im' objects or functions (x,y,seg,tp) or linfun-like
  # coefficients: numeric vector (named or in same order)
  # scale_factor: multiply coordinates by this factor (>0)
  # standardize: whether to z-score covariate images
  # rescale_intensity: whether to normalize intensity to have expected n_points
  # return_scaled_network: return scaled L if TRUE
  # out_unitname: unitname for output images (character vector length 2)
  
  if (!inherits(L, "linnet")) stop("L must be a 'linnet' object")
  if (!is.list(covariates_list) || length(covariates_list) == 0) stop("covariates_list must be a non-empty list")
  if (!is.numeric(scale_factor) || length(scale_factor) != 1 || scale_factor <= 0) stop("scale_factor must be a single positive number")
  
  # Keep a copy of original L for plotting/comparison
  L_orig <- L
  
  # --- RESCALE NETWORK & WINDOW (if requested) ---
  W <- as.owin(L)
  if (scale_factor != 1) {
    # use spatstat.geom::affine to scale linnet and window
    L <- spatstat.geom::affine(L, sx = scale_factor, sy = scale_factor)
    W <- spatstat.geom::affine(W, sx = scale_factor, sy = scale_factor)
  }
  
  # --- Rescale / wrap covariates ---
  covariates_list_scaled <- lapply(covariates_list, function(f_orig) {
    # im objects: use spatstat.geom::rescale to correctly update internals
    if (inherits(f_orig, "im")) {
      if (scale_factor != 1) {
        f_scaled <- spatstat.geom::rescale(f_orig, s = scale_factor, unitname = out_unitname)
      } else f_scaled <- f_orig
      return(f_scaled)
    }
    # functions/linfun: create robust wrapper that rescales x,y before calling original
    if (is.function(f_orig)) {
      wrapper <- function(...) {
        args <- list(...)
        if (!is.null(args$x)) args$x <- args$x / scale_factor
        if (!is.null(args$y)) args$y <- args$y / scale_factor
        # Try call; fallback removing seg/tp if original doesn't accept them
        res <- tryCatch({
          do.call(f_orig, args)
        }, error = function(e1) {
          args2 <- args
          args2$seg <- NULL; args2$tp <- NULL
          args2 <- args2[!sapply(args2, is.null)]
          tryCatch({
            do.call(f_orig, args2)
          }, error = function(e2) {
            # positional fallback: x,y
            if (!is.null(args$x) && !is.null(args$y)) {
              do.call(f_orig, list(args$x, args$y))
            } else stop("covariate function failed inside wrapper: ", e2$message)
          })
        })
        res
      }
      return(wrapper)
    }
    # else: return as-is (as.im_force may coerce)
    f_orig
  })
  names(covariates_list_scaled) <- names(covariates_list)
  
  # --- 1. Build 2D images for covariates covering the scaled window ---
  dims <- c(n_pix, n_pix)
  cov_images_2D <- lapply(covariates_list_scaled, function(f_loop) {
    as.im_force(f_loop, W = W, dimyx = dims)
  })
  names(cov_images_2D) <- names(covariates_list_scaled)
  
  # optional standardization
  if (standardize) {
    cov_images_2D <- lapply(cov_images_2D, standardize_im_safe)
  }
  
  # --- Align coefficients ---
  cov_names <- names(cov_images_2D)
  if (is.null(names(coefficients))) {
    if (length(coefficients) != length(cov_images_2D)) stop(sprintf("coefficients must be length %d (or be named).", length(cov_images_2D)))
    coefficients <- as.numeric(coefficients)
    names(coefficients) <- cov_names
  } else {
    if (!all(cov_names %in% names(coefficients))) stop("Named 'coefficients' must cover all covariate names: ", paste(cov_names, collapse = ", "))
    coefficients <- coefficients[cov_names]
  }
  
  # --- 2. Build log-intensity image robustly ---
  template <- cov_images_2D[[1]]
  zero_mat <- matrix(0, nrow = nrow(template$v), ncol = ncol(template$v))
  logLambda_im <- im(zero_mat, xcol = template$xcol, yrow = template$yrow,
                     xrange = template$xrange, yrange = template$yrange,
                     unitname = out_unitname)
  
  safe_im_vals <- function(imobj) {
    v <- imobj$v
    v[!is.finite(v)] <- NA
    v[is.na(v)] <- 0
    im(matrix(v, nrow = nrow(imobj$v), ncol = ncol(imobj$v)),
       xcol = imobj$xcol, yrow = imobj$yrow,
       xrange = imobj$xrange, yrange = imobj$yrange,
       unitname = imobj$unitname)
  }
  
  if (intensity_form == 'linear') {
    for (i in seq_along(cov_images_2D)) {
      coef_i <- coefficients[i]
      im_i_clean <- safe_im_vals(cov_images_2D[[i]])
      logLambda_im <- logLambda_im + coef_i * im_i_clean
    }
  } else if (intensity_form == 'complex') {
    for (i in seq_along(cov_images_2D)) {
      coef_i <- coefficients[i]
      cycle_step <- (i - 1) %% 4
      im_i <- safe_im_vals(cov_images_2D[[i]])
      if (cycle_step == 0) {
        term_im <- coef_i * im_i
      } else if (cycle_step == 1) {
        im_prev <- safe_im_vals(cov_images_2D[[max(1, i-1)]])
        term_im <- coef_i * (im_prev * im_i)
      } else if (cycle_step == 2) {
        term_im <- coef_i * safe_exp_im(im_i)
      } else {
        mat <- im_i$v
        mat[is.na(mat)] <- 0
        term_im <- coef_i * im(matrix(sin(mat), nrow = nrow(mat), ncol = ncol(mat)),
                               xcol = im_i$xcol, yrow = im_i$yrow,
                               xrange = im_i$xrange, yrange = im_i$yrange,
                               unitname = im_i$unitname)
      }
      term_mat <- term_im$v
      term_mat[!is.finite(term_mat)] <- 0
      term_im$v <- term_mat
      logLambda_im <- logLambda_im + term_im
    }
  } else if (intensity_form == 'complex_sparse') {
    if (length(cov_images_2D) < 2 || length(coefficients) < 2) stop("For 'complex_sparse', need at least 2 covariates and 2 coefficients.")
    im1 <- safe_im_vals(cov_images_2D[[1]])
    im2 <- safe_im_vals(cov_images_2D[[2]])
    term1_mat <- (im1$v + im2$v) / 2
    term2_mat <- sqrt(pmax(0, exp(im1$v * im2$v)))
    term_mat <- coefficients[1] * term1_mat + coefficients[2] * term2_mat
    term_mat[!is.finite(term_mat)] <- 0
    logLambda_im <- im(matrix(term_mat, nrow = nrow(term_mat), ncol = ncol(term_mat)),
                       xcol = template$xcol, yrow = template$yrow,
                       xrange = template$xrange, yrange = template$yrange,
                       unitname = template$unitname)
  } else {
    stop("intensity_form must be 'linear', 'complex', or 'complex_sparse'.")
  }
  
  if (!is.null(smooth_sigma)) logLambda_im <- Smooth(logLambda_im, sigma = smooth_sigma)
  
  # --- 3. exponentiate to get Lambda_im_2D ---
  Lambda_im_2D <- safe_exp_im(logLambda_im)
  # ensure Lambda_im_2D covers window W
  if (!identical(Lambda_im_2D$xrange, W$xrange) || !identical(Lambda_im_2D$yrange, W$yrange)) {
    Lambda_im_2D <- as.im(Lambda_im_2D, W = W, dimyx = dims)
  }
  # clean values
  vals <- Lambda_im_2D$v
  if (all(is.na(vals))) stop("Lambda_im_2D contains only NA values — check covariates and coefficients")
  vals[is.na(vals)] <- median(vals, na.rm = TRUE)
  vals[vals < 0] <- 0
  Lambda_im_2D$v <- vals
  Lambda_im_2D$unitname <- out_unitname
  
  # --- 4. Convert to linim on network safely ---
  Lambda_linim_1D <- tryCatch(spatstat.linnet::linim(L, Lambda_im_2D), error = function(e) {
    warning("linim conversion failed: ", e$message, " -> building constant linim fallback")
    const_val <- mean(Lambda_im_2D$v, na.rm = TRUE)
    im_const <- im(matrix(const_val, nrow = nrow(Lambda_im_2D$v), ncol = ncol(Lambda_im_2D$v)),
                   xcol = Lambda_im_2D$xcol, yrow = Lambda_im_2D$yrow,
                   xrange = Lambda_im_2D$xrange, yrange = Lambda_im_2D$yrange,
                   unitname = Lambda_im_2D$unitname)
    spatstat.linnet::linim(L, im_const)
  })
  
  # --- 5. Rescale intensity to get approximately n_points expected (optional) ---
  total_integral <- tryCatch(spatstat.linnet::integral(Lambda_linim_1D), error = function(e) NA_real_)
  beta0 <- NA_real_
  Lambda_final_linim <- NULL
  if (!is.finite(total_integral) || total_integral <= 0) {
    warning("Calculated intensity integral is zero or non-finite. Falling back to uniform intensity on network.")
    total_length <- sum(lengths(L))
    const_val <- n_points / total_length
    im_const <- im(matrix(const_val, nrow = nrow(Lambda_im_2D$v), ncol = ncol(Lambda_im_2D$v)),
                   xcol = Lambda_im_2D$xcol, yrow = Lambda_im_2D$yrow,
                   xrange = Lambda_im_2D$xrange, yrange = Lambda_im_2D$yrange,
                   unitname = Lambda_im_2D$unitname)
    Lambda_final_linim <- spatstat.linnet::linim(L, im_const)
    beta0 <- NA_real_
  } else {
    beta0 <- log(n_points / total_integral)
    if (rescale_intensity) {
      Lambda_final_linim <- Lambda_linim_1D * exp(beta0)
    } else {
      Lambda_final_linim <- Lambda_linim_1D
      beta0 <- NA_real_
    }
  }
  
  # --- 6. Simulate Poisson LPP from the final intensity and build quadrature ---
  sim_points_lpp <- rpoislpp(Lambda_final_linim)
  qd_scheme <- spatstat.linnet::linequad(sim_points_lpp)
  
  data_df <- as.data.frame(qd_scheme$data)
  data_df$weight <- qd_scheme$weights$data
  dummy_df <- as.data.frame(qd_scheme$dummy)
  dummy_df$weight <- qd_scheme$weights$dummy
  quad_df <- bind_rows(data_df, dummy_df)
  
  full_lpp <- lpp(quad_df[, c(1,2)], L)
  network_coords <- as.data.frame(coords(full_lpp))
  network_coords$label <- spatstat.geom::is.data(qd_scheme)
  network_coords$vol <- w.quad(qd_scheme)
  
  qd_scheme_logi <- quadscheme.logi.linnet(sim_points_lpp, dummytype = 'binomial')
  
  data_df_logi <- as.data.frame(qd_scheme_logi$data)
  dummy_df_logi <- as.data.frame(qd_scheme_logi$dummy)
  quad_df_logi <- bind_rows(data_df_logi, dummy_df_logi)
  
  full_lpp_logi <- lpp(quad_df_logi[, c(1,2)], L)
  network_coords_logi <- as.data.frame(coords(full_lpp_logi))
  network_coords_logi$weight <- qd_scheme_logi$w
  network_coords_logi$label <- spatstat.geom::is.data(qd_scheme_logi)
  network_coords_logi$vol <- w.quad(qd_scheme_logi)
  
  # --- 7. Evaluate covariates at quadrature locations robustly using do.call ---
  cov_values_all <- lapply(covariates_list_scaled, function(f_loop) {
    args <- list(x = network_coords$x, y = network_coords$y, seg = network_coords$seg, tp = network_coords$tp)
    args <- args[!sapply(args, is.null)]
    vals <- tryCatch(do.call(f_loop, args), error = function(e) {
      args2 <- args; args2$seg <- NULL; args2$tp <- NULL
      args2 <- args2[!sapply(args2, is.null)]
      do.call(f_loop, args2)
    })
    vals[!is.finite(vals)] <- NA
    vals
  })
  
  df_full <- as.data.frame(c(list(x = network_coords$x, y = network_coords$y), cov_values_all,
                             list(label = network_coords$label, vol = network_coords$vol)))
  
  cov_values_all_logi <- lapply(covariates_list_scaled, function(f_loop) {
    args <- list(x = network_coords_logi$x, y = network_coords_logi$y, seg = network_coords_logi$seg, tp = network_coords_logi$tp)
    args <- args[!sapply(args, is.null)]
    vals <- tryCatch(do.call(f_loop, args), error = function(e) {
      args2 <- args; args2$seg <- NULL; args2$tp <- NULL
      args2 <- args2[!sapply(args2, is.null)]
      do.call(f_loop, args2)
    })
    vals[!is.finite(vals)] <- NA
    vals
  })
  
  df_full_logi <- as.data.frame(c(list(x = network_coords_logi$x, y = network_coords_logi$y), cov_values_all_logi,
                                  list(label = network_coords_logi$label, vol = network_coords_logi$vol)))
  
  # --- 8. Prepare outputs ---
  out <- list(
    sim_points_lpp     = sim_points_lpp,
    intensity_im_2D    = Lambda_im_2D,
    intensity_linim_1D = Lambda_final_linim,
    sim_data_full      = df_full,
    sim_data_full_logi = df_full_logi,
    quad_scheme        = qd_scheme,
    quad_scheme_logi   = qd_scheme_logi,
    covs               = cov_images_2D,
    alpha              = beta0,
    scale_factor       = scale_factor
  )
  if (return_scaled_network) out$scaled_L <- L
  out$original_L <- L_orig
  return(out)
}

# -----------------------------
# Example usage (replace with your objects)
# -----------------------------
# Suppose you have:
# - L0: a linnet object in original units (e.g., dam)
# - covariates_list0: named list of im objects or functions defined on original units
# - coeffs: numeric vector (same length as covariates)
#
# Example (pseudo):
# out1 <- simulate_LPP_process(L0, covariates_list0, coeffs, n_points=500, scale_factor = 1,
#                              standardize = TRUE, rescale_intensity = TRUE)
# out10 <- simulate_LPP_process(L0, covariates_list0, coeffs, n_points=500, scale_factor = 10,
#                               standardize = TRUE, rescale_intensity = TRUE)
#
# Quick compare (plot windows & intensity images side-by-side):
# par(mfrow = c(2,2))
# plot(as.owin(out1$original_L), main="Original window")
# plot(as.owin(out10$scaled_L), main="Scaled window (factor=10)")
# plot(out1$intensity_im_2D, main="Intensity (scale=1)")
# plot(out10$intensity_im_2D, main="Intensity (scale=10)")
#
# To see the area effect on point counts, run with rescale_intensity = FALSE:
# out10_noscale <- simulate_LPP_process(L0, covariates_list0, coeffs, n_points=500,
#                                       scale_factor = 10, rescale_intensity = FALSE, standardize = FALSE)
# cat("npoints scale=1:", npoints(out1$sim_points_lpp), "\n")
# cat("npoints scale=10 (no re-normalize):", npoints(out10_noscale$sim_points_lpp), "\n")
#
# -----------------------------
# End of script
# -----------------------------
