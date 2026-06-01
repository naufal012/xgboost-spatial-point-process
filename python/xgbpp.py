import numpy as np
import pandas as pd
import xgboost as xgb
import time
import os
from tqdm.auto import tqdm
# --- NEW: Import Optuna for Bayesian Optimization ---
import optuna

# =============================================================
# BAGIAN 1: FUNGSI-FUNGSI OBJECTIVE & METRIC UNTUK XGBOOST
# =============================================================

def make_poisson_obj_xgb(vol_vec):
    def poisson_obj(preds, dtrain):
        preds = np.clip(preds, -50, 50)
        labels = dtrain.get_label()
        weights = np.ones_like(labels)
        dummy_idx = np.where(labels < 0)[0]; event_idx = np.where(labels > 0)[0]
        norm = np.sum(weights * vol_vec) if len(dummy_idx) > 0 else 1.0
        if abs(norm) < 1e-9: norm = 1.0
        weights /= norm
        grad = np.zeros_like(labels, dtype=float); hess = np.zeros_like(labels, dtype=float)
        if len(event_idx) > 0:
            mu_ev = np.exp(preds[event_idx]); ge = weights[event_idx] * (-1 + mu_ev * vol_vec[event_idx])
            grad[event_idx] = ge; hess[event_idx] = np.maximum((ge + weights[event_idx]), 1e-6)
        if len(dummy_idx) > 0:
            mu_dm = np.exp(preds[dummy_idx]); gh = weights[dummy_idx] * mu_dm * vol_vec[dummy_idx]
            grad[dummy_idx] = gh; hess[dummy_idx] = np.maximum(gh, 1e-6)
        return grad, hess
    return poisson_obj

def make_weighted_poisson_obj_xgb(vol_vec, F_prime_val, hess_min=1e-6):
    def weighted_poisson_obj(preds, dtrain):
        preds = np.clip(preds, -50, 50)
        label = dtrain.get_label()
        weight = 1 / (1 + np.exp(preds) * F_prime_val)
        neg_idx = np.where(label < 0)[0]
        sum_neg_weights = np.sum(weight * vol_vec)
        if sum_neg_weights > 1e-9:
            weight /= sum_neg_weights
        pos_idx = np.where(label > 0)[0]
        grad = np.zeros_like(label, dtype=float); hess = np.zeros_like(label, dtype=float)
        grad[pos_idx] = -1 + np.exp(preds[pos_idx]) * vol_vec[pos_idx]; grad[neg_idx] = np.exp(preds[neg_idx]) * vol_vec[neg_idx]
        grad *= weight
        hess[pos_idx] = np.exp(preds[pos_idx]) * vol_vec[pos_idx]; hess[neg_idx] = np.exp(preds[neg_idx]) * vol_vec[neg_idx]
        hess = np.maximum(weight * hess, hess_min)
        return grad, hess
    return weighted_poisson_obj

def make_poisson_metric_xgb(vol_vec):
    def poisson_metric(preds, dtrain):
        preds = np.clip(preds, -50, 50)
        labels = dtrain.get_label()
        event_idx = np.where(labels > 0)[0]; dummy_idx = np.where(labels < 0)[0]
        err = np.zeros_like(labels, dtype=float)
        if len(event_idx) > 0: err[event_idx] = -preds[event_idx]
        if len(dummy_idx) > 0: err[dummy_idx] = np.exp(preds[dummy_idx]) * vol_vec[dummy_idx]
        corrections = np.sum(np.exp(preds[event_idx]) * vol_vec[event_idx])
        return 'poisson_err', (np.sum(err) + corrections) / 1e6
    return poisson_metric

def make_logistic_obj_xgb(vol_vec, hess_min=1e-6):
    def logistic_obj(preds, dtrain):
        preds = np.clip(preds, -50, 50)
        label = dtrain.get_label()
        
        neg_idx = np.where(label < 0)[0]
        pos_idx = np.where(label > 0)[0]
        
        grad = np.zeros_like(label, dtype=float); hess = np.zeros_like(label, dtype=float)
        safe_vol_vec_neg = vol_vec[neg_idx].copy()
        safe_vol_vec_neg[safe_vol_vec_neg == 0] = 1e-9
        safe_vol_vec_pos = vol_vec[pos_idx].copy()
        safe_vol_vec_pos[safe_vol_vec_pos == 0] = 1e-9
        safe_vol_vec = vol_vec.copy()
        safe_vol_vec[safe_vol_vec == 0] = 1e-9
        #delta = len(neg_idx)/np.sum(vol_vec); delta_pos = delta; delta_neg = delta
        delta_pos = 1.0/safe_vol_vec_pos; delta_neg = 1.0/safe_vol_vec_neg; delta = 1.0/safe_vol_vec
        exp_preds_pos = np.exp(preds[pos_idx]); exp_preds_neg = np.exp(preds[neg_idx])

        #weight = (np.exp(preds) + delta) / (delta)
        weight = np.ones_like(label)
        sum_neg_weights = np.sum(weight * vol_vec)
        if sum_neg_weights > 1e-9:
            weight /= sum_neg_weights

        grad[pos_idx] = weight[pos_idx] * (((exp_preds_pos)/(exp_preds_pos + delta_pos)) - 1)
        grad[neg_idx] = weight[neg_idx] * (((exp_preds_neg)/(exp_preds_neg + delta_neg)))
        hess[pos_idx] = weight[pos_idx] * ((exp_preds_pos * delta_pos)/((exp_preds_pos + delta_pos)**2))
        hess[neg_idx] = weight[neg_idx] * ((exp_preds_neg * delta_neg)/((exp_preds_neg + delta_neg)**2))
        hess = np.maximum(hess, hess_min)
        return grad, hess
    return logistic_obj

def make_weighted_logistic_obj_xgb(vol_vec, F_prime_val, hess_min=1e-6):
    def weighted_logistic_obj(preds, dtrain):
        preds = np.clip(preds, -50, 50)
        label = dtrain.get_label()
        
        neg_idx = np.where(label < 0)[0]
        pos_idx = np.where(label > 0)[0]
        
        grad = np.zeros_like(label, dtype=float); hess = np.zeros_like(label, dtype=float)
        safe_vol_vec_neg = vol_vec[neg_idx].copy()
        safe_vol_vec_neg[safe_vol_vec_neg == 0] = 1e-9
        safe_vol_vec_pos = vol_vec[pos_idx].copy()
        safe_vol_vec_pos[safe_vol_vec_pos == 0] = 1e-9
        safe_vol_vec = vol_vec.copy()
        safe_vol_vec[safe_vol_vec == 0] = 1e-9
        delta_pos = 1.0/safe_vol_vec_pos; delta_neg = 1.0/safe_vol_vec_neg; delta = 1.0/safe_vol_vec
        exp_preds_pos = np.exp(preds[pos_idx]); exp_preds_neg = np.exp(preds[neg_idx])

        weight = (np.exp(preds) + delta) / (delta * (1 + np.exp(preds) * F_prime_val))
        sum_neg_weights = np.sum(weight * vol_vec)
        if sum_neg_weights > 1e-9:
            weight /= sum_neg_weights

        grad[pos_idx] = weight[pos_idx] * (((exp_preds_pos)/(exp_preds_pos + delta_pos)) - 1)
        grad[neg_idx] = weight[neg_idx] * (((exp_preds_neg)/(exp_preds_neg + delta_neg)))
        hess[pos_idx] = weight[pos_idx] * ((exp_preds_pos * delta_pos)/((exp_preds_pos + delta_pos)**2))
        hess[neg_idx] = weight[neg_idx] * ((exp_preds_neg * delta_neg)/((exp_preds_neg + delta_neg)**2))
        hess = np.maximum(hess, hess_min)
        return grad, hess
    return weighted_logistic_obj

def make_logistic_metric_xgb(vol_vec,F_prime_val):
    def logistic_metric(preds, dtrain):
        preds = np.clip(preds, -50, 50)
        labels = dtrain.get_label()
        event_idx = np.where(labels > 0)[0]; dummy_idx = np.where(labels < 0)[0]
        err = np.zeros_like(labels, dtype=float)

        safe_vol_vec_neg = vol_vec[dummy_idx].copy()
        safe_vol_vec_neg[safe_vol_vec_neg == 0] = 1e-9
        safe_vol_vec_pos = vol_vec[event_idx].copy()
        safe_vol_vec_pos[safe_vol_vec_pos == 0] = 1e-9
        safe_vol_vec = vol_vec.copy()
        safe_vol_vec[safe_vol_vec == 0] = 1e-9
        delta_pos = 1.0/safe_vol_vec_pos; delta_neg = 1.0 / safe_vol_vec_neg; delta = 1/safe_vol_vec

        weight = np.ones_like(labels)
        #weight = (np.exp(preds) + delta) / (delta * (1 + np.exp(preds) * F_prime_val))
        #sum_neg_weights = np.sum(weight[dummy_idx] * safe_vol_vec_neg)
        #if sum_neg_weights > 1e-9:
        #    weight /= sum_neg_weights
        if len(event_idx) > 0: err[event_idx] = -(weight[event_idx] * np.log(np.exp(preds[event_idx]) / (delta_pos + np.exp(preds[event_idx]))))
        if len(dummy_idx) > 0: err[dummy_idx] = (weight[dummy_idx] * delta_neg * np.log((np.exp(preds[dummy_idx]) + delta_neg) / delta_neg)) * safe_vol_vec_neg
        return 'logistic_err', np.sum(err) / 1e6
    return logistic_metric

def xgbpp_py(dtrain, vol, params, loss="poisson", F_prime=1, **kwargs):
    if loss == "poisson": objective_func = make_poisson_obj_xgb(vol)
    elif loss == "weighted_poisson": objective_func = make_weighted_poisson_obj_xgb(vol, F_prime)
    elif loss == "logistic": objective_func = make_logistic_obj_xgb(vol)
    elif loss == "weighted_logistic": objective_func = make_weighted_logistic_obj_xgb(vol, F_prime)
    else: raise ValueError(f"Loss '{loss}' tidak dikenali.")
    
    if loss in ["poisson", "weighted_poisson"]: eval_func = make_poisson_metric_xgb(vol)
    elif loss in ["logistic", "weighted_logistic"]: eval_func = make_logistic_metric_xgb(vol, F_prime)
    else: raise ValueError(f"Loss '{loss}' tidak dikenali.")

    params_copy = params.copy()
    
    model = xgb.train(
        params=params_copy,
        dtrain=dtrain,
        obj=objective_func,
        custom_metric=eval_func,
        maximize=False,
        **kwargs
    )
    return model

# =============================================================
# BAGIAN 2: FUNGSI WRAPPER UNTUK TUNING DENGAN OPTUNA (BAYESIAN)
# =============================================================

def tune_xgbpp(X_df, y_series, vol_series, loss, F_prime=1.0, n_trials=100, constrain_events=False, constraint_strength=1.0):
    start_time_total = time.time()
    
    if not isinstance(y_series, pd.Series): y_series = pd.Series(y_series)
    if not isinstance(vol_series, pd.Series): vol_series = pd.Series(vol_series)

    results_dir = f"xgboost/{loss}"
    if not os.path.exists(results_dir): os.makedirs(results_dir)
        
    dtrain = xgb.DMatrix(data=X_df, label=y_series)
    dpred = xgb.DMatrix(data=X_df)
    
    dummy_mask = (y_series < 0).values
    event_mask = (y_series > 0).values
    vol_dummy = vol_series[dummy_mask].values
    vol_event = vol_series[event_mask].values
    
    real_event_count = np.sum(event_mask)

    # Fungsi objective untuk Optuna
    def objective_optuna(trial):
        base_params = {
            'booster': 'gbtree',
            'subsample': 0.8,
            'colsample_bytree': 1/3,
            'nthread': -1,
            'tree_method': 'hist',
            'verbosity': 0
        }
        
        params_to_tune = {
            "eta": trial.suggest_float("eta", 0.001, 0.5),
            "alpha": trial.suggest_float("alpha", 2**-10, 2**10, log=True),
            "lambda": trial.suggest_float("lambda", 2**-10, 2**10, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
        }
        
        current_params = {**base_params, **params_to_tune}
        
        booster = xgbpp_py(
            dtrain=dtrain, vol=vol_series.values, params=current_params,
            loss=loss, F_prime=F_prime,
            num_boost_round=5000,
            evals=[(dtrain, 'train')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        all_preds = booster.predict(dpred)
        y_pred_dummy = all_preds[dummy_mask]
        y_pred_event = all_preds[event_mask]
        
        if loss in ["poisson", "weighted_poisson"]:
            # --- Log-likelihood Poisson ---
            left_val = np.sum(y_pred_event)
            num_events_val = np.sum(np.exp(y_pred_dummy) * vol_dummy)
            log_likelihood = left_val - num_events_val

        elif loss in ["logistic", "weighted_logistic"]:
            # --- Log-likelihood Logistic ---
            # rho(u;β) = exp(f(u))
            rho_event = np.exp(y_pred_event)
            rho_dummy = np.exp(y_pred_dummy)

            delta_dummy = 1/vol_dummy
            delta_event = 1/vol_event

            prob_event_event = rho_event / (rho_event + delta_event)
            prob_event_dummy = rho_dummy / (rho_dummy + delta_dummy)

            num_events_val = np.sum(prob_event_event) + np.sum(prob_event_dummy)
            
            # Bagian sum atas event
            left_val = np.sum(np.log(rho_event / (delta_event + rho_event)))

            # Bagian integral (dummy)
            right_val = np.sum(
                delta_dummy * np.log((rho_dummy + delta_dummy) / delta_dummy)
            )

            log_likelihood = left_val - right_val

        else:
            raise ValueError(f"Loss '{loss}' tidak dikenali.")
        
        trial.set_user_attr("left", left_val)
        trial.set_user_attr("right", num_events_val)
        trial.set_user_attr("num_trees", booster.best_iteration)
        trial.set_user_attr("log_likelihood_original", log_likelihood)

        if constrain_events:
            penalty = constraint_strength * ((num_events_val - real_event_count) ** 2)
            return log_likelihood - penalty
        else:
            return log_likelihood

    # Jalankan studi Optuna
    print(f"--- Memulai Bayesian Optimization dengan {n_trials} percobaan ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_optuna, n_trials=n_trials, show_progress_bar=True)

    # --- Analisis Hasil ---
    print("\n--- Hasil Optimasi Selesai ---")
    best_trial = study.best_trial
    best_params_from_optuna = best_trial.params
    best_value = best_trial.value
    
    print(f"Nilai Skor Terbaik (dengan penalti jika ada): {best_value:.4f}")
    print("Parameter terbaik yang ditemukan:")
    print(best_params_from_optuna)

    results_df = study.trials_dataframe()
    
    # --- UPDATED: Select, rename, and save specific columns including duration ---
    results_df['duration_seconds'] = (results_df['datetime_complete'] - results_df['datetime_start']).dt.total_seconds()
    
    params_cols = [f'params_{p}' for p in best_params_from_optuna.keys()]
    attrs_cols = ['user_attrs_left', 'user_attrs_right', 'user_attrs_log_likelihood_original', 'user_attrs_num_trees']
    value_col = ['value']
    duration_col = ['duration_seconds']
    
    cols_to_save = value_col + duration_col + params_cols + attrs_cols
    existing_cols_to_save = [col for col in cols_to_save if col in results_df.columns]
    
    results_to_save = results_df[existing_cols_to_save]
    
    results_to_save = results_to_save.rename(columns=lambda c: c.replace('params_', '').replace('user_attrs_', ''))
    results_to_save = results_to_save.rename(columns={'value': 'score', 'log_likelihood_original': 'log_likelihood'})

    csv_path = os.path.join(results_dir, "optuna_search_results.csv")
    results_to_save.to_csv(csv_path, index=False)
    print(f"\nSemua hasil (kolom terpilih) disimpan di: {csv_path}")
    # --- END UPDATED ---

    # --- Melatih Model Final ---
    print("\n--- Melatih model final dengan parameter terbaik... ---")
    
    base_params_final = {
        'booster': 'gbtree',
        'subsample': 0.8,
        'colsample_bytree': 1/3,
        'nthread': -1,
        'tree_method': 'hist',
        'verbosity': 0
    }
    final_params = {**base_params_final, **best_params_from_optuna}
    
    optimal_rounds = best_trial.user_attrs.get("num_trees", 5000)
    if optimal_rounds <= 0:
        print(f"Peringatan: num_trees adalah {optimal_rounds}. Melatih ulang dengan jumlah ronde maksimum (5000).")
        optimal_rounds = 5000

    print("Parameter Final:")
    print(final_params)

    final_model = xgbpp_py(
        dtrain=dtrain, vol=vol_series.values, params=final_params,
        loss=loss, F_prime=F_prime,
        num_boost_round=optimal_rounds,
        evals=[(dtrain, 'train')],
        verbose_eval=False
    )
    
    all_preds_final = final_model.predict(dpred)
    y_pred_dummy_final = all_preds_final[dummy_mask]
    if loss in ["poisson", "weighted_poisson"]:
        final_num_events = np.sum(np.exp(y_pred_dummy_final) * vol_dummy)
    elif loss in ["logistic", "weighted_logistic"]:
        y_pred_event_final = all_preds_final[event_mask]
        rho_event_final = np.exp(y_pred_event_final)
        rho_dummy_final = np.exp(y_pred_dummy_final)
        delta_event = 1/vol_event
        delta_dummy = 1/vol_dummy
        prob_event_event = rho_event_final / (rho_event_final + delta_event)
        prob_event_dummy = rho_dummy_final / (rho_dummy_final + delta_dummy)
        final_num_events = np.sum(prob_event_event) + np.sum(prob_event_dummy)
    
    final_log_likelihood = best_trial.user_attrs.get("log_likelihood_original", best_value)
    
    print("\n--- HASIL AKHIR ---")
    final_results = {
        'num_events': final_num_events,
        'best_params': final_params,
        'best_log_likelihood': final_log_likelihood,
        'final_model': final_model
    }
    
    end_time_total = time.time()
    print(f"\nTotal waktu tuning: {(end_time_total - start_time_total) / 60:.2f} menit")
    
    return final_results
