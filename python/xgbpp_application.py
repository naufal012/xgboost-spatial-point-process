import numpy as np
import pandas as pd
import xgboost as xgb
import time
import os

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
            grad[event_idx] = ge
            #hess[event_idx] = np.maximum((ge + weights[event_idx]), 1e-6)
            hess[event_idx] = weights[event_idx] * mu_ev * vol_vec[event_idx]
        if len(dummy_idx) > 0:
            mu_dm = np.exp(preds[dummy_idx]); gh = weights[dummy_idx] * mu_dm * vol_vec[dummy_idx]
            grad[dummy_idx] = gh; hess[dummy_idx] = np.maximum(gh, 1e-6)
        return grad, hess
    return poisson_obj

def make_weighted_poisson_obj_xgb(vol_vec, F_prime_val, hess_min=1e-6):
    def weighted_poisson_obj(preds, dtrain):
        preds = np.clip(preds, -50, 50)
        label = dtrain.get_label()
        weight = 1 / (1 + F_prime_val)
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

def make_weighted_logistic_obj_xgb(vol_vec, F_prime_val, lambdau, hess_min=1e-6):
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

        weight = (lambdau + delta) / (delta * (1 + F_prime_val))
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

def xgbpp_py(dtrain, vol, params, loss="poisson", F_prime=1, lambdau=0, **kwargs):
    if loss == "poisson": objective_func = make_poisson_obj_xgb(vol)
    elif loss == "weighted_poisson": objective_func = make_weighted_poisson_obj_xgb(vol, F_prime)
    elif loss == "logistic": objective_func = make_logistic_obj_xgb(vol)
    elif loss == "weighted_logistic": objective_func = make_weighted_logistic_obj_xgb(vol, F_prime, lambdau)
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




def get_feature_importance_xgbpp(model, feature_names=None, importance_type="gain", top_k=None, sort=True):
    """
    Extracts the feature importance ranking from an XGBoostPP model.

    Parameters
    ----------
    model : xgb.Booster
        The trained model returned by xgbpp_py().
    feature_names : list of str, optional
        List of feature names (must match the order of columns used to create DMatrix).
        If None, feature names are taken directly from the model.
    importance_type : str, default="gain"
        The type of feature importance to compute.
        Options: "weight", "gain", "cover", "total_gain", "total_cover".
    top_k : int, optional
        If provided, only the top-k most important features are returned.
    sort : bool, default=True
        Whether to sort the features by importance score (descending).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with two columns: ['feature', 'importance'].
    """
    # Get importance values from the model
    importance_dict = model.get_score(importance_type=importance_type)
    
    if not importance_dict:
        print("⚠️ No feature importance found. Ensure the model contains trained trees.")
        return pd.DataFrame(columns=["feature", "importance"])
    
    # If feature names are not provided, use keys from the importance dictionary
    if feature_names is None:
        features = list(importance_dict.keys())
    else:
        # Map 'f0', 'f1', ... to corresponding feature names
        features = []
        for key in importance_dict.keys():
            try:
                idx = int(key[1:])  # e.g., 'f0' -> 0
                features.append(feature_names[idx] if idx < len(feature_names) else key)
            except:
                features.append(key)
    
    # Create DataFrame for importance values
    importance_df = pd.DataFrame({
        "feature": features,
        "importance": list(importance_dict.values())
    })
    
    # Sort by importance score
    if sort:
        importance_df = importance_df.sort_values("importance", ascending=False, ignore_index=True)
    
    # Limit to top_k if specified
    if top_k is not None:
        importance_df = importance_df.head(top_k)
    
    return importance_df