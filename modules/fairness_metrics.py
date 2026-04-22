import numpy as np
import pandas as pd
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    equal_opportunity_difference,
)

# ── Manual implementations (show you understand the maths) ──

def demographic_parity(y_pred, sensitive):
    """
    Demographic Parity Difference = P(y=1|A=0) - P(y=1|A=1)
    Goal: difference as close to 0 as possible.
    """
    s = np.array(sensitive)
    y = np.array(y_pred)
    groups = np.unique(s)
    rates = {g: y[s == g].mean() for g in groups}
    vals = list(rates.values())
    return round(max(vals) - min(vals), 4)

def equal_opportunity(y_true, y_pred, sensitive):
    """
    Equal Opportunity Difference = TPR(A=0) - TPR(A=1)
    Ensures true positive rates are equal across groups.
    """
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    s   = np.array(sensitive)
    tprs = {}
    
    for g in np.unique(s):
        mask = (s == g) & (y_t == 1)
        tprs[g] = y_p[mask].mean() if mask.sum() > 0 else 0
        
    vals = list(tprs.values())
    return round(max(vals) - min(vals), 4)

def equalized_odds(y_true, y_pred, sensitive):
    """
    Max of (TPR diff, FPR diff) across groups.
    Stricter than Equal Opportunity — equalises both TPR and FPR.
    """
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    s   = np.array(sensitive)
    tprs, fprs = {}, {}
    
    for g in np.unique(s):
        pos_mask = (s == g) & (y_t == 1)
        neg_mask = (s == g) & (y_t == 0)
        tprs[g] = y_p[pos_mask].mean() if pos_mask.sum() > 0 else 0
        fprs[g] = y_p[neg_mask].mean() if neg_mask.sum() > 0 else 0
        
    tpr_vals = list(tprs.values())
    fpr_vals = list(fprs.values())
    
    tpr_diff = max(tpr_vals) - min(tpr_vals)
    fpr_diff = max(fpr_vals) - min(fpr_vals)
    return round(max(tpr_diff, fpr_diff), 4)

# ── Fairlearn-powered wrapper (production-grade) ───────────

def compute_all_metrics(y_true, y_pred, sensitive):
    """
    Returns dict with all four fairness metrics computed using Fairlearn
    where applicable, ensuring standardization for the framework.
    """
    s = np.array(sensitive)
    y_p = np.array(y_pred)
    
    # Calculate Disparate Impact safely
    rates = [y_p[s == g].mean() for g in np.unique(s)]
    di_max = max(rates)
    di_min = min(rates)
    di = round(di_min / di_max, 4) if di_max > 0 else 1.0

    return {
        "demographic_parity": demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive),
        "equal_opportunity": equal_opportunity_difference(y_true, y_pred, sensitive_features=sensitive),
        "equalized_odds": equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive),
        "disparate_impact": di,
    }

if __name__ == "__main__":
    from modules.data_loader import load_adult_dataset, encode_features, get_splits
    from modules.model_trainer import train_baseline

    print("Loading data...")
    df = encode_features(load_adult_dataset())
    X_tr, X_te, y_tr, y_te, s_tr, s_te = get_splits(df)

    print("Training baseline model...")
    model = train_baseline(X_tr, y_tr)
    
    print("Generating predictions...")
    y_pred = model.predict(X_te)
    
    print("\n--- Manual Fairness Metrics ---")
    print(f"Demographic Parity: {demographic_parity(y_pred, s_te)}")
    print(f"Equal Opportunity:  {equal_opportunity(y_te, y_pred, s_te)}")
    print(f"Equalized Odds:     {equalized_odds(y_te, y_pred, s_te)}")
    
    print("\n--- Fairlearn Wrapper Metrics ---")
    metrics = compute_all_metrics(y_te, y_pred, s_te)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")