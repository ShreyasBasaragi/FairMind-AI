import pandas as pd
import numpy as np

def detect_bias(y_pred, sensitive_series, label="Group"):
    """
    For each unique value in sensitive_series, compute:
    - positive_rate  : fraction predicted as positive
    Returns a DataFrame sorted by positive_rate.
    """
    s = pd.Series(sensitive_series).reset_index(drop=True)
    y = pd.Series(y_pred).reset_index(drop=True)
    results = []
    
    for group in s.unique():
        mask = (s == group)
        pos_rate = y[mask].mean()
        count    = mask.sum()
        results.append({
            label: group, 
            "count": count,
            "positive_rate": round(pos_rate, 4)
        })
        
    df = pd.DataFrame(results).sort_values("positive_rate", ascending=False)
    return df

def compute_disparate_impact_ratio(bias_df, label="Group"):
    """
    Disparate Impact = min_group_rate / max_group_rate.
    A value < 0.8 is considered discriminatory (80% rule).
    """
    rates = bias_df["positive_rate"]
    # Prevent division by zero if a model predicts 0 positives
    if rates.max() == 0:
        return 1.0
        
    di = rates.min() / rates.max()
    return round(di, 4)

def is_biased(di_ratio, threshold=0.8):
    """Return True if model fails the 80% Disparate Impact test."""
    return di_ratio < threshold

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
    
    print("\n--- Bias Detection Results (Sex) ---")
    # Note: s_te contains the encoded sensitive attribute (0 and 1)
    bias_df = detect_bias(y_pred, s_te, label="sex_encoded")
    print(bias_df.to_string(index=False))
    
    di = compute_disparate_impact_ratio(bias_df)
    print(f"\nDisparate Impact Ratio: {di}")
    print(f"Is Biased (< 0.8)? {is_biased(di)}")