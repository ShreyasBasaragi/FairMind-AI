import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# The sensitive attributes defined in your project plan
SENSITIVE_ATTRS = ["sex", "race"]

def load_adult_dataset(path="data/adult.csv"):
    """Load the UCI Adult Income dataset and return a clean DataFrame."""
    cols = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
    # Read CSV, handle spaces, and flag '?' as NaN
    df = pd.read_csv(path, names=cols, na_values=" ?", skipinitialspace=True)
    df.dropna(inplace=True)
    
    # Binarize the target variable: 1 if >50K, 0 otherwise
    df["income"] = (df["income"].str.strip() == ">50K").astype(int)
    return df

def encode_features(df):
    """One-hot encode categorical columns, keeping sensitive columns intact."""
    le = LabelEncoder()
    sensitive_encoded = {}
    
    # Encode sensitive attributes separately
    for col in SENSITIVE_ATTRS:
        sensitive_encoded[col] = le.fit_transform(df[col])
        
    # One-hot encode all other categorical columns
    cat_cols = df.select_dtypes(include="object").columns.difference(SENSITIVE_ATTRS + ["income"])
    df_enc = pd.get_dummies(df, columns=cat_cols)
    
    # Re-insert the sensitive attributes
    for col, vals in sensitive_encoded.items():
        df_enc[col] = vals
        
    return df_enc

from sklearn.preprocessing import StandardScaler

def get_splits(df, sensitive_attr="sex", test_size=0.3, random_state=42):
    """Split data into train/test sets, separating out the sensitive attribute."""
    y = df["income"]
    X = df.drop(columns=["income"])
    
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    s_tr = X_tr[sensitive_attr].copy()
    s_te = X_te[sensitive_attr].copy()
    
    # Scale the features so gradient descent doesn't explode
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
    X_te = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)
    
    return X_tr, X_te, y_tr, y_te, s_tr, s_te

if __name__ == "__main__":
    # Quick execution test
    print("Loading data...")
    raw_df = load_adult_dataset()
    encoded_df = encode_features(raw_df)
    X_train, X_test, y_train, y_test, s_train, s_test = get_splits(encoded_df)
    print(f"Data successfully loaded and split!")
    print(f"Training features shape: {X_train.shape}")