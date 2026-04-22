import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

class CustomFairLogisticRegression:
    """
    Custom in-processing mitigation: Logistic Regression with a 
    Demographic Parity penalty embedded directly in the gradient descent.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_penalty=0.8):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_penalty = lambda_penalty
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        z = np.array(z, dtype=np.float64)
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, sensitive_attr):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).flatten()
        s = np.array(sensitive_attr, dtype=np.float64).flatten()
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        # Create masks for the sensitive groups
        mask_1 = (s == 1)
        mask_0 = (s == 0)
        n_1 = np.sum(mask_1)
        n_0 = np.sum(mask_0)

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # 1. Standard Binary Cross-Entropy (BCE) Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # 2. Fairness Penalty Gradients (Demographic Parity proxy)
            if n_1 > 0 and n_0 > 0:
                mean_1 = np.mean(y_pred[mask_1])
                mean_0 = np.mean(y_pred[mask_0])
                diff = mean_1 - mean_0
                
                # Derivative of the penalty w.r.t predictions
                dp_dy = np.zeros(n_samples)
                dp_dy[mask_1] = 2 * self.lambda_penalty * diff * (1 / n_1)
                dp_dy[mask_0] = 2 * self.lambda_penalty * diff * (-1 / n_0)
                
                # Chain rule to get gradients w.r.t weights and bias
                derivative_sigmoid = y_pred * (1 - y_pred)
                dw_fair = np.dot(X.T, dp_dy * derivative_sigmoid)
                db_fair = np.sum(dp_dy * derivative_sigmoid)

                # Add fairness gradients to BCE gradients
                dw += dw_fair
                db += db_fair

            # 3. Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
        return self

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# ── Wrapper functions for the Framework ───────────────────────────

def train_custom_fair_model(X_train, y_train, sensitive_train, lambda_penalty=0.8):
    """Train the custom model with the math-based fairness constraint."""
    model = CustomFairLogisticRegression(lambda_penalty=lambda_penalty)
    model.fit(X_train, y_train, sensitive_train)
    return model

def compute_sample_weights(y_train, sensitive_train):
    """Pre-processing mitigation: Reweighting."""
    df = pd.DataFrame({"y": np.array(y_train), "s": np.array(sensitive_train)})
    
    # Add a dummy column specifically for counting to avoid KeyError
    df["dummy_counter"] = 1
    group_counts = df.groupby(["s", "y"])["dummy_counter"].transform("count")
    
    # Calculate weights: total_samples / (num_groups * group_count)
    weights = len(df) / (df.groupby(["s","y"]).ngroups * group_counts)
    return weights.values

def train_with_reweighting(X_train, y_train, sensitive_train):
    weights = compute_sample_weights(y_train, sensitive_train)
    model = LogisticRegression(max_iter=300, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)
    return model

def train_exponentiated_gradient(X_train, y_train, sensitive_train, constraint="dp"):
    """In-processing mitigation using Fairlearn."""
    base = LogisticRegression(max_iter=300, solver="saga")
    constr = DemographicParity() if constraint == "dp" else EqualizedOdds()
    # eps=0.05 added to speed up computation on i5 processors
    mitigator = ExponentiatedGradient(estimator=base, constraints=constr, eps=0.05)
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
    return mitigator

def train_threshold_optimizer(base_model, X_train, y_train, sensitive_train):
    """Post-processing mitigation using Fairlearn."""
    optimizer = ThresholdOptimizer(
        estimator=base_model,
        constraints="equalized_odds",
        predict_method="predict_proba",
        objective="balanced_accuracy_score"
    )
    optimizer.fit(X_train, y_train, sensitive_features=sensitive_train)
    return optimizer

class AdversarialDebiaser(BaseEstimator, ClassifierMixin):
    """
    Advanced AI Implementation: Competitive Adversarial Learning.
    A Predictor learns to classify income while a Multi-Layer 
    Adversary attempts to recover sensitive information.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, alpha=1.5):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.alpha = alpha  # Strength of the adversary's influence
        self.w_pred = None
        self.b_pred = 0
        self.w_adv_h = None # Adversary hidden layer
        self.b_adv_h = 0
        self.w_adv_o = None # Adversary output layer
        self.b_adv_o = 0
        self.classes_ = np.array([0, 1])

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y, s):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).reshape(-1, 1)
        s = np.array(s, dtype=np.float64).reshape(-1, 1)
        n_samples, n_features = X.shape

        # Init Predictor
        self.w_pred = np.zeros((n_features, 1))
        self.b_pred = 0
        
        # Init Adversary (1 input -> 4 hidden units -> 1 output)
        self.w_adv_h = np.random.randn(1, 4) * 0.01
        self.b_adv_h = np.zeros((1, 4))
        self.w_adv_o = np.random.randn(4, 1) * 0.01
        self.b_adv_o = 0

        for i in range(self.n_iterations):
            # --- 1. ADVERSARIAL WARMUP ---
            # Slowly increase alpha from 0 to self.alpha over the first 500 iterations
            # This prevents the adversary from collapsing the model early on.
            current_alpha = self.alpha * min(1.0, i / 500)

            # --- 2. FORWARD PASS ---
            y_hat = self._sigmoid(np.dot(X, self.w_pred) + self.b_pred)
            adv_hidden = self._sigmoid(np.dot(y_hat, self.w_adv_h) + self.b_adv_h)
            s_hat = self._sigmoid(np.dot(adv_hidden, self.w_adv_o) + self.b_adv_o)

            # --- 3. GRADIENTS ---
            # Adversary BCE Gradients (The Spy trying to catch bias)
            ds_hat = (s_hat - s)
            dw_adv_o = np.dot(adv_hidden.T, ds_hat) / n_samples
            db_adv_o = np.sum(ds_hat) / n_samples
            
            d_adv_hidden = np.dot(ds_hat, self.w_adv_o.T) * (adv_hidden * (1 - adv_hidden))
            dw_adv_h = np.dot(y_hat.T, d_adv_hidden) / n_samples
            db_adv_h = np.sum(d_adv_hidden, axis=0) / n_samples

            # Predictor Gradients (The Brain trying to be accurate and fair)
            dy_hat_base = (y_hat - y)
            dy_hat_adv = np.dot(d_adv_hidden, self.w_adv_h.T) 
            
            # The Minimax Equation: Loss = Performance - (Alpha * Fairness_Penalty)
            dy_total = dy_hat_base - (current_alpha * dy_hat_adv)
            
            dw_pred = np.dot(X.T, dy_total * y_hat * (1 - y_hat)) / n_samples
            db_pred = np.sum(dy_total * y_hat * (1 - y_hat)) / n_samples

            # --- 4. GRADIENT CLIPPING ---
            # We "clip" the gradients to a max value of 1.0 to prevent 
            # the weights from exploding and causing a 0.000 Disparate Impact.
            clip_val = 1.0
            dw_pred = np.clip(dw_pred, -clip_val, clip_val)
            dw_adv_h = np.clip(dw_adv_h, -clip_val, clip_val)
            dw_adv_o = np.clip(dw_adv_o, -clip_val, clip_val)

            # --- 5. PARAMETER UPDATES ---
            self.w_pred -= self.learning_rate * dw_pred
            self.b_pred -= self.learning_rate * db_pred
            self.w_adv_h -= self.learning_rate * dw_adv_h
            self.b_adv_h -= self.learning_rate * db_adv_h
            self.w_adv_o -= self.learning_rate * dw_adv_o
            self.b_adv_o -= self.learning_rate * db_adv_o
            
        return self

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        p = self._sigmoid(np.dot(X, self.w_pred) + self.b_pred).flatten()
        return np.vstack((1 - p, p)).T

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def train_adversarial_model(X_train, y_train, s_train, alpha=1.5):
    """
    Wrapper function to initialize and fit the AdversarialDebiaser.
    This is what the framework imports.
    """
    model = AdversarialDebiaser(alpha=alpha)
    model.fit(X_train, y_train, s_train)
    return model

from fairlearn.reductions import ExponentiatedGradient, DemographicParity

def train_reduction_model(X_train, y_train, sensitive_train):
    """
    Algorithm: Exponentiated Gradient Reduction.
    An advanced AI meta-learner that treats fairness as a 
    mathematical constraint during optimization.
    """
    base_model = LogisticRegression(solver='liblinear')
    # This reduction algorithm iteratively solves a series of 
    # weighted classification problems to find the fair equilibrium.
    mitigator = ExponentiatedGradient(
        base_model, 
        constraints=DemographicParity()
    )
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
    return mitigator

class CorrelationScrubber(BaseEstimator, ClassifierMixin):
    """
    Algorithm: Geometric Orthogonal Projection.
    An AI preprocessing step that projects data into a 'Fair Manifold' 
    where the sensitive attribute has zero linear correlation.
    """
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.projection_matrix = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, s):
        # Convert to numpy and ensure correct shapes
        X = np.array(X, dtype=np.float64)
        s = np.array(s, dtype=np.float64).reshape(-1, 1)
        n_samples, n_features = X.shape
        
        # 1. Find the Bias Direction (v) in Feature Space
        # Calculate how each of the 103 features correlates with the sensitive attribute
        s_feature_correlation = np.dot(X.T, s) / n_samples
        
        # 2. Normalize the Bias Vector
        norm = np.linalg.norm(s_feature_correlation)
        v = s_feature_correlation / norm if norm > 1e-9 else s_feature_correlation

        # 3. Create the Projection Matrix (P = I - vv^T)
        # This yields a (103, 103) matrix that "filters" the bias direction
        self.projection_matrix = np.eye(n_features) - np.dot(v, v.T)
        
        # 4. Transform X to the "Fair Manifold" and train
        X_fair = np.dot(X, self.projection_matrix)
        self.model.fit(X_fair, y)
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        X_fair = np.dot(X, self.projection_matrix)
        return self.model.predict(X_fair)

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        X_fair = np.dot(X, self.projection_matrix)
        return self.model.predict_proba(X_fair)
    
# --- 5. The True "AI Correcting AI" Pipeline ---
# --- 5. The "Utopian Arbiter" (Advanced AI Correcting AI) ---
class Utopian_Arbiter_Pipeline(BaseEstimator, ClassifierMixin):
    def __init__(self):
        # AI #1: The Biased Oracle
        self.oracle = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        # AI #2: The Utopian Neural Network
        self.arbiter = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, s):
        X_arr, y_arr, s_arr = np.array(X), np.array(y), np.array(s).reshape(-1)
        
        # 1. Oracle learns the biased world
        self.oracle.fit(X_arr, y_arr)
        probs = self.oracle.predict_proba(X_arr)[:, 1]
        
        # 2. The Arbiter simulates a "Utopian" reality by relabeling the data
        y_utopian = y_arr.copy()
        
        priv_mask = (s_arr == 1)
        unpriv_mask = (s_arr == 0)
        
        priv_pos = np.sum(y_arr[priv_mask] == 1)
        unpriv_pos = np.sum(y_arr[unpriv_mask] == 1)
        
        # Target rate is perfect Demographic Parity
        target_rate = (priv_pos / np.sum(priv_mask) + unpriv_pos / np.sum(unpriv_mask)) / 2.0
        target_priv_pos = int(target_rate * np.sum(priv_mask))
        target_unpriv_pos = int(target_rate * np.sum(unpriv_mask))
        
        # Promote unprivileged (Give them the highest probability borderline rejections)
        if unpriv_pos < target_unpriv_pos:
            needed = target_unpriv_pos - unpriv_pos
            candidates = np.where(unpriv_mask & (y_arr == 0))[0]
            candidates_sorted = candidates[np.argsort(probs[candidates])[::-1]]
            y_utopian[candidates_sorted[:needed]] = 1
            
        # Demote privileged (Remove the lowest confidence acceptances)
        if priv_pos > target_priv_pos:
            needed = priv_pos - target_priv_pos
            candidates = np.where(priv_mask & (y_arr == 1))[0]
            candidates_sorted = candidates[np.argsort(probs[candidates])]
            y_utopian[candidates_sorted[:needed]] = 0

        # 3. Train the Deep Neural Network on this new perfectly fair reality
        self.arbiter.fit(X_arr, y_utopian)
        return self

    def predict(self, X):
        return self.arbiter.predict(np.array(X))

# Wrapper for the framework
def train_ai_arbiter(X, y, s):
    return Utopian_Arbiter_Pipeline().fit(X, y, s)

if __name__ == "__main__":
    from modules.data_loader import load_adult_dataset, encode_features, get_splits
    from modules.fairness_metrics import compute_all_metrics

    print("Loading scaled data...")
    df = encode_features(load_adult_dataset())
    X_tr, X_te, y_tr, y_te, s_tr, s_te = get_splits(df)

    print("Training Custom Fair Model (In-processing)...")
    fair_model = train_custom_fair_model(X_tr, y_tr, s_tr, lambda_penalty=1.5)
    y_pred_fair = fair_model.predict(X_te)
    
    print("\n--- Fairness Metrics (Custom Mitigated Model) ---")
    metrics = compute_all_metrics(y_te, y_pred_fair, s_te)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")