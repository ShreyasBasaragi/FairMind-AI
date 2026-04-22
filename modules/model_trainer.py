import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

class CustomLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    A from-scratch implementation of Logistic Regression using pure NumPy.
    This serves as the raw, biased baseline model.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        # Ensure z is a float to prevent type errors, then clip to prevent overflow
        z = np.array(z, dtype=np.float64)
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Force inputs to be strictly float arrays to handle Pandas booleans
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).flatten()
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent Loop
        for _ in range(self.n_iterations):
            # Forward pass: Calculate linear model and apply activation
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Backward pass: Calculate gradients (Binary Cross-Entropy derivative)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
        # Tell Scikit-Learn this model is officially "fitted"
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        # Force inputs to be strictly float arrays
        X = np.array(X, dtype=np.float64)
        linear_model = np.dot(X, self.weights) + self.bias
        
        prob_positive = self._sigmoid(linear_model)
        prob_negative = 1 - prob_positive
        
        # Stack into a 2D array: Col 0 is negative prob, Col 1 is positive prob
        return np.vstack((prob_negative, prob_positive)).T

    def predict(self, X, threshold=0.5):
        # Grab just the positive class probabilities (column index 1)
        probs = self.predict_proba(X)[:, 1]
        y_predicted_cls = probs >= threshold
        return y_predicted_cls.astype(int)

def train_baseline(X_train, y_train):
    """Train the custom Logistic Regression — no fairness constraints."""
    model = CustomLogisticRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(y_test, y_pred, label="Baseline"):
    """Return a dict of standard performance metrics."""
    acc  = accuracy_score(y_test, y_pred)
    rep  = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return {
        "label":     label,
        "accuracy":  acc,
        "precision": rep["1"]["precision"],
        "recall":    rep["1"]["recall"],
        "f1":        rep["1"]["f1-score"],
    }
    
def train_random_forest(X_train, y_train):
    """A Non-linear Ensemble model to compare bias in tree structures."""
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_neural_net(X_train, y_train):
    """A Deep Learning baseline (Multi-layer Perceptron)."""
    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    from modules.data_loader import load_adult_dataset, encode_features, get_splits
    
    print("Loading data for model testing...")
    df = encode_features(load_adult_dataset())
    X_tr, X_te, y_tr, y_te, s_tr, s_te = get_splits(df)
    
    print("Training custom baseline model (this might take a few seconds)...")
    model = train_baseline(X_tr, y_tr)
    
    print("Evaluating model...")
    # Updated to pass y_pred directly
    results = evaluate_model(y_te, model.predict(X_te))
    print(f"Baseline Accuracy: {results['accuracy']:.4f}")