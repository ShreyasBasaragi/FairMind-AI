import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from modules.model_trainer import train_baseline, train_random_forest, train_neural_net, evaluate_model
from modules.fairness_metrics import compute_all_metrics
from modules.mitigation import train_with_reweighting, train_adversarial_model, train_ai_arbiter
from modules.visualizer import FairnessVisualizer

class FairAIFramework:
    def __init__(self, sensitive_attr="sex", dataset_path="data/adult.csv"):
        self.sensitive_attr = sensitive_attr
        self.dataset_path = dataset_path
        self.results = {}
        self.models = {}
        self.viz = FairnessVisualizer()

    # loading data
    def load_data(self):
        print("[Framework] Loading default Adult dataset...")
        from modules.data_loader import load_adult_dataset, encode_features
        
        df = load_adult_dataset()
        df = encode_features(df)
        
        target_col = "income" if "income" in df.columns else df.columns[-1]
        
        if self.sensitive_attr in df.columns:
            sens_col = self.sensitive_attr
        else:
            sens_col = next((c for c in df.columns if self.sensitive_attr in c), df.columns[0])
        
        y = df[target_col]
        s = df[sens_col]
        X = df.drop(columns=[target_col, sens_col])
        
        self.X_tr, self.X_te, self.y_tr, self.y_te, self.s_tr, self.s_te = train_test_split(
            X, y, s, test_size=0.3, random_state=42
        )
        
        sc = StandardScaler()
        self.X_tr = sc.fit_transform(self.X_tr)
        self.X_te = sc.transform(self.X_te)

    def load_data_dynamically(self, df, target_col, sensitive_col):
        print(f"\n[Auto-Ingest] Processing custom dataset...")
        self.sensitive_attr = sensitive_col

        top_class = df[target_col].value_counts().idxmin() 
        df[target_col] = (df[target_col] == top_class).astype(int)

        approval_rates = df.groupby(self.sensitive_attr)[target_col].mean()
        privileged_group = approval_rates.idxmax()
        df[self.sensitive_attr] = (df[self.sensitive_attr] == privileged_group).astype(int)

        X = pd.get_dummies(df.drop(target_col, axis=1), drop_first=True)
        y = df[target_col]
        s = df[self.sensitive_attr]
        
        self.X_tr, self.X_te, self.y_tr, self.y_te, self.s_tr, self.s_te = train_test_split(
            X, y, s, test_size=0.3, random_state=42
        )
        sc = StandardScaler()
        self.X_tr = sc.fit_transform(self.X_tr)
        self.X_te = sc.transform(self.X_te)

    # evaluate
    def _evaluate_and_store(self, model, model_name, y_pred, display_name):
        # Calculate accuracy/performance
        perf = evaluate_model(self.y_te, y_pred)
        # Handle case if evaluate_model returns just a float instead of a dict
        if not isinstance(perf, dict):
            perf = {"accuracy": perf}
            
        # Calculate fairness metrics
        fairness = compute_all_metrics(self.y_te, y_pred, self.s_te)
        
        # Save to dashboard dictionary
        self.results[model_name] = {
            "performance": perf,
            "fairness": fairness,
            "display_name": display_name
        }
        self.models[model_name] = model
        print(f"   [{display_name}] Accuracy: {perf['accuracy']:.4f} | Fairness Score: {fairness.get('disparate_impact', 0):.4f}")

    def generate_report(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        out_dir = os.path.join(base_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            "sensitive_attribute": self.sensitive_attr,
            "models": {name: {"performance": r["performance"], "fairness": r["fairness"]} for name, r in self.results.items()}
        }
        
        path = os.path.join(out_dir, f"fairness_report_{ts}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
            
        try:
            # pass visualizer path
            self.viz.plot_all(self.results, out_dir=out_dir)
            print(f"[Framework] Visualizations successfully saved to {out_dir}")
        except Exception as e:
            print(f"\n[CRITICAL WARNING] Visualizer crashed and skipped making graphs! Error: {e}\n")
            
        return report

    # training models
    def train_biased_model(self):
        print("[Framework] Training Standard Baseline (Biased)...")
        model = train_baseline(self.X_tr, self.y_tr)
        self._evaluate_and_store(model, "baseline", model.predict(self.X_te), "Baseline (LR)")

    def train_additional_baselines(self):
        print("[Framework] Training Random Forest & Neural Net...")
        rf = train_random_forest(self.X_tr, self.y_tr)
        self._evaluate_and_store(rf, "random_forest", rf.predict(self.X_te), "Random Forest")
        
        nn = train_neural_net(self.X_tr, self.y_tr)
        self._evaluate_and_store(nn, "neural_net", nn.predict(self.X_te), "Neural Net")

    def apply_mitigation(self):
        print("[Framework] Applying Fair Mitigations...")
        try:
            m_rw = train_with_reweighting(self.X_tr, self.y_tr, self.s_tr)
            self._evaluate_and_store(m_rw, "reweighting", m_rw.predict(self.X_te), "Reweighting")
        except Exception as e:
            print(f"   [Skipped] Reweighting: {e}")

        try:
            m_adv = train_adversarial_model(self.X_tr, self.y_tr, self.s_tr)
            self._evaluate_and_store(m_adv, "adversarial", m_adv.predict(self.X_te), "Adversarial GAN")
        except Exception as e:
            print(f"   [Skipped] Adversarial: {e}")

    # main()
    def run(self):
        print("\n" + "=" * 60)
        print("  FairMind AI: Multi-Model Evaluation — Starting...")
        print("=" * 60)

        if not hasattr(self, 'X_tr'):
            self.load_data()

        self.train_biased_model()          
        self.train_additional_baselines()   
        self.apply_mitigation()           

        print("[Framework] Deploying the AI Arbiter Pipeline ")
        m_arbiter = train_ai_arbiter(self.X_tr, self.y_tr, self.s_tr)
        self._evaluate_and_store(m_arbiter, "ai_arbiter_pipeline", m_arbiter.predict(self.X_te), "AI Arbiter")
        
        self.generate_report()
        print("\n[Framework] Execution Complete.")

if __name__ == "__main__":
    fw = FairAIFramework(sensitive_attr="sex")
    fw.run()
