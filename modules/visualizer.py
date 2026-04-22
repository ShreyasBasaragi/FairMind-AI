import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

class FairnessVisualizer:
    def plot_all(self, results, out_dir="outputs"):
        os.makedirs(out_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        
        # ---------------------------------------------------------
        # 1. Plot the Accuracy vs. Fairness Trade-off (Scatter)
        # ---------------------------------------------------------
        plt.figure(figsize=(8, 6))
        
        models = []
        accs = []
        fairs = []
        
        for name, r in results.items():
            models.append(r.get("display_name", name.upper()))
            
            # Safely extract accuracy
            perf = r["performance"]
            acc = perf["accuracy"] if isinstance(perf, dict) else getattr(perf, "accuracy", perf)
            accs.append(acc)
            
            # Safely extract fairness
            fairs.append(r["fairness"].get("disparate_impact", 0))

        # Create scatter plot
        sns.scatterplot(x=fairs, y=accs, hue=models, s=200, palette="deep")
        
        plt.title("The Fairness-Accuracy Trade-off", fontsize=14, fontweight='bold')
        plt.xlabel("Fairness Score (Disparate Impact - Closer to 1.0 is better)")
        plt.ylabel("Model Accuracy")
        
        # Draw the ideal fairness threshold line
        plt.axvline(x=0.8, color='red', linestyle='--', label='80% Fairness Rule (Legal Standard)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save Trade-off Plot
        plt.savefig(os.path.join(out_dir, "accuracy_vs_fairness.png"), bbox_inches="tight")
        plt.close()

        # ---------------------------------------------------------
        # 2. Plot Individual Bias Profiles (Bar Graphs)
        # ---------------------------------------------------------
        for name, r in results.items():
            plt.figure(figsize=(6, 4))
            
            # Extract metrics, ensure they are numbers
            metrics = {k.replace("_", " ").title(): v for k, v in r["fairness"].items() if isinstance(v, (int, float))}
            
            if metrics:
                keys = list(metrics.keys())
                vals = list(metrics.values())
                
                # Create Bar Graph
                sns.barplot(x=keys, y=vals, palette="viridis")
                
                plt.title(f"{r.get('display_name', name.upper())}: Bias Profile")
                plt.ylabel("Metric Score")
                plt.axhline(y=0, color='black', linewidth=1)
                
                # Rotate X-axis labels so they don't overlap
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Save Individual Plot
                safe_name = name.replace(" ", "_").lower()
                plt.savefig(os.path.join(out_dir, f"{safe_name}_bias_profile.png"))
            
            plt.close()