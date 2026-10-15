import matplotlib.pyplot as plt
import os
import json
from pathlib import Path

# project root (parent of src/)
ROOT = Path(__file__).resolve().parents[1]


def model_comparison() -> None:
    names_values = []
    log_dir = str(ROOT / 'logs')

    for item in os.listdir(log_dir):
        if item.startswith('.'):
            continue

        exp_path = os.path.join(log_dir, item)
        
        if os.path.isdir(exp_path):
            path = os.path.join(exp_path, 'best_trial.json')
            
            if os.path.exists(path):
                
                with open(path, 'r', encoding='utf-8') as file:
                    history = json.load(file)

                base_name = item 
                
                value = history['best_r2']
                trial_num = history['trial_number']
                
                name = f"{base_name} (Trial {trial_num})"
                
                names_values.append((name, value))
                
                
    sorted_data = sorted(names_values, key=lambda x: x[1], reverse=True)
    
    sorted_names = [item[0] for item in sorted_data]
    sorted_values = [item[1] for item in sorted_data]

    plt.figure(figsize=(8, 5))
    plt.barh(sorted_names[::-1], sorted_values[::-1])
    plt.xlabel('R^2')
    plt.ylabel('Experiment name')
    plt.title('Model Comparison (Best R^2)')
    plt.tight_layout()
    plt.grid(color='grey', linestyle='--', linewidth=0.5, axis='x')
    
    os.makedirs(ROOT / 'plots', exist_ok=True)
    plt.savefig(ROOT / 'plots' / 'ModelComparison_BestR2_per_Experiment.png')
    plt.show()
    plt.close()


def plot_loss_function(train_loss, test_loss, experiment_path) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Train Loss')      
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Function for {os.path.basename(experiment_path)}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = os.path.join(experiment_path, "loss.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def generate_best_trial_plots() -> None:
    log_dir = str(ROOT / 'logs')
    
    if not os.path.exists(log_dir):
        print(f"ERROR: Directory '{log_dir}' does not exist.")
        return

    for item in os.listdir(log_dir):
        if item.startswith('.'):
            continue

        experiment_dir = os.path.join(log_dir, item)
        
        if os.path.isdir(experiment_dir):
            best_trial_path = os.path.join(experiment_dir, 'best_trial.json')
            
            if os.path.exists(best_trial_path):
                
                try:
                    with open(best_trial_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    print(f"JSON DECODE ERROR: Invalid JSON format in: {best_trial_path}")
                    continue
                
                if 'trial_number' not in data:
                    print(f"KEY ERROR: Missing key 'trial_number' in {best_trial_path}")
                    continue
                
                trial_num = data['trial_number']
                
                trial_results_dir = os.path.join(experiment_dir, f'trial_{trial_num}')
                results_path = os.path.join(trial_results_dir, 'results.json')

                if not os.path.exists(results_path):
                    print(f"SKIPPED: Missing results.json for the best trial in: {trial_results_dir}")
                    continue
                    
                try:
                    with open(results_path, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                except json.JSONDecodeError:
                    print(f"JSON DECODE ERROR: Invalid JSON format in: {results_path}")
                    continue

                if 'train_loss' not in results or 'test_loss' not in results:
                    print(f"KEY ERROR: Missing 'train_loss' or 'test_loss' in {results_path}")
                    continue
                    
                train_loss = results['train_loss']
                test_loss = results['test_loss']

                plot_loss_function(train_loss, test_loss, experiment_dir)
                print(f"GENERATED loss plot for experiment: {item}")


def plot_r2(r2, experiment_name='regression') -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='R^2')         
    plt.title('R^2')
    plt.xlabel('Epochs')
    plt.ylabel('R^2')
    
    # saving
    os.makedirs(ROOT / 'logs' / experiment_name, exist_ok=True)
    plot_path = ROOT / 'logs' / experiment_name / 'loss.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()