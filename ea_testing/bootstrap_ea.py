'''
import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt



# Append path to import run_all from backend script
sys.path.append("/home/insybio/Downloads/backend-application/insybio-biomarkers/05.Testing_Multibiomarker_Predictive_Analytics_Model/")
from testing_multibiomarker_predictive_analytics_model_backend import run_all

# === CONFIGURATION ===
test_data_file = "/home/insybio/Documents/BioreactorOptimization/bioreactor/github/datasets/testing/ea_data/preprocessed_dataset_testing.csv"
test_labels_file = "/home/insybio/Documents/BioreactorOptimization/bioreactor/github/datasets/testing/ea_data/area_test_labels.txt"
model_zip = "/home/insybio/Documents/BioreactorOptimization/bioreactor/github/models/ea/area_2/models_1.zip"
features_file = "/home/insybio/Documents/BioreactorOptimization/bioreactor/github/models/ea/area_2/features_list.txt"
training_labels_file = "/home/insybio/Documents/BioreactorOptimization/bioreactor/github/models/ea/area_2/training_labels.txt"
length_features_training = "/home/insybio/Documents/BioreactorOptimization/bioreactor/github/models/ea/area_2/length_of_features_from_training.txt"
output_folder = "/home/insybio/Downloads/Output_folder_testing_area_2/bootstrapped_runs/"
n_iterations = 100
bootstrap_ratio = 0.9

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Load full dataset (includes feature names in first column)
X_full = pd.read_csv(test_data_file, header=None)
features = X_full.iloc[:, 0]       # Feature names (first column)
data = X_full.iloc[:, 1:]          # Sample values (remaining columns)

# Load labels (assumed to be in a row, tab-separated)
y = pd.read_csv(test_labels_file, sep="\t", header=None).values.flatten()
n_samples = len(y)

# Store metrics
all_rmse = []

# Start bootstrapping
for i in range(n_iterations):
    np.random.seed(i)
    indices = np.random.choice(n_samples, size=int(n_samples * bootstrap_ratio), replace=True)
    
    sampled_data = data.iloc[:, indices]
    sampled_labels = y[indices]

    # Reconstruct bootstrapped dataset
    boot_df = pd.concat([features, sampled_data], axis=1)

    # Save dataset
    tmp_data = f"{output_folder}boot_data_{i}.csv"
    boot_df.to_csv(tmp_data, sep=",", index=False, header=False)

    # Save labels in one line
    tmp_labels = f"{output_folder}boot_labels_{i}.txt"
    with open(tmp_labels, 'w') as f:
        f.write("\t".join([f"{val:.4f}" for val in sampled_labels]))

    # Output directory for this run
    out_dir = os.path.join(output_folder, f"run_{i}/")
    os.makedirs(out_dir, exist_ok=True)

    # Run model
    result = run_all(
        tmp_data, tmp_labels, "", "", "", features_file,
        2, 1, model_zip, 1, 1, "", 8, 1, 0,
        training_labels_file, "", length_features_training,
        out_dir, "", thread_num=2
    )

    print(f"Run {i + 1}: {result}")

    # Try reading RMSE from metrics
    try:
        with open(os.path.join(out_dir, "metrics.txt")) as f:
            for line in f:
                if "Root Mean Square Error" in line:
                    rmse = float(line.split(":")[1].strip())
                    all_rmse.append(rmse)
                    break
    except FileNotFoundError:
        all_rmse.append(None)

# Summary
rmse_valid = [v for v in all_rmse if v is not None]
print(f"\nBootstrapping completed ({len(rmse_valid)} valid runs).")
if rmse_valid:
    print(f"Average RMSE: {np.mean(rmse_valid):.4f} ± {np.std(rmse_valid):.4f}")
else:
    print("No valid runs. Please check logs for errors.")

r2_list = []
spearman_list = []
pvals = []
rae_list = []
rmse_list = []

for i in range(n_iterations):
    out_dir = os.path.join(output_folder, f"run_{i}/")
    pred_file = os.path.join(out_dir, "result_labels.txt")
    label_file = f"{output_folder}boot_labels_{i}.txt"

    if not os.path.exists(pred_file):
        continue

    # Load predicted and true labels
    with open(pred_file) as f:
        pred = [float(x) for x in f.read().strip().split("\t") if x]

    with open(label_file) as f:
        true = [float(x) for x in f.read().strip().split("\t") if x]

    if len(pred) != len(true):
        continue

    # Metrics
    r2_list.append(r2_score(true, pred))
    rho, p = spearmanr(true, pred)
    spearman_list.append(rho)
    pvals.append(p)
    rmse_list.append(np.sqrt(mean_squared_error(true, pred)))

    mean_true = np.mean(true)
    rae = sum(abs(np.array(pred) - np.array(true))) / sum(abs(np.array(true) - mean_true))
    rae_list.append(rae * 100)

# Final summary table
print("\nBootstrapping completed with detailed metrics:\n")
print(f"R²:      {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
print(f"Spearman: {np.mean(spearman_list):.4f} ± {np.std(spearman_list):.4f}")
print(f"P-value:  {np.mean(pvals):.2e}")
print(f"RAE:      {np.mean(rae_list):.2f}% ± {np.std(rae_list):.2f}%")
print(f"RMSE:     {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")


print("True:", true[:10])
print("Pred:", pred[:10])

# Plot predictions vs. true labels from the first successful run
for i in range(n_iterations):
    out_dir = os.path.join(output_folder, f"run_{i}/")
    pred_file = os.path.join(out_dir, "result_labels.txt")
    label_file = f"{output_folder}boot_labels_{i}.txt"

    if not os.path.exists(pred_file):
        continue

    with open(pred_file) as f:
        pred = [float(x) for x in f.read().strip().split("\t") if x]

    with open(label_file) as f:
        true = [float(x) for x in f.read().strip().split("\t") if x]

    if len(pred) != len(true):
        continue

    # Make the plot
    plt.figure(figsize=(6, 6))
    plt.scatter(true, pred, alpha=0.6, edgecolors='k')
    plt.plot([min(true), max(true)], [min(true), max(true)], 'r--', label='Ideal (y = x)')
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")
    plt.title(f"Predicted vs. True Labels (Run {i})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    break  # only show the first valid run

'''
###################################################################3
#####################################################################

import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

# Append backend module path
sys.path.append("/home/insybio/Downloads/backend-application/insybio-biomarkers/05.Testing_Multibiomarker_Predictive_Analytics_Model/")
from testing_multibiomarker_predictive_analytics_model_backend import run_all

# === CONFIGURATION ===
test_data_file = "/home/insybio/Documents/BioreactorOptimization/bioreactor/github/datasets/testing/ea_data/preprocessed_dataset_testing.csv"
test_labels_file = "/home/insybio/Documents/BioreactorOptimization/bioreactor/github/datasets/testing/ea_data/porosity_test_labels.txt"
model_zip = "/home/insybio/Documents/BioreactorOptimization/bioreactor/github/models/ea/porosity_3/models_1.zip"
features_file = "/home/insybio/Documents/BioreactorOptimization/bioreactor/github/models/ea/porosity_3/features_list.txt"
training_labels_file = "/home/insybio/Documents/BioreactorOptimization/bioreactor/github/models/ea/porosity_3/training_labels.txt"
length_features_training = "/home/insybio/Documents/BioreactorOptimization/bioreactor/github/models/ea/porosity_3/length_of_features_from_training.txt"
output_folder = "/home/insybio/Downloads/Output_folder_testing_porosity_3/bootstrapped_runs/"
n_iterations = 100
bootstrap_ratio = 0.9

# Load data
X_full = pd.read_csv(test_data_file, header=None)
features = X_full.iloc[:, 0]
data = X_full.iloc[:, 1:]
y = pd.read_csv(test_labels_file, sep="\t", header=None).values.flatten()
train_labels = pd.read_csv(training_labels_file, sep="\t", header=None).values.flatten()
min_label, max_label = np.min(train_labels), np.max(train_labels)
n_samples = len(y)

# Create output dir
os.makedirs(output_folder, exist_ok=True)

# Metrics
r2_list, spearman_list, pvals, rae_list, rmse_list = [], [], [], [], []

# Bootstrap
for i in range(n_iterations):
    np.random.seed(i)
    idx = np.random.choice(n_samples, int(n_samples * bootstrap_ratio), replace=True)
    sampled_data = data.iloc[:, idx]
    sampled_labels = y[idx]

    # Save bootstrapped data
    boot_df = pd.concat([features, sampled_data], axis=1)
    tmp_data = f"{output_folder}boot_data_{i}.csv"
    boot_df.to_csv(tmp_data, sep=",", index=False, header=False)
    tmp_labels = f"{output_folder}boot_labels_{i}.txt"
    with open(tmp_labels, 'w') as f:
        f.write("\t".join([f"{val:.4f}" for val in sampled_labels]))

    out_dir = os.path.join(output_folder, f"run_{i}/")
    os.makedirs(out_dir, exist_ok=True)

    run_all(tmp_data, tmp_labels, "", "", "", features_file,
            2, 1, model_zip, 1, 1, "", 8, 1, 0,
            training_labels_file, "", length_features_training,
            out_dir, "", thread_num=2)

    pred_file = os.path.join(out_dir, "result_labels.txt")
    if not os.path.exists(pred_file):
        continue

    # Load predictions and rescale
    with open(pred_file) as f:
        pred = [float(x) for x in f.read().strip().split("\t") if x]
    #pred = [p * (max_label - min_label) + min_label for p in pred]

    # Load true labels
    with open(tmp_labels) as f:
        true = [float(x) for x in f.read().strip().split("\t") if x]

    if len(pred) != len(true):
        continue

    # Metrics
    r2_list.append(r2_score(true, pred))
    rho, p = spearmanr(true, pred)
    spearman_list.append(rho)
    pvals.append(p)
    rmse_list.append(np.sqrt(mean_squared_error(true, pred)))
    rae = sum(abs(np.array(pred) - np.array(true))) / sum(abs(np.array(true) - np.mean(true)))
    rae_list.append(rae * 100)

print("Predicted min/max:", min(pred), max(pred))
print("True labels min/max:", min(true), max(true))

# Final report
print("\nBootstrapping completed with detailed metrics:\n")
print(f"R²:       {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
print(f"Spearman: {np.mean(spearman_list):.4f} ± {np.std(spearman_list):.4f}")
print(f"P-value:  {np.mean(pvals):.2e}")
print(f"RAE:      {np.mean(rae_list):.2f}% ± {np.std(rae_list):.2f}%")
print(f"RMSE:     {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")

#################3
'''
import pandas as pd
import numpy as np

def load_labels(path, normalized=False, min_val=None, max_val=None):
    labels = pd.read_csv(path, header=None, sep="\t").values.flatten()
    if normalized and min_val is not None and max_val is not None:
        labels = labels * (max_val - min_val) + min_val
    return labels

def check_feature_alignment(train_features_path, test_data_path):
    train_features = pd.read_csv(train_features_path, header=None).values.flatten().tolist()
    test_data = pd.read_csv(test_data_path, header=None)
    test_features = test_data.iloc[:, 0].tolist()
    if train_features != test_features:
        missing = set(train_features) - set(test_features)
        extra = set(test_features) - set(train_features)
        print("❌ Feature mismatch!")
        if missing:
            print(f" - Missing in test data: {missing}")
        if extra:
            print(f" - Extra in test data: {extra}")
    else:
        print("✅ Features match.")

def check_label_range(train_labels_path, test_labels_path):
    train = pd.read_csv(train_labels_path, header=None, sep="\t").values.flatten()
    test = pd.read_csv(test_labels_path, header=None, sep="\t").values.flatten()
    print(f"Train label range: {train.min():.4f} to {train.max():.4f}")
    print(f"Test label range: {test.min():.4f} to {test.max():.4f}")
    if (test.min() < train.min() or test.max() > train.max()):
        print("⚠️  Test labels are outside training label range — may hurt generalization.")
    else:
        print("✅ Test labels are within training label range.")

def check_rescaling_applied(preds, train_labels_path):
    train = pd.read_csv(train_labels_path, header=None, sep="\t").values.flatten()
    min_label, max_label = np.min(train), np.max(train)
    rescaled = np.array(preds) * (max_label - min_label) + min_label
    print(f"Rescaled predictions range: {rescaled.min():.4f} to {rescaled.max():.4f}")
    return rescaled

# === CONFIGURE ===
train_features_file = "/home/insybio/Documents/BioreactorOptimization/bioreactor/github/models/ea/thickness_3/features_list.txt"
test_data_file = "/home/insybio/Documents/BioreactorOptimization/bioreactor/github/datasets/testing/ea_data/preprocessed_dataset_testing.csv"
train_labels_file = "/home/insybio/Documents/BioreactorOptimization/bioreactor/github/datasets/ea_data/thickness_labels.txt"
test_labels_file = "/home/insybio/Documents/BioreactorOptimization/bioreactor/github/datasets/testing/ea_data/thickness_test_labels.txt"

# === RUN CHECKS ===
check_feature_alignment(train_features_file, test_data_file)
check_label_range(train_labels_file, test_labels_file)
'''