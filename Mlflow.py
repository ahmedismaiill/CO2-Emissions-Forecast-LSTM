import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load your test data
# ----------------------------
X_test = np.load("models/X_test.npy")  
y_test = np.load("models/y_test.npy")  

# ----------------------------
# 2. Load the trained model
# ----------------------------
model_path = "models/lstm_model.keras"
model = load_model(model_path)

# ----------------------------
# 3. Predict
# ----------------------------
y_pred = model.predict(X_test)

# ----------------------------
# 4. Compute evaluation metrics
# ----------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# ----------------------------
# 5. Plot predicted vs true
# ----------------------------

#Figure 1: Scatter plot of predicted vs true values
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred, color="#FF7F0E", alpha=0.6, label="Predicted", edgecolors='k', s=50)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Perfect Prediction")
plt.xlabel("True CO2 Emissions", fontsize=14)
plt.ylabel("Predicted CO2 Emissions", fontsize=14)
plt.title("Predicted vs True CO2 Emissions", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("pred_vs_true.png", dpi=300)
plt.close()

# Figure 2: Line plot of predicted and true values
sectors = ["Domestic Aviation", "Ground Transport", "Industry", "International Aviation", "Power", "Residential"]
colors_true = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"] 
colors_pred = ["#ff7f0e", "#ffbb78", "#17becf", "#bcbd22", "#7f7f7f", "#aec7e8"]   

num_sectors = y_test.shape[1]  
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12))
axes = axes.flatten() 

for i in range(num_sectors):
    ax = axes[i]
    ax.plot(y_test[:, i], label="True", color=colors_true[i], linewidth=2)
    ax.plot(y_pred[:, i], label="Predicted", color=colors_pred[i], linestyle='--', linewidth=2)
    ax.set_title(sectors[i], fontsize=14)
    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("CO2 Emissions (MtCO2/day)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("pred_vs_true_sectors_subplot.png", dpi=300)
plt.close()

# ----------------------------
# 6. Log into MLflow
# ----------------------------
mlflow.set_experiment("CO2_Emission_Evaluation")

with mlflow.start_run(run_name="Evaluate_Test_Data"):
    # Log metrics
    mlflow.log_metrics({
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    # Log training parameters
    mlflow.log_params({
        "epochs": 30,
        "batch_size": 32
    })

    # Log tags
    mlflow.set_tags({
        "framework": "TensorFlow",
        "model_type": "LSTM",
        "task": "Regression",
        "outputs": 6
    })

    # Log the figure
    mlflow.log_artifact("pred_vs_true.png")
    mlflow.log_artifact("pred_vs_true_sectors_subplot.png")

    # Log model with input/output signature
    signature = infer_signature(X_test, y_pred)
    mlflow.keras.log_model(model, artifact_path="model", signature=signature)

    print("Evaluation metrics, parameters, plot, and model logged to MLflow successfully!")
