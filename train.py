import argparse
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd

# Parse CLI arguments
parser = argparse.Argument()
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--max_iter", type=int, default=1000)
args = parser.parse_args()

# Load dataset
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train model
model = Ridge(alpah=args.alpha, max_iter=args.max_iter)
model.fit(X_train, y_train)

# Log model with MLflow
mlflow.log_param("alpha", args.alpha)
mlflow.log_param("max_iter", args.max_iter)
mlflow.sklearn.log_model(model, "model")

# Print results
print(f"Model trained with alpha={args.alpha}, max_iter={args.max_iter}")
