import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json

# Load data
df = pd.read_csv('data.csv')
print(f"Dataset loaded: {len(df)} records")

# Features and target
X = df.drop('math score', axis=1)
y = df['math score']
X = pd.get_dummies(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model comparison
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0)
}

results = {}
best_model = None
best_r2 = -1
best_name = ""

print("\n=== Model Comparison ===")
print(f"{'Model':<25} {'R²':<10} {'RMSE':<10} {'MAE':<10}")
print("-" * 55)

for name, m in models.items():
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    r2 = round(r2_score(y_test, preds), 4)
    rmse = round(np.sqrt(mean_squared_error(y_test, preds)), 4)
    mae = round(mean_absolute_error(y_test, preds), 4)
    
    # Cross-validation
    cv_scores = cross_val_score(m, X, y, cv=5, scoring='r2')
    cv_mean = round(cv_scores.mean(), 4)
    cv_std = round(cv_scores.std(), 4)
    
    results[name] = {
        'r2': r2, 'rmse': rmse, 'mae': mae,
        'cv_mean': cv_mean, 'cv_std': cv_std
    }
    print(f"{name:<25} {r2:<10} {rmse:<10} {mae:<10} (CV: {cv_mean}±{cv_std})")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = m
        best_name = name

print(f"\n✅ Best Model: {best_name} (R²={best_r2})")

# Save best model
pickle.dump(best_model, open('model.pkl', 'wb'))

# Save comparison results
with open('model_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ model.pkl saved")
print("✅ model_comparison.json saved")

# Feature importance (if RF or GB)
if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    top5 = importances.nlargest(5)
    print("\n=== Top 5 Feature Importances ===")
    for feat, imp in top5.items():
        print(f"  {feat}: {imp:.4f}")
