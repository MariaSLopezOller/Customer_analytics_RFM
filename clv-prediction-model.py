# Predictive Model to Forecast Future CLV Based on Early Customer Behavior

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
n_customers = 5000

data = {
    'customer_id': range(1, n_customers + 1),
    'first_month_purchases': np.random.randint(0, 5, n_customers),
    'first_month_avg_order_value': np.random.uniform(10, 200, n_customers),
    'second_month_purchases': np.random.randint(0, 5, n_customers),
    'second_month_avg_order_value': np.random.uniform(10, 200, n_customers),
    'product_categories_bought': np.random.randint(1, 6, n_customers),
    'days_since_first_purchase': np.random.randint(30, 90, n_customers),
    'website_visits_first_60_days': np.random.randint(1, 20, n_customers),
    'email_opens_first_60_days': np.random.randint(0, 10, n_customers),
    'customer_support_contacts': np.random.randint(0, 3, n_customers)
}

df = pd.DataFrame(data)

# Calculate a simplified CLV (target variable)
# This is a hypothetical future CLV that we'll try to predict
df['future_clv'] = (
    (df['first_month_purchases'] * df['first_month_avg_order_value'] + 
     df['second_month_purchases'] * df['second_month_avg_order_value']) * 
    (1 + 0.1 * df['product_categories_bought']) * 
    (1 + 0.05 * df['website_visits_first_60_days']) * 
    (1 + 0.03 * df['email_opens_first_60_days']) * 
    (1 - 0.1 * df['customer_support_contacts'])
) * 12  # Multiplying by 12 to simulate a yearly CLV

# Display the first few rows of our dataset
print(df.head())

# Split features and target variable
X = df.drop(['customer_id', 'future_clv'], axis=1)
y = df['future_clv']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Visualize predicted vs actual CLV
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual CLV")
plt.ylabel("Predicted CLV")
plt.title("Predicted vs Actual CLV")
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title("Feature Importance")
plt.show()

# SHAP values for model interpretability
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_scaled)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.show()

# Function to predict CLV for new customers
def predict_clv(customer_data):
    scaled_data = scaler.transform(customer_data)
    predicted_clv = rf_model.predict(scaled_data)
    return predicted_clv[0]

# Example of using the prediction function
new_customer = pd.DataFrame({
    'first_month_purchases': [3],
    'first_month_avg_order_value': [150],
    'second_month_purchases': [2],
    'second_month_avg_order_value': [180],
    'product_categories_bought': [4],
    'days_since_first_purchase': [45],
    'website_visits_first_60_days': [10],
    'email_opens_first_60_days': [5],
    'customer_support_contacts': [1]
})

predicted_clv = predict_clv(new_customer)
print(f"Predicted CLV for the new customer: ${predicted_clv:.2f}")

# Insights and recommendations
print("\nInsights and Recommendations:")
print("1. The model shows good predictive power with an R-squared score of {:.2f}".format(r2))
print("2. Early purchase behavior (first and second month) are strong predictors of future CLV")
print("3. Website visits and email engagement in the first 60 days also play significant roles")
print("4. To improve CLV predictions:")
print("   - Collect more detailed data on customer interactions and behaviors")
print("   - Consider incorporating demographic data if available")
print("   - Regularly retrain the model as more data becomes available")
print("5. Use this model to:")
print("   - Identify high-potential customers early for targeted marketing")
print("   - Personalize onboarding experiences to encourage behaviors associated with high CLV")
print("   - Set dynamic customer acquisition costs based on predicted CLV")
print("6. Monitor the actual vs. predicted CLV over time to refine the model and strategies")
