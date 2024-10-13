# Customer Lifetime Value (CLV) Calculation and Visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn')

# Generate sample data
np.random.seed(42)
n_customers = 1000

data = {
    'customer_id': range(1, n_customers + 1),
    'total_purchases': np.random.randint(1, 50, n_customers),
    'avg_order_value': np.random.uniform(10, 200, n_customers),
    'purchase_frequency': np.random.uniform(0.1, 2, n_customers),
    'customer_age': np.random.randint(1, 60, n_customers)  # in months
}

df = pd.DataFrame(data)

# Display the first few rows of our dataset
print(df.head())

# Basic CLV Calculation
# CLV = (Average Order Value * Purchase Frequency) * Customer Lifespan

# Let's assume an average customer lifespan of 36 months (3 years)
avg_customer_lifespan = 36

df['basic_clv'] = df['avg_order_value'] * df['purchase_frequency'] * avg_customer_lifespan

# Visualize the distribution of Basic CLV
plt.figure(figsize=(10, 6))
sns.histplot(df['basic_clv'], kde=True)
plt.title('Distribution of Basic Customer Lifetime Value')
plt.xlabel('CLV ($)')
plt.ylabel('Count')
plt.show()

# Advanced CLV Calculation
# We'll use a more sophisticated approach that considers customer churn

# Calculate churn rate (simplified for this example)
# Assume churn rate is inversely proportional to purchase frequency
df['churn_rate'] = 1 / (df['purchase_frequency'] * 12)  # Adjust to yearly rate
df['churn_rate'] = df['churn_rate'].clip(upper=1)  # Ensure churn rate doesn't exceed 1

# Calculate Retention Rate
df['retention_rate'] = 1 - df['churn_rate']

# Calculate discount rate (cost of capital)
# For this example, let's assume a 10% discount rate
discount_rate = 0.1

# Calculate CLV using the formula:
# CLV = (Average Order Value * Purchase Frequency) * (Retention Rate / (1 + Discount Rate - Retention Rate))

df['advanced_clv'] = (df['avg_order_value'] * df['purchase_frequency'] * 12) * (df['retention_rate'] / (1 + discount_rate - df['retention_rate']))

# Visualize the comparison between Basic and Advanced CLV
plt.figure(figsize=(12, 6))
sns.scatterplot(x='basic_clv', y='advanced_clv', data=df, alpha=0.6)
plt.title('Comparison of Basic CLV vs Advanced CLV')
plt.xlabel('Basic CLV ($)')
plt.ylabel('Advanced CLV ($)')
plt.plot([0, df['advanced_clv'].max()], [0, df['advanced_clv'].max()], 'r--', linewidth=2)
plt.show()

# Analyze the relationship between CLV and other variables
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
sns.scatterplot(x='avg_order_value', y='advanced_clv', data=df, alpha=0.6)
plt.title('CLV vs Average Order Value')
plt.xlabel('Average Order Value ($)')
plt.ylabel('CLV ($)')

plt.subplot(2, 2, 2)
sns.scatterplot(x='purchase_frequency', y='advanced_clv', data=df, alpha=0.6)
plt.title('CLV vs Purchase Frequency')
plt.xlabel('Purchase Frequency (per month)')
plt.ylabel('CLV ($)')

plt.subplot(2, 2, 3)
sns.scatterplot(x='customer_age', y='advanced_clv', data=df, alpha=0.6)
plt.title('CLV vs Customer Age')
plt.xlabel('Customer Age (months)')
plt.ylabel('CLV ($)')

plt.subplot(2, 2, 4)
sns.scatterplot(x='total_purchases', y='advanced_clv', data=df, alpha=0.6)
plt.title('CLV vs Total Purchases')
plt.xlabel('Total Purchases')
plt.ylabel('CLV ($)')

plt.tight_layout()
plt.show()

# Calculate and display key statistics
clv_stats = df['advanced_clv'].describe()
print("\nCLV Statistics:")
print(clv_stats)

# Identify top 10% of customers by CLV
top_10_percent = df['advanced_clv'].quantile(0.9)
top_customers = df[df['advanced_clv'] >= top_10_percent]

print(f"\nNumber of top 10% customers: {len(top_customers)}")
print(f"Average CLV of top 10% customers: ${top_customers['advanced_clv'].mean():.2f}")
print(f"Average CLV of all customers: ${df['advanced_clv'].mean():.2f}")

# Visualize the Pareto principle (80/20 rule)
df_sorted = df.sort_values('advanced_clv', ascending=False)
df_sorted['cumulative_clv'] = df_sorted['advanced_clv'].cumsum() / df_sorted['advanced_clv'].sum()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(df_sorted) + 1), df_sorted['cumulative_clv'], 'b-')
plt.title('Cumulative CLV Distribution (Pareto Principle)')
plt.xlabel('Number of Customers')
plt.ylabel('Cumulative % of Total CLV')
plt.axhline(y=0.8, color='r', linestyle='--')
plt.axvline(x=0.2 * len(df), color='r', linestyle='--')
plt.show()

# Calculate the actual percentage of customers contributing to 80% of CLV
customers_80_percent = df_sorted[df_sorted['cumulative_clv'] <= 0.8].shape[0]
percent_customers_80_clv = (customers_80_percent / len(df)) * 100

print(f"\n{percent_customers_80_clv:.2f}% of customers contribute to 80% of total CLV")

# Conclusion and insights
print("\nConclusion and Insights:")
print("1. The advanced CLV calculation provides a more nuanced view of customer value, considering churn and discount rates.")
print("2. There's a strong positive correlation between average order value and CLV, suggesting that increasing order values could significantly impact customer lifetime value.")
print("3. Purchase frequency also shows a positive correlation with CLV, indicating that encouraging repeat purchases is crucial.")
print("4. The relationship between customer age and CLV is not as strong, suggesting that newer customers can be just as valuable as long-standing ones if they make frequent, high-value purchases.")
print("5. The Pareto principle is evident in our CLV distribution, with a small percentage of customers contributing a large portion of the total CLV.")
print("\nRecommendations:")
print("- Focus on strategies to increase average order value and purchase frequency.")
print("- Develop targeted retention programs for high-CLV customers.")
print("- Implement personalized marketing to move lower-CLV customers into higher value segments.")
print("- Regularly update CLV calculations to track changes over time and assess the impact of marketing initiatives.")
