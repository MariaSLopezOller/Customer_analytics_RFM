import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Generate sample data (as before)
np.random.seed(42)
n_customers = 1000

data = {
    'customer_id': range(1, n_customers + 1),
    'last_purchase_date': [datetime.now() - timedelta(days=np.random.randint(1, 365)) for _ in range(n_customers)],
    'purchase_frequency': np.random.randint(1, 20, n_customers),
    'total_spend': np.random.uniform(50, 1000, n_customers),
    'avg_basket_size': np.random.uniform(1, 10, n_customers),
    'product_categories': np.random.randint(1, 6, n_customers),
    'customer_support_interactions': np.random.randint(0, 5, n_customers),
    'website_visits': np.random.randint(1, 100, n_customers),
    'discount_used': np.random.choice([0, 1], n_customers, p=[0.7, 0.3])
}

df = pd.DataFrame(data)

# RFM Analysis (as before)
current_date = datetime.now()
df['recency'] = (current_date - df['last_purchase_date']).dt.days
df['frequency'] = df['purchase_frequency']
df['monetary'] = df['total_spend']

r_labels = range(4, 0, -1)
f_labels = range(1, 5)
m_labels = range(1, 5)

r_quartiles = pd.qcut(df['recency'], q=4, labels=r_labels)
f_quartiles = pd.qcut(df['frequency'], q=4, labels=f_labels)
m_quartiles = pd.qcut(df['monetary'], q=4, labels=m_labels)

df['R'] = r_quartiles
df['F'] = f_quartiles
df['M'] = m_quartiles

df['RFM_Score'] = df['R'].astype(str) + df['F'].astype(str) + df['M'].astype(str)

# CLV Calculation (Enhanced version)
avg_purchase_value = df['total_spend'] / df['purchase_frequency']
purchase_frequency = df['purchase_frequency']
churn_rate = 1 / df['recency']  # Simple churn rate estimation
df['CLV'] = (avg_purchase_value * purchase_frequency) / churn_rate

# Advanced Segmentation
segmentation_features = [
    'recency', 'frequency', 'monetary', 'CLV', 'avg_basket_size',
    'product_categories', 'customer_support_interactions', 'website_visits', 'discount_used'
]

# Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[segmentation_features])

# Perform K-means clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Define customer tiers based on CLV and RFM score
def assign_tier(row):
    clv = row['CLV']
    rfm = int(row['RFM_Score'])
    cluster = row['Cluster']
    
    if clv > df['CLV'].quantile(0.9) and rfm > 900:
        return 'Diamond'
    elif clv > df['CLV'].quantile(0.75) and rfm > 700:
        return 'Platinum'
    elif clv > df['CLV'].quantile(0.5) and rfm > 500:
        return 'Gold'
    elif clv > df['CLV'].quantile(0.25) and rfm > 300:
        return 'Silver'
    else:
        return 'Bronze'

df['Customer_Tier'] = df.apply(assign_tier, axis=1)

# Calculate additional metrics
df['purchase_recency'] = df['recency']
df['average_order_value'] = df['total_spend'] / df['purchase_frequency']

# Visualizations
plt.figure(figsize=(20, 15))

# RFM Score Distribution
plt.subplot(3, 3, 1)
sns.histplot(df['RFM_Score'], kde=True)
plt.title('RFM Score Distribution')
plt.xlabel('RFM Score')
plt.ylabel('Count')

# CLV Distribution
plt.subplot(3, 3, 2)
sns.histplot(df['CLV'], kde=True)
plt.title('Customer Lifetime Value Distribution')
plt.xlabel('CLV')
plt.ylabel('Count')

# Recency vs Frequency
plt.subplot(3, 3, 3)
sns.scatterplot(x='recency', y='frequency', data=df, hue='Customer_Tier', palette='viridis')
plt.title('Recency vs Frequency')
plt.xlabel('Recency (days)')
plt.ylabel('Frequency')

# Monetary vs Frequency
plt.subplot(3, 3, 4)
sns.scatterplot(x='monetary', y='frequency', data=df, hue='Customer_Tier', palette='viridis')
plt.title('Monetary vs Frequency')
plt.xlabel('Monetary Value')
plt.ylabel('Frequency')

# CLV vs Average Basket Size
plt.subplot(3, 3, 5)
sns.scatterplot(x='CLV', y='avg_basket_size', data=df, hue='Customer_Tier', palette='viridis')
plt.title('CLV vs Average Basket Size')
plt.xlabel('CLV')
plt.ylabel('Average Basket Size')

# Customer Tier Distribution
plt.subplot(3, 3, 6)
sns.countplot(x='Customer_Tier', data=df, order=['Diamond', 'Platinum', 'Gold', 'Silver', 'Bronze'])
plt.title('Customer Tier Distribution')
plt.xlabel('Customer Tier')
plt.ylabel('Count')

# Cluster Distribution
plt.subplot(3, 3, 7)
sns.countplot(x='Cluster', data=df)
plt.title('Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Count')

# Website Visits vs Customer Support Interactions
plt.subplot(3, 3, 8)
sns.scatterplot(x='website_visits', y='customer_support_interactions', data=df, hue='Customer_Tier', palette='viridis')
plt.title('Website Visits vs Customer Support Interactions')
plt.xlabel('Website Visits')
plt.ylabel('Customer Support Interactions')

# Product Categories vs Discount Used
plt.subplot(3, 3, 9)
sns.boxplot(x='discount_used', y='product_categories', data=df)
plt.title('Product Categories vs Discount Used')
plt.xlabel('Discount Used')
plt.ylabel('Product Categories')

plt.tight_layout()
plt.show()

# Customer Funnel (as before)
funnel_stages = ['Visitors', 'Registered', 'Added to Cart', 'Purchased', 'Repeat Customers']
funnel_values = [1000, 800, 600, 400, 200]  # Sample values

plt.figure(figsize=(10, 6))
plt.bar(funnel_stages, funnel_values)
plt.title('Customer Funnel')
plt.xlabel('Funnel Stage')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.show()

# Advanced Loyalty Program Metrics
loyalty_metrics = {
    'Total Customers': len(df),
    'Average CLV': df['CLV'].mean(),
    'Median CLV': df['CLV'].median(),
    'Average Order Value': df['average_order_value'].mean(),
    'Average Purchase Frequency': df['purchase_frequency'].mean(),
    'Average Purchase Recency': df['purchase_recency'].mean(),
    'Customer Tier Distribution': df['Customer_Tier'].value_counts(normalize=True),
    'Average Basket Size': df['avg_basket_size'].mean(),
    'Average Product Categories': df['product_categories'].mean(),
    'Average Customer Support Interactions': df['customer_support_interactions'].mean(),
    'Average Website Visits': df['website_visits'].mean(),
    'Discount Usage Rate': df['discount_used'].mean(),
    'Cluster Distribution': df['Cluster'].value_counts(normalize=True)
}

print("Advanced Loyalty Program Metrics:")
for metric, value in loyalty_metrics.items():
    print(f"{metric}:")
    print(value)
    print()

# Correlation matrix of key metrics
correlation_matrix = df[['recency', 'frequency', 'monetary', 'CLV', 'avg_basket_size', 
                         'product_categories', 'customer_support_interactions', 'website_visits']].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Matrix of Key Metrics')
plt.show()
