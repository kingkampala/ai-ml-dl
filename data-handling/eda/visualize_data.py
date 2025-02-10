import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use the non-GUI backend

import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# 1. Histogram for Age Distribution
plt.figure(figsize=(6, 4))
sns.histplot(df['Age'], bins=5, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.savefig('age_distribution.png')  # Save as image

# 2. Boxplot for Salary (Detecting Outliers)
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Salary'], color='orange')
plt.title('Salary Boxplot')
plt.savefig('salary_boxplot.png')  # Save as image

# 3. Correlation Heatmap
plt.figure(figsize=(6, 4))
numeric_data = df.select_dtypes(include=['number'])  # Select only numeric columns
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')  # Save as image