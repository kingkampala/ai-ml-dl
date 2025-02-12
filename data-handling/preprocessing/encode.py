import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Sample dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Department': ['HR', 'Finance', 'IT', 'HR']
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

# Label Encoding (assigns numeric labels)
label_encoder = LabelEncoder()
df['Department_Label'] = label_encoder.fit_transform(df['Department'])

# One-Hot Encoding (creates separate binary columns)
one_hot = OneHotEncoder()
one_hot = pd.get_dummies(df['Department'], prefix='Dept')

# Combine the DataFrame
df_encoded = pd.concat([df, one_hot], axis=1)

print("\nEncoded Data:\n", df_encoded)