import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data with missing values
data = {
    'Feature1': [1, 2, np.nan, 4, 5],
    'Feature2': [2, 4, 6, 8, 10],
    'Target': [1, 3, 5, 7, 9]
}
df = pd.DataFrame(data)

# Fill missing values with mean
imputer = SimpleImputer(strategy='mean')
df['Feature1'] = imputer.fit_transform(df[['Feature1']])

# Split features and target
X = df[['Feature1', 'Feature2']]
y = df['Target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.2f}")

# Correlation matrix visualization
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()