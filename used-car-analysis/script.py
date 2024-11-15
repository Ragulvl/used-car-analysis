import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import os

# Define the relative path to the dataset
file_path = os.path.join('data', 'used_cars_data.csv')
data = pd.read_csv(file_path)

print(data.columns)

data['Mileage'] = data['Mileage'].fillna(data['Mileage'].mode()[0])
data['Engine'] = data['Engine'].fillna(data['Engine'].mode()[0])
data['Power'] = data['Power'].fillna(data['Power'].mode()[0])
data['Seats'] = data['Seats'].fillna(data['Seats'].mean())

data['Mileage'] = data['Mileage'].str.extract('(\d+\.?\d*)').astype(float)
data['Engine'] = data['Engine'].str.extract('(\d+\.?\d*)').astype(float)
data['Power'] = data['Power'].str.extract('(\d+\.?\d*)').astype(float)

numerical_data = data.select_dtypes(include=[np.number])

plt.figure(figsize=(10, 6))
sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

imputer = SimpleImputer(strategy='mean')
data['Price'] = imputer.fit_transform(data[['Price']])

features = ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats']
target = 'Price'

X = data[features]
y = data[target]

X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}\n")

comparison = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
comparison = comparison.round(2)

print("\nTop 5 Actual vs Predicted Prices:")
print(comparison.head())

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue')
plt.title("Actual vs Predicted Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

clustering_features = data[['Price', 'Mileage', 'Power']]
clustering_features = imputer.fit_transform(clustering_features)
clustering_features = pd.DataFrame(clustering_features, columns=['Price', 'Mileage', 'Power'])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(clustering_features)

sns.scatterplot(x='Mileage', y='Price', hue='Cluster', data=data)
plt.title("Clusters of Cars")
plt.show()

if 'Transmission' in data.columns:
    transmission_counts = data['Transmission'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=transmission_counts.index, y=transmission_counts.values)
    plt.title("Cars by Transmission Type")
    plt.xlabel("Transmission")
    plt.ylabel("Count")
    plt.show()

if 'Fuel_Type' in data.columns:
    fuel_type_counts = data['Fuel_Type'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=fuel_type_counts.index, y=fuel_type_counts.values)
    plt.title("Cars by Engine Type (Fuel Type)")
    plt.xlabel("Fuel Type")
    plt.ylabel("Count")
    plt.show()

seat_capacity_counts = data['Seats'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=seat_capacity_counts.index, y=seat_capacity_counts.values)
plt.title("Cars by Seat Capacity")
plt.xlabel("Seat Capacity")
plt.ylabel("Count")
plt.show()

if 'Location' in data.columns:
    state_counts = data['Location'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=state_counts.index, y=state_counts.values)
    plt.title("Cars by State (Seller Location)")
    plt.xlabel("State")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.show()

if 'Owner_Type' in data.columns:
    owner_type_counts = data['Owner_Type'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=owner_type_counts.index, y=owner_type_counts.values)
    plt.title("Cars by Ownership Type (Individual vs Dealer)")
    plt.xlabel("Owner Type")
    plt.ylabel("Count")
    plt.show()

output_path = os.path.join('output', 'cleaned_used_cars_data.csv')
os.makedirs('output', exist_ok=True)
data.to_csv(output_path, index=False)
print(f"\nCleaned data saved as {output_path}.")
