import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load dataset
data = pd.read_csv("houses.csv")

# 2. Select input and output columns
X = data[["area_sqft", "bedrooms", "bathrooms", "age_years"]]
y = data["price"]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the Machine Learning model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate the model
predictions = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# 6. Predict a new house price
new_house = pd.DataFrame([{
    "area_sqft": 1500,
    "bedrooms": 3,
    "bathrooms": 2,
    "age_years": 10
}])

predicted_price = model.predict(new_house)[0]

print("\nPredicted House Price:", round(predicted_price, 2))
