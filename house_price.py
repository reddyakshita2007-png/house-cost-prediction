import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("houses.csv")
X = df[["area_sqft", "bedrooms", "bathrooms", "age_years"]]
y = df["price"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42).fit(Xtr, ytr)
print("MAE:", round(mean_absolute_error(yte, model.predict(Xte)), 2))
v = [[float(input("Area sqft: ")), int(input("Bedrooms: ")),
      int(input("Bathrooms: ")), float(input("Age years: "))]]
print("Estimated house price:", round(model.predict(v)[0], 2))
