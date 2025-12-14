import pandas as pd

# Load dataset
df = pd.read_csv("data/used_car.csv")

# Show first 5 rows
print(df.head())

# Show dataset info
print(df.info())

df = df.drop(['Unnamed: 0', 'car_name', 'model'], axis=1)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

cat_cols = ['brand', 'seller_type', 'fuel_type', 'transmission_type']

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print(df.head())
print(df.info())

from sklearn.model_selection import train_test_split

X = df.drop('selling_price', axis=1)
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

from sklearn.metrics import r2_score, mean_absolute_error

y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

import joblib

joblib.dump(model, "model/car_price_model.pkl")
print("Model saved successfully")

import pandas as pd

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

feature_importance.to_csv("model/feature_importance.csv")

import pandas as pd

st.subheader("📊 What Affects the Price Most?")

fi = pd.read_csv("model/feature_importance.csv", index_col=0)
st.bar_chart(fi)

with st.expander("ℹ️ About this tool"):
    st.write("""
    This application uses a Machine Learning model trained on real used-car market data.
    It considers factors like age, mileage, engine capacity, fuel type, and brand
    to estimate a fair selling price.

    The prediction is an approximation and should be used as a reference value.
    """)
