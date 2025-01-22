import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


data = pd.read_csv("turkey_car_market.csv")  


selected_columns = ['Marka', 'Arac Tip Grubu', 'Arac Tip', 'Model Yıl', 'Yakıt Turu', 'Vites', 'Kasa Tipi', 'Km', 'Fiyat']
cleaned_data = data[selected_columns].dropna()


categorical_columns = ['Marka', 'Arac Tip Grubu', 'Arac Tip', 'Yakıt Turu', 'Vites', 'Kasa Tipi']
cleaned_data = pd.get_dummies(cleaned_data, columns=categorical_columns, drop_first=True)


X = cleaned_data.drop('Fiyat', axis=1)
y = cleaned_data['Fiyat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)


with open("car_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model başarıyla kaydedildi!")
