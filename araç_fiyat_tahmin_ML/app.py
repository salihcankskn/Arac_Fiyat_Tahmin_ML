import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
#Gerekli kütüphanelerin kurulması ve import edilmesi

# Daha önce eğitilmiş olan model ".pkl" dosyasının eklenmesi
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Türkiye pazarındaki araç verilerinin bulunduğu csv dosyasını yükleme
data = pd.read_csv("turkey_car_market.csv")

# Veri seti içerisinde bulunan kullanılacak sütunları seçme ve eksik verileri temizleme
selected_columns = ['Marka', 'Arac Tip Grubu', 'Arac Tip', 'Model Yıl', 'Yakıt Turu', 'Vites', 'Kasa Tipi', 'Km', 'Fiyat']
cleaned_data = data[selected_columns].dropna()  # Eksik verileri kaldırma

# Kategorik değişkenleri "one-hot encoding" ile dönüştürme işlemi
categorical_columns = ['Marka', 'Arac Tip Grubu', 'Arac Tip', 'Yakıt Turu', 'Vites', 'Kasa Tipi']
cleaned_data = pd.get_dummies(cleaned_data, columns=categorical_columns, drop_first=True)

# Modelin bağımsız - bağımlı değişkenleri "x - y" ayırma
X = cleaned_data.drop('Fiyat', axis=1)
y = cleaned_data['Fiyat']

# Eğitim için %80, test için %20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit uygulamasında ekranda görünecek başlık
st.title("Araç Fiyat Tahmini")

# Kullanıcıdan araç markasını girdi olarak alma
marka = st.selectbox("Marka", data['Marka'].unique())

# Marka ve araç modelinin bulunduğu araç tipi sütununu bağladık marka seçiminin ardından o markaya ait model seçme işlemi. 
filtered_arac_tip_grubu = data[data['Marka'] == marka]['Arac Tip Grubu'].unique()
arac_tip_grubu = st.selectbox("Araç Tip Grubu", filtered_arac_tip_grubu)

filtered_arac_tip = data[(data['Marka'] == marka) & (data['Arac Tip Grubu'] == arac_tip_grubu)]['Arac Tip'].unique()
arac_tip = st.selectbox("Araç Tip", filtered_arac_tip)

# Model yılı , km gibi bilgileri kullanıcıdan alma işlemi.
model_yil = st.number_input("Model Yılı", min_value=int(data['Model Yıl'].min()), max_value=int(data['Model Yıl'].max()), step=1)
yakit_turu = st.selectbox("Yakıt Türü", data['Yakıt Turu'].unique())
vites = st.selectbox("Vites", data['Vites'].unique())
kasa_tipi = st.selectbox("Kasa Tipi", data['Kasa Tipi'].unique())
km = st.number_input("KM", min_value=0, max_value=int(data['Km'].max()), step=1000)

# Kullanıcıdan alınan girdilere göre modelin beklediği formatta bir veri çerçevesi oluşturma
input_data = pd.DataFrame({
    'Marka_' + marka: [1],
    'Arac Tip Grubu_' + arac_tip_grubu: [1],
    'Arac Tip_' + arac_tip: [1],
    'Yakıt Turu_' + yakit_turu: [1],
    'Vites_' + vites: [1],
    'Kasa Tipi_' + kasa_tipi: [1],
    'Model Yıl': [model_yil],
    'Km': [km]
})

# Eksik sütunları sıfır olarak doldurma
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Kullanıcı "Tahmini Göster" butonuna tıkladığında tahmini hesaplama ve gösterme
if st.button("Tahmini Göster"):
    prediction = model.predict(input_data[X.columns])[0]  # Model ile tahmin yapma
    st.header(f"Öngörülen Araç Fiyatı: {prediction:.2f} TL")  # Modelin tahmin ettiği sonucu ekranda gösterme.
