import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

# Mengabaikan peringatan FutureWarning dan UserWarning untuk membersihkan output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

data_path = './Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
df = pd.read_csv(data_path)

# Menghapus spasi di awal kalimat
df.columns = df.columns.str.lstrip()

# Menampilkan informasi dataset, nama kolom, dan distribusi label
print("Jumlah Label dalam Dataset:")
print(df['Label'].value_counts())

# Memisahkan fitur (X) dan label (y) dari dataset
X = df.drop('Label', axis=1)
y = df['Label']

# Mengganti nilai tak terhingga dengan NaN dan mengisi NaN dengan median fitur
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# Membuat objek LabelEncoder yang akan digunakan untuk melakukan encoding pada label kategorikal.
label_encoder = LabelEncoder()
# Mengidentifikasi kategori unik dalam label y dan menetapkan nilai numerik untuk setiap kategori.
y_encoded = label_encoder.fit_transform(y)

# Menormalisasi fitur menggunakan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Membuat objek XGBClassifier dengan beberapa parameter yang ditentukan.
# Parameter use_label_encoder=False untuk menentukan apakah XGBClassifier akan menggunakan LabelEncoder internal untuk mengubah label kategorikal menjadi nilai numerik.
# Parameter eval_metric='logloss' untuk menentukan metrik evaluasi yang digunakan selama pelatihan model.
# Parameter random_state=42 untuk menentukan seed untuk generator random,
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Training model XGBoost
try:
    model.fit(X_scaled, y_encoded)
# Jika training ada error, maka akan mengoutput kannya. Digunakannya exception karena ada bug dari XGBoost dengan skilearn nya.
except Exception as e:
    print(f"Error pada saat memuat model fitting: {e}")

# Menghitung dan menampilkan Importance Score pada fitur teratas
feature_importances = model.feature_importances_
feature_scores = pd.DataFrame({'Fitur': df.columns[:-1], 'Importance': feature_importances})
feature_scores = feature_scores.sort_values(by='Importance', ascending=False)
print("\nFeature Importance Score:")
print(feature_scores.head(20))

# Menyimpan 20 fitur dengan importance score teratas sebagai rekomendasi fitur
recommended_features = feature_scores['Fitur'].head(20).values
print("\nRekomendasi Fitur untuk Deteksi/Klasifikasi Serangan DDoS:")
print(recommended_features)

# Memilih fitur teratas
X_selected = df[recommended_features]

# Mengganti nilai tak terhingga dengan NaN
X_selected.replace([np.inf, -np.inf], np.nan, inplace=True)

# Mengisi NaN dengan median
X_selected.fillna(X_selected.median(), inplace=True)

# Menormalisasi fitur
X_selected_scaled = scaler.fit_transform(X_selected)

# Memisahkan data menjadi set pelatihan 80% dan pengujian 20%
X_train, X_test, y_train, y_test = train_test_split(X_selected_scaled, y_encoded, test_size=0.2, random_state=42)

# Muat model dari masing-masing node
print("\nSedang memuat model 1, 2, dan 3")

model1 = xgb.Booster()
model1.load_model('model1.json')

model2 = xgb.Booster()
model2.load_model('model2.json')

model3 = xgb.Booster()
model3.load_model('model3.json')

print("Model berhasil dimuat!\n")

# Membuat DMatrix untuk prediksi
dtest = xgb.DMatrix(X_test, label=y_test)

# Prediksi dengan setiap model
y_pred1 = model1.predict(dtest)
y_pred2 = model2.predict(dtest)
y_pred3 = model3.predict(dtest)

# Agregasi prediksi menggunakan rata-rata (FedAvg)
y_pred_avg = (y_pred1 + y_pred2 + y_pred3) / 3

# Mengonversi prediksi rata-rata ke kelas
y_pred_class = np.argmax(y_pred_avg, axis=1)

# Evaluasi model teragregasi
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Test Accuracy (FedAvg Aggregation): {accuracy * 100:.2f}%")

# Evaluasi dengan classification report
class_report = classification_report(y_test, y_pred_class, target_names=label_encoder.classes_, digits=4)
print("Classification Report:")
print(class_report)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)

# Ubah urutan label untuk memindahkan DDoS ke baris dan kolom pertama
reordered_indices = [list(label_encoder.classes_).index('DDoS'), list(label_encoder.classes_).index('BENIGN')]
reordered_conf_matrix = conf_matrix[reordered_indices, :][:, reordered_indices]

# Visualisasi confusion matrix yang telah diatur ulang
plt.figure(figsize=(10, 7))
sns.heatmap(reordered_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['DDoS', 'BENIGN'], 
            yticklabels=['DDoS', 'BENIGN'])
plt.title('Confusion Matrix Untuk Aggregated Model')
plt.xlabel('Predicted')
plt.ylabel('True')

# Simpan confusion matrix ke file
reordered_conf_matrix_file = "Confusion Matrix.png"
plt.savefig(reordered_conf_matrix_file, dpi=300, bbox_inches='tight')
print(f"File Gambar Confusion Matrix Tersimpan dengan Nama: {reordered_conf_matrix_file}")

plt.show()

# print("\n menyimpan model aggregated")
# # Simpan model teragregasi ke file .json
# aggregated_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_))

# # Fit model agregat dengan data training jika perlu
# aggregated_model.fit(X_train, y_train)

# # Simpan model agregat ke file .json
# aggregated_model.get_booster().save_model('aggregated_model.json')
# print("Aggregated model saved to 'aggregated_model.json'")

