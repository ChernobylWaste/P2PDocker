import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data_path = './Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
df_ddos = pd.read_csv(data_path)
df_ddos.columns = df_ddos.columns.str.lstrip()

# Select columns Untuk DDoS
columns_to_keep = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
    'Bwd URG Flags', 'FIN Flag Count', 'SYN Flag Count',
    'RST Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'Average Packet Size', 'Subflow Fwd Packets', 'Subflow Bwd Packets', 'Label'
]
df_ddos = df_ddos[columns_to_keep]

# Preprocess data
X = df_ddos.drop('Label', axis=1)
y = df_ddos['Label']
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# Label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

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

