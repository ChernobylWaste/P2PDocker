# Memanggil library yang akan digunakan.
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

# Mendefinisikan parameter untuk model XGBoost
params = {
    # Menentukan tujuan dari model, yaitu klasifikasi multi-kelas.
    # 'multi:softprob' menghasilkan probabilitas untuk setiap kelas.
    'objective': 'multi:softprob',

    # Menentukan jumlah kelas dalam klasifikasi.
    # Ini diambil dari jumlah kelas yang dihasilkan oleh label encoder.
    'num_class': len(label_encoder.classes_),

    # Metode evaluasi untuk mengukur performa model.
    # 'mlogloss' (multiclass logarithmic loss) digunakan untuk klasifikasi multi-kelas.
    'eval_metric': 'mlogloss',

    # Nilai ini mengontrol seberapa cepat model akan beradaptasi dengan data.
    'learning_rate': 0.1,

    # Kedalaman maksimum untuk setiap tree dalam model. Parameter ini membantu mengontrol overfitting.
    'max_depth': 6,

    # Proporsi fitur yang akan dipilih secara acak untuk digunakan dalam setiap pohon. 0.8 berarti 80% fitur dipilih.
    'colsample_bytree': 0.8,

    # Proporsi sampel yang digunakan untuk membuat setiap pohon. 0.8 berarti 80% data digunakan.
    'subsample': 0.8,

    # Seed atau nilai acak untuk menjaga konsistensi hasil model.
    'random_state': 42
}


# Membuat DMatrix untuk XGBoost dan menyiapkan watchlist untuk evaluasi
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
watchlist = [(dtrain, 'train'), (dtest, 'eval')]
evals_result = {}

# Melatih model XGBoost dengan parameter dan watchlist
model_final = xgb.train(
    params,
    dtrain,
    num_boost_round=35, # Mengatur untuk seberapa banyak data ingin di training
    evals=watchlist,
    evals_result=evals_result,
    verbose_eval=1
)

# Menyimpan model yang dilatih ke file JSON
model_file_path = # "<Model Name>.json"
model_final.save_model(model_file_path)
print(f"Model saved to {model_file_path}")

# Melakukan prediksi
y_pred_prob = model_final.predict(dtest)
y_pred_encoded = np.argmax(y_pred_prob, axis=1)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred_encoded)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred_encoded, target_names=label_encoder.classes_, digits=4))

# Membuat untuk evaluasi model
conf_matrix = confusion_matrix(y_test, y_pred_encoded)
reordered_indices = [list(label_encoder.classes_).index(label) for label in ['DDoS', 'BENIGN']]
conf_matrix = conf_matrix[reordered_indices, :][:, reordered_indices]

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['DDoS', 'BENIGN'],
            yticklabels=['DDoS', 'BENIGN'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# Menyimpan confusion matrix menjadi png
conf_matrix_file_path = "files/Confusion_matrix.png"
plt.savefig(conf_matrix_file_path)
print(f"Reordered confusion matrix saved to {conf_matrix_file_path}")

plt.show()