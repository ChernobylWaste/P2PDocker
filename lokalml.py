import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Load dataset
data_path = './Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
df = pd.read_csv(data_path)
df.columns = df.columns.str.lstrip()

# Select DDoS columns
columns_to_keep = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
    'Bwd URG Flags', 'FIN Flag Count', 'SYN Flag Count',
    'RST Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'Average Packet Size', 'Subflow Fwd Packets', 'Subflow Bwd Packets', 'Label'
]
df_ddos = df[columns_to_keep]

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

# XGBoost parameters
params = {
    'objective': 'multi:softprob',
    'num_class': len(label_encoder.classes_),
    'eval_metric': 'mlogloss',
    'learning_rate': 0.1,
    'max_depth': 6,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'random_state': 42
}

# Create DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Watchlist untuk memantau metrik pada data latih dan uji
watchlist = [(dtrain, 'train'), (dtest, 'eval')]

# Training model
evals_result = {}
model_final = xgb.train(
    params, 
    dtrain, 
    num_boost_round=100, 
    evals=watchlist,
    evals_result=evals_result,
    verbose_eval=1 
)

# Menyimpan model yang sudah di training
model_file_path = "model1.json"
model_final.save_model(model_file_path)
print(f"Model saved to {model_file_path}")

# Predict dan evaluasi
y_pred_prob = model_final.predict(dtest)  # Hasil berupa probabilitas untuk setiap kelas
y_pred_encoded = np.argmax(y_pred_prob, axis=1)  # Ambil indeks kelas dengan probabilitas tertinggi
y_pred = label_encoder.inverse_transform(y_pred_encoded)  # Konversi kembali ke label asli

# Metrics
accuracy = accuracy_score(y_test, y_pred_encoded)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred_encoded, target_names=label_encoder.classes_, digits=4))

# Confusion Matrix
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

# Save confusion matrix menjadi sebuah file
conf_matrix_file_path = "Confusion_matrix.png"
plt.savefig(conf_matrix_file_path)
print(f"Reordered confusion matrix saved to {conf_matrix_file_path}")

plt.show()
