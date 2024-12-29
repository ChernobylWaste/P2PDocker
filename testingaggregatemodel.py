import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data_path = './Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
df_ddos = pd.read_csv(data_path)
df_ddos.columns = df_ddos.columns.str.lstrip()

# Select columns
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

# Load aggregated model
aggregated_model = xgb.Booster()
aggregated_model.load_model('aggregated_model.json')
print("Aggregated model successfully loaded!")

# Membuat DMatrix untuk prediksi
dtest = xgb.DMatrix(X_test, label=y_test)

# Prediksi dengan model yang teragregasi
y_pred_aggregated = aggregated_model.predict(dtest)

# Jika output adalah kelas langsung (bukan probabilitas), tidak perlu np.argmax
y_pred_class = y_pred_aggregated.astype(int)

# Evaluasi model teragregasi
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Test Accuracy (Loaded Aggregated Model): {accuracy * 100:.2f}%")

# Classification report
class_report = classification_report(y_test, y_pred_class, target_names=label_encoder.classes_, digits=4)
print("Classification Report:")
print(class_report)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)

# Visualize confusion matrix using seaborn heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix for Aggregated Model')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Save confusion matrix plot
conf_matrix_file = "aggregated_model_confusion_matrix.png"
plt.savefig(conf_matrix_file, dpi=300, bbox_inches='tight')
print(f"Confusion matrix saved to {conf_matrix_file}")
