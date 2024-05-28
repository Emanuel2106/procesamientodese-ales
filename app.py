import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Función para extraer características de cada señal CTG
def extract_features_from_signal(signal_file):
    signal_data = pd.read_csv(signal_file)
    fhr_mean = signal_data['FHR'].mean()
    fhr_std = signal_data['FHR'].std()
    uc_mean = signal_data['UC'].mean()
    uc_std = signal_data['UC'].std()
    return fhr_mean, fhr_std, uc_mean, uc_std

# ruta de el archivo train.csv
train_data = pd.read_csv('train.csv')

# Ruta a la carpeta de señales de entrenamiento
train_signals_path = 'signals/train'

#Dataframe para almacenar las características de las señales de entrenamiento
train_signal_features = []

for record_id in train_data['recordID']:
    signal_file = os.path.join(train_signals_path, f"{record_id}.csv")
    fhr_mean, fhr_std, uc_mean, uc_std = extract_features_from_signal(signal_file)
    train_signal_features.append({
        'recordID': record_id,
        'fhr_mean': fhr_mean,
        'fhr_std': fhr_std,
        'uc_mean': uc_mean,
        'uc_std': uc_std
    })

train_signal_features_df = pd.DataFrame(train_signal_features)

# Dataframe de características de señales con el dataframe de datos tabulados
train_data_full = pd.merge(train_data, train_signal_features_df, on='recordID')

# Valores NaN, infinitos y valores extremadamente grandes
train_data_full.fillna(train_data_full.mean(), inplace=True)
train_data_full.replace([np.inf, -np.inf], np.nan, inplace=True)
train_data_full.fillna(train_data_full.mean(), inplace=True)

# Seleccionar las features y la variable a evaluar
features = ['gestweeks', 'age', 'gravidity', 'parity', 'diabetes', 'hypertension',
            'preeclampsia', 'pyrexia', 'meconium', 'noprogress', 'rectype',
            'fhr_mean', 'fhr_std', 'uc_mean', 'uc_std']
X = train_data_full[features]
y = train_data_full['babyhealth']

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divido los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Sobremuestrea el conjunto de entrenamiento
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Verific0 el balance después del sobremuestreo
print(pd.Series(y_train_resampled).value_counts())

# Entrenp el modelo de regresión logística con ajuste de class_weight
logistic_model = LogisticRegression(solver='liblinear', max_iter=2000, class_weight='balanced')
logistic_model.fit(X_train_resampled, y_train_resampled)

# Predecir en el conjunto de prueba
y_pred_logistic = logistic_model.predict(X_test)

# Evaluo el modelo de regresión logística
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
report_logistic = classification_report(y_test, y_pred_logistic)
print("---------------------------------------------------------")
print(f"Logistic Regression Accuracy: {accuracy_logistic}")
print(report_logistic)

# Entreno un modelo Random Forest con ajuste de class_weight
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train_resampled, y_train_resampled)

# Predecir en el conjunto de prueba
y_pred_rf = rf_model.predict(X_test)

# Evaluar el modelo Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)
print("---------------------------------------------------------")
print(f"Random Forest Accuracy: {accuracy_rf}")
print(report_rf)

# cogi el modelo regresion logistica y randomforest 
# para comparar su accuracy 
#  nota: uds pueden coger el de mayor acuracy 
#  o dejar los 2 para hacer una comparacion en el informe


# Proceso para el conjunto de prueba
test_data = pd.read_csv('test.csv')

# Definir la ruta a la carpeta de señales de prueba
test_signals_path = 'signals/test'

# Crear un dataframe para almacenar las características de las señales de prueba
test_signal_features = []

for record_id in test_data['recordID']:
    signal_file = os.path.join(test_signals_path, f"{record_id}.csv")
    fhr_mean, fhr_std, uc_mean, uc_std = extract_features_from_signal(signal_file)
    test_signal_features.append({
        'recordID': record_id,
        'fhr_mean': fhr_mean,
        'fhr_std': fhr_std,
        'uc_mean': uc_mean,
        'uc_std': uc_std
    })

test_signal_features_df = pd.DataFrame(test_signal_features)

# Unir el dataframe de características de señales con el dataframe de datos tabulados de prueba
test_data_full = pd.merge(test_data, test_signal_features_df, on='recordID')

# Verificar y manejar valores NaN, infinitos y valores extremadamente grandes
test_data_full.fillna(test_data_full.mean(), inplace=True)
test_data_full.replace([np.inf, -np.inf], np.nan, inplace=True)
test_data_full.fillna(test_data_full.mean(), inplace=True)

# Seleccionar las características para el conjunto de prueba
X_test_final = test_data_full[features]
X_test_final_scaled = scaler.transform(X_test_final)


# print("---------------------------------------------------------")

# Predecir y evaluar en el conjunto de prueba con ambos modelos

# Predicción con el modelo de regresión logística
y_pred_logistic_test = logistic_model.predict(X_test_final_scaled)
print("---------------------------------------------------------")
print("Logistic Regression Predictions on Test Data")
print(y_pred_logistic_test)

# Predicción con el modelo Random Forest
y_pred_rf_test = rf_model.predict(X_test_final_scaled)
print("---------------------------------------------------------")
print("Random Forest Predictions on Test Data")
print(y_pred_rf_test)

