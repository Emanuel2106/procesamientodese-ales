import pandas as pd
import os
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score


# Cargar metadatos
metadatos_train = pd.read_csv('/home/usco/dinamic_test/señales/data/data/train.csv')
metadatos_test = pd.read_csv('/home/usco/dinamic_test/señales/data/data/test.csv')

# Ver las primeras filas de los metadatos
print(metadatos_train.head())
print(metadatos_test.head())




# Función para cargar señales CTG
def cargar_senal(file_path):
    return pd.read_csv(file_path)

# Ejemplo de preprocesamiento: manejo de valores faltantes
def preprocesar_senal(df):
    df['seconds'] = pd.to_datetime(df['seconds'], unit='s')
    df.set_index('seconds', inplace=True)
    df = df.resample('1S').mean().interpolate()
    return df

# Aplicar el preprocesamiento a todas las señales
carpeta_senales_train = '/home/usco/dinamic_test/señales/data/data/signals/train'
senales_procesadas = []

for file_name in os.listdir(carpeta_senales_train):
    file_path = os.path.join(carpeta_senales_train, file_name)
    senal_df = cargar_senal(file_path)
    senal_df_procesada = preprocesar_senal(senal_df)
    senales_procesadas.append(senal_df_procesada)




# Ejemplo de análisis de señales con FFT
def analizar_senal(df):
    fft_values = fft(df['FCF'])
    return np.abs(fft_values[:len(fft_values)//2])

# Aplicar el análisis a una señal de ejemplo
senal_ejemplo = senales_procesadas[0]
componentes_frecuencia = analizar_senal(senal_ejemplo)

# Graficar la señal original y su transformada


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(senal_ejemplo.index, senal_ejemplo['FCF'])
plt.title('Señal Original')

plt.subplot(1, 2, 2)
plt.plot(componentes_frecuencia)
plt.title('Transformada de Fourier')
plt.show()




# Ejemplo de extracción de características
def extraer_caracteristicas(signal_df):
    features = {
        'mean_FCF': np.mean(signal_df['FCF']),
        'std_FCF': np.std(signal_df['FCF']),
        'max_FCF': np.max(signal_df['FCF']),
        'min_FCF': np.min(signal_df['FCF']),
        'mean_CU': np.mean(signal_df['CU']),
        'std_CU': np.std(signal_df['CU']),
        'max_CU': np.max(signal_df['CU']),
        'min_CU': np.min(signal_df['CU'])
    }
    return features

# Procesar todas las señales y crear el conjunto de datos final
caracteristicas = []

for signal_df in senales_procesadas:
    features = extraer_caracteristicas(signal_df)
    caracteristicas.append(features)

df_caracteristicas = pd.DataFrame(caracteristicas)
df_final = pd.concat([metadatos_train, df_caracteristicas], axis=1)


 


# Definir variables independientes (X) y dependientes (y)
X = df_final.drop(columns=['babyhealth'])
y = df_final['babyhealth']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predicciones y evaluación
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))







# Validación cruzada
scores = cross_val_score(clf, X, y, cv=5)
print("Cross-Validation Scores:", scores)
print("Mean CV Score:", scores.mean())