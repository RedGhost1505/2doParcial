import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

# Normalización de los datos - Aporte de Joshua Alejandro y Diego Rodríguez ;) (Trabajo en conjunto)

# cargamos el dataset con pandas
train = pd.read_csv("emnist-balanced-train.csv", header=None)
test = pd.read_csv("emnist-balanced-test.csv",  header=None)

class_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e',
    40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
}

# X = data[:, 1:]  # Datos (todas las columnas excepto la primera)
# y = data[:, 0]   # Etiquetas (la primera columna)

# Separar los datos en características (X) y etiquetas (y)

x_train = train.iloc[:, 1:].values / 255.0  # Normalizar los datos entre 0 y 1, excluyendo la primera columna
x_train_reshaped = x_train.reshape(-1, 28, 28)  # Redimensionamos los datos a imágenes de 28x28
y_train = train.iloc[:, 0].values 
x_test = test.iloc[:, 1:].values / 255.0  # Normalizar los datos entre 0 y 1, excluyendo la primera columna
x_test_reshaped = x_test.reshape(-1, 28, 28)  # Redimensionamos los datos a imágenes de 28x28
y_test = test.iloc[:, 0].values 

# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)

#---------------------------------------------------------

# Entrenamiento de modelo - (hecho por cada quién). 
# Aquí se debe entrenar el modelo con los datos normalizados

#---------------------------------------------------------

# modelo
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),  # EMNIST imágenes de 28x28
  tf.keras.layers.Dense(128, activation='relu'),  # Capa oculta con 128 neuronas
  tf.keras.layers.Dropout(0.2),  # Eliminamos neuronas 
  # tf.keras.layers.Dense(256, activation='relu'), # Capa oculta con 256 neuronas  
  # tf.keras.layers.Dropout(0.5),  # Eliminamos neuronas
  tf.keras.layers.Dense(47, activation='softmax')  # Capa de salida con 47 neuronas (número de clases), softmax para clasificación multiclase
])

# Compilar el modelo
model.compile(optimizer='adam',  # Optimizador Adam
              loss='sparse_categorical_crossentropy',  # Función de pérdida para clasificación multiclase
              metrics=['accuracy'])  # Métrica de evaluación

# Entrenar el modelo
print("Entrenando modelo...")
history = model.fit(x_train_reshaped , y_train, epochs=10, validation_data=(x_test_reshaped, y_test))


#Evaluamos la precisión del modelo
print(model.evaluate(x_test_reshaped, y_test))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Progeso de perdida en el entrenamiento")
plt.xlabel("Pérdida")
plt.ylabel("Epocas")
plt.show()

predict=model.predict(x_test_reshaped)
n=random.randint(0,1000)
print(f'Original: {y_test[n]} Prediccion: {np.argmax(predict[n])}')
plt.imshow(x_test_reshaped[n],cmap='binary_r')
plt.xlabel(f"Yo digo que es {np.argmax(predict[n])}")
plt.show()


# Guardar el modelo
# model.save('modelo.h5')




