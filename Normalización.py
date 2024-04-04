import pandas as pd
import numpy as np

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
y_train = train.iloc[:, 0].values  



print(x_train)
print(y_train)

