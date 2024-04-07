import cv2
import numpy as np
import tensorflow as tf
from PIL import Image,ImageEnhance,ImageFilter
import matplotlib.pyplot as plt


# Recorte de la imagen - Aporte de Joshua Alejandro ;) (Me quedó bien chido el código, apoco no profe?)

#                        _
#             _,..-"""--' `,.-".
#           ,'      __.. --',  |
#         _/   _.-"' |    .' | |       ____
#   ,.-""'    `-"+.._|     `.' | `-..,',--.`.
#  |   ,.                      '    j 7    l \__
#  |.-'                            /| |    j||  .
#  `.                   |         / L`.`""','|\  \
#    `.,----..._       ,'`"'-.  ,'   \ `""'  | |  l
#      Y        `-----'       v'    ,'`,.__..' |   .
#       `.                   /     /   /     `.|   |
#         `.                /     l   j       ,^.  |L
#           `._            L       +. |._   .' \|  | \
#             .`--...__,..-'""'-._  l L  """    |  |  \
#           .'  ,`-......L_       \  \ \     _.'  ,'.  l
#        ,-"`. / ,-.---.'  `.      \  L..--"'  _.-^.|   l
#  .-"".'"`.  Y  `._'   '    `.     | | _,.--'"     |   |
#   `._'   |  |,-'|      l     `.   | |"..          |   l
#   ,'.    |  |`._'      |      `.  | |_,...---"""""`    L
#  /   |   j _|-' `.     L       | j ,|              |   |
# `--,"._,-+' /`---^..../._____,.L',' `.             |\  |
#    |,'      L                   |     `-.          | \j
#             .                    \       `,        |  |
#              \                __`.Y._      -.     j   |
#               \           _.,'       `._     \    |  j
#               ,-"`-----""""'           |`.    \  7   |
#              /  `.        '            |  \    \ /   |
#             |     `      /             |   \    Y    |
#             |      \    .             ,'    |   L_.-')
#              L      `.  |            /      ]     _.-^._
#               \   ,'  `-7         ,-'      / |  ,'      `-._
#              _,`._       `.   _,-'        ,',^.-            `.
#           ,-'     v....  _.`"',          _:'--....._______,.-'
#         ._______./     /',,-'"'`'--.  ,-'  `.
#                  """""`.,'         _\`----...' 
#                         --------""'

# Cargamos la imagen
imagen = cv2.imread("placa.jpeg")

#Extraemos las doiensiones de la imagen
m, n = imagen.shape[:2] 

# Mostramos las dimensiones de la imagen
print("Altura:", m)
print("Ancho:", n)

#---------------------------------------------------------

# Dimensiones de la región a recortar

x_inicio = [370,393,412,450,472,495,530]
y_inicio = [340,340,340,345,350,350,357]
x_fin = [395,415,439,475,495,520,560]
y_fin = [400,400,400,410,410,420,425]

# Calculamos las dimensiones de la región a recortar
ancho = []
alto = []

for i in range(0, len(x_inicio)):
    ancho.append( x_fin[i] - x_inicio[i])
    alto.append( y_fin[i] - y_inicio[i])

# for i in range(0, len(ancho)):
#     print("Ancho de recorte:", ancho[i])
#     print("Alto de recorte:", alto[i])

#---------------------------------------------------------

# # Canvas para las imagenes recortadas
nueva_img=[]
for i in range(0, len(ancho)):
    nueva_img.append(np.zeros((alto[i], ancho[i], 3), dtype=np.uint8))

#Recorte de imagen en OpenCV
for i in range(0, len(x_inicio)):
    for j in range(y_inicio[i], y_fin[i]):
        for k in range(x_inicio[i], x_fin[i]):
            nueva_img[i][j-y_inicio[i], k-x_inicio[i], :] = imagen[j, k, :]

# Indicamos donde recortamos para los 7 caracteres de la placa
for i in range(0, len(x_inicio)):
    cv2.rectangle(imagen,(x_inicio[i],y_inicio[i]),(x_fin[i],y_fin[i]),(0,255,0),3)

for i in range(0, len(nueva_img)):
    cv2.imshow(f"Recorte {i+1}", nueva_img[i].astype(np.uint8))

cv2.imshow("Imagen original", imagen.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()


#Nota para Diego: Aquí ya puedes empeza a cargar el modelo entrenado y usar las imagenes recortadas para predecir los caracteres, lo que sigue ya es el código de cada quien. 

#Cargamos modelo
model = tf.keras.models.load_model('modelo.h5')

class_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e',
    40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
}

#Cargamos las imagenes recortadas

for i, img_array in enumerate(nueva_img):
    # Convertir a imagen PIL y escala de grises
    img = Image.fromarray(img_array).convert('L')
    img = img.resize((28, 28))
    img=Image.fromarray(255-np.array(img))

    # Reducir contraste
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(0.5)  # Ajusta este valor según sea necesario

    # Aplicar filtro Gaussiano para reducir ruido
    img = img.filter(ImageFilter.GaussianBlur(1))  # El radio ajusta la cantidad de suavizado

    # Convertir a numpy array para procesamiento con TF/Keras
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(-1, 28, 28)  # Asegurarse de que tenga la forma correcta

    # Mostrar la imagen procesada
    predict = model.predict(img_array)
    plt.imshow(img_array[0], cmap='binary_r')
    plt.xlabel(f"Yo digo que es un: {class_mapping[np.argmax(predict)]}")
    plt.show()

    # Realizar la predicción
    prediccion = model.predict(img_array)
    clase_predicha = np.argmax(prediccion, axis=1)[0]
    print(f"Predicción {i+1}: {class_mapping[clase_predicha]}")