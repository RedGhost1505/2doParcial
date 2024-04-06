import cv2
import numpy as np

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