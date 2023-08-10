import tensorflow as tf
import numpy as np
#import matplotlib as plt

#Declaracion de eventos pasados:

entrada = np.array([1, 6, 30, 7, 70, 43, 503, 201, 1005, 99], dtype = float)
resultados = np.array ([0.0254, 0.1524, 0.762, 0.1778, 1.778, 1.0922, 12.776, 5.1054, 25.527, 2.514], dtype = float)

#Topografia de la red: 

capa1 = tf.keras.layers.Dense(units = 1, input_shape = [1])

modelo = tf.keras.Sequential([capa1])

#Optimizador y métrica de pérdida:

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)

print ("Entrenando la red...")

#Entramiento del modelo:

entrenamiento = modelo.fit(entrada, resultados, epochs=500, verbose = False)

#Guardar red neuronal:

modelo.save ('Redneuronal.h5')
modelo.save_weights('Pesos.h5')

#Verificacion del entrenamiento de la red:

print("FINALIZADO")

i = input ("Ingresar el valor en pulgadas: ")
i=float(i)

prediccion = modelo.predict([i])
print ("El valor en metro es: ", str(prediccion))