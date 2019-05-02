from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from CharacteristicsExtractor import CharacteristicsExtractor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import os
def crear_fichero_resultado():
# Creamos un nuevo fichero resultado.txt
    f = open("resultado.txt", "w")
    f.close()     
def write(text):
#escribe el resultado 
    f = open("resultado.txt", "a+")
    f.write(text + "\n")
    f.close()

class SignalRecognizer:
    #dada una lista de los vectores de caracteristicas, una lista de las etiquetas de todas las imagenes de
    #entrenamientod y la ruta de las imagenes de test reduce la dimensionalidad de los vectores de caracteristicas
    #con LDA, entrena el clasificador Bayesiano con Gaussianas y clasifica las imagenes de test
    def calculate_lda(self, formatted_vectors, formatted_classes,TEST_PATH):
        crear_fichero_resultado()
        # for t in formatted_vectors:
            # print(len(t))
        x = np.array(formatted_vectors)[:,:,-1]
        y = formatted_classes
        
        # Creamos el clasificador bayesiano
        clasificador = LinearDiscriminantAnalysis(n_components=42)
        
        # reduce la dimensionalidad de los vectores de caracteristicas y entrena el clasificador
        lda = LinearDiscriminantAnalysis(n_components=42)
        print("\n\nEntrenando con las imágenes de entrenamiento y reduciendo la dimensión de los vectores de característicias...")
        x = lda.fit_transform(x, y)
        clasificador.fit(x,y)
        #extraemos los vectores de caracteristicas de las imagenes de test
        extensions = ['jpg', 'png', 'bmp', 'jpeg', 'ppm']
        file_names = [file for file in os.listdir(TEST_PATH) if
                       not file.endswith(".DS_Store") and not file.endswith(".directory") and any(file.endswith(extension) for extension in extensions)]
        test=[]
        etiquetasTest  =[]
        imagenes=[]
        for file in file_names:
            print(file)
            imagenes.append(file)
            etiquetasTest.append(file.split("-")[0])
            ch_ext = CharacteristicsExtractor()
            characteristics_vector=ch_ext.extract_characteristics_vector(TEST_PATH +"/"+ file)
            test.append(characteristics_vector[:,-1])                  
        test=np.array(test)
        
        # reduce la dimensionalidad de los vectores de caracteristicas de las imagenes de test
        print("\n\nReduciendo la dimensión de las imágenes de test...")
        test=lda.transform(test)
         # Predecimos los resultados de Test
        print("\n\nClasificando...")
        y_pred = clasificador.predict(test)
        print("\n\nVector con los resultados de las clases a las que pertenecen las imágenes...")
        print(y_pred)
        
        #escribimos el resultado
        i=0
        for predicction in y_pred:
            write(imagenes[i]+"; "+predicction)
            i+=1

        print("\n\nResultados escritos en el fichero resultado.txt")

        #calcula la matriz de confusión
        print("Calculando matriz de confusión...")
        etiquetas=["00","01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42"]
        matriz=confusion_matrix(etiquetasTest, y_pred,etiquetas)
        n_errores = 0

        for i in range(len(matriz)):
            for j in range(len(matriz[0])):
                if i != j:
                    if matriz[i][j] != 0:
                        n_errores = n_errores + matriz[i][j]
        
        print("\n\nNúmero de fallos de clasificación: " + str(n_errores) + "/" + str(len(imagenes)))

        # Descomentar las siguientes líneas para visualizar los gráficos/estadísticas

        """
        plt.matshow(matriz)
        plt.colorbar()
        plt.show()

        f1_score_result = f1_score(etiquetasTest, y_pred, average=None)

        print("\n\nResultado del F1 Score...")
        print(f1_score_result)
        plt.ylabel("Puntuación")
        plt.xlabel("Clase")
        plt.plot(f1_score_result)
        plt.show()

        precision_score_result = precision_score(etiquetasTest, y_pred, average=None)
        print("precision_score_result:\n " + str(precision_score_result))

        plt.ylabel("Precisión")
        plt.xlabel("Clase")
        plt.plot(precision_score_result)
        plt.show()
        """

