from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from CharacteristicsExtractor import CharacteristicsExtractor
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2 as cv
import os
def crear_fichero_restultado():
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
        crear_fichero_restultado()        
        for t in formatted_vectors:
            print(len(t))      
        x = np.array(formatted_vectors)[:,:,-1]
        y = formatted_classes
        
        # Creamos el objeto LDA
        lda = LinearDiscriminantAnalysis(n_components=42)
        
        # reduce la dimensionalidad de los vectores de caracteristicas y entrena el clasificador
        x = lda.fit_transform(x, y)

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
        
        
         # Predecimos los resultados de Test
        y_pred = lda.predict(test)
        print(y_pred)
        
        #escribimos el resultado
        i=0
        for predicction in y_pred:
            write(imagenes[i]+"; "+predicction)
            i+=1
            
        #calcula la matriz de confusión
        etiquetas=["00","01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42"]
        matriz=confusion_matrix(etiquetasTest, y_pred,etiquetas)
        print()
    


