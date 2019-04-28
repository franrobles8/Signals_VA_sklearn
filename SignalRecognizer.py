from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from CharacteristicsExtractor import CharacteristicsExtractor
import numpy as np
import cv2 as cv
import os
def maxLength(formatted_vectors):
    size=0
    for vector in formatted_vectors:
        if(size<len(vector)):
            size=len(vector)
    return size
def resizeVectors(formatted_vectors):
    maxSize=maxLength(formatted_vectors)
    for vector in formatted_vectors:
        while (len(vector)<maxSize):
            np.append(vector,0)
            
class SignalRecognizer:


    def calculate_lda(self, formatted_vectors, formatted_classes):

        # Error en la linea X_train = standar_scaler.fit_transform(X_train)
        # Ocurre porque los vectores dentro de X_train, tienen distinto número de elementos en su interior... hay que arreglarlo
        for t in formatted_vectors:
            print(len(t))
      
        x = np.array(formatted_vectors)[:,:,-1]
        
        #x = formatted_vectors[:,:,-1]
        y = formatted_classes
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
        
        standar_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        """
       
        
        """solver='svd',"""
        """
        X_train = standar_scaler.fit_transform(x)
        
        X_test = standar_scaler.transform(X_test)len(y)-1
        """
        # Creamos el objeto LDA
        lda = LinearDiscriminantAnalysis(n_components=42)
        # Creamos la matriz de proyección y
        # lda.fit(x,y)
        # x = lda.transform(x)
        # lda.fit(x, y)
        x = lda.fit_transform(x, y)

        

        # Predecimos los resultados de Test
        """
        classifier = LogisticRegression(random_state=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        """
        
        extensions = ['jpg', 'png', 'bmp', 'jpeg', 'ppm']
        file_names = [file for file in os.listdir("./test_reconocimiento") if
                       not file.endswith(".DS_Store") and not file.endswith(".directory") and any(file.endswith(extension) for extension in extensions)]
        test=[]
        etiquetasTest  =[]
        for file in file_names:
            print(file)
            etiquetasTest.append(file.split("-")[0])
            ch_ext = CharacteristicsExtractor()
            characteristics_vector=ch_ext.extract_characteristics_vector("./test_reconocimiento/" + file)
            test.append(characteristics_vector[:,-1])            
            
            
        test=np.array(test)
        
        # ldaTest = LinearDiscriminantAnalysis(n_components=35)
        """
        solver='svd',
        # Creamos la matriz de proyección y
        ldaTest.fit(test,y)
        """
        # test = ldaTest.fit_transform(test, etiquetasTest)
        # print(test.shape)
        # test = lda.transform(test)
        # print(test.shape)
        
        y_pred = lda.predict(test)
        print(y_pred)


    # def prepare_dataset(self, ch_vectors):


