from CharacteristicsExtractor import CharacteristicsExtractor
from CharacteristicsExtractorLBP import CharacteristicsExtractorLBP
from SignalRecognizer import SignalRecognizer
from SignalRecognizerKNN import SignalRecognizerKNN
from SignalRecognizerPCA import SignalRecognizerPCA
from SignalRecognizerLBP import SignalRecognizerLBP

import numpy as np
import sys
import getopt
def main():
    """ """
    try:
        lista_opciones, args = getopt.getopt(sys.argv[1:], '', ['train_path=', 'test_path=', 'classifier='])
    except getopt.GetoptError as error:
        print(error)
        sys.exit(2)

    if len(lista_opciones) != 3:
        print("Error en el numero de argumentos")
        print("Los argumentos deben ser de la forma: python main.py --train_path /home/usuario/train --test_path /home/usuario/test --classifier classifier ")
        sys.exit(2)

    for opcion, valor_opcion in lista_opciones:
        if opcion == '--train_path':
            TRAIN_PATH = valor_opcion
        elif opcion == '--test_path':
            TEST_PATH = valor_opcion
        elif opcion == '--classifier':
            CLASSIFIER= valor_opcion
    """
    TEST_PATH = "./test_reconocimiento" 
    TRAIN_PATH="./train_recortadas" 
    CLASSIFIER= "LBP-LDA-Bayes"
    """
    if(CLASSIFIER=="LDA-Bayes"):
        #extraemos los vectores de caracteristicas
        ch_ext = CharacteristicsExtractor()
        ch_vectors_by_class = ch_ext.extract_characteristics_vectors(TRAIN_PATH)
    
        # Ordenamos por clase para que el vector de características coincida con su respectiva clase
        ch_vectors_by_class.sort(key=lambda obj: obj.get_belonging_class(), reverse=False)
    
        #formamos la matriz de vectores de caracteristicas
        formatted_classes = []
        formatted_vectors = []
        for obj_ch_vector in ch_vectors_by_class:
            
            formatted_classes.append(obj_ch_vector.get_belonging_class())
            formatted_vector=np.array(obj_ch_vector.get_characteristics_vector(),np.float64)
            formatted_vectors.append(formatted_vector)
    
        #clasifica las muestras de test
        signal_recognizer = SignalRecognizer()
        signal_recognizer.calculate_lda(formatted_vectors, formatted_classes,TEST_PATH)
    elif(CLASSIFIER=="PCA-Bayes"):
        #extraemos los vectores de caracteristicas
        ch_ext = CharacteristicsExtractor()
        ch_vectors_by_class = ch_ext.extract_characteristics_vectors(TRAIN_PATH)
    
        # Ordenamos por clase para que el vector de características coincida con su respectiva clase
        ch_vectors_by_class.sort(key=lambda obj: obj.get_belonging_class(), reverse=False)
    
        #formamos la matriz de vectores de caracteristicas
        formatted_classes = []
        formatted_vectors = []
        for obj_ch_vector in ch_vectors_by_class:
            
            formatted_classes.append(obj_ch_vector.get_belonging_class())
            formatted_vector=np.array(obj_ch_vector.get_characteristics_vector(),np.float64)
            formatted_vectors.append(formatted_vector)
    
        #clasifica las muestras de test
        signal_recognizerPCA = SignalRecognizerPCA()
        signal_recognizerPCA.calculate_lda(formatted_vectors, formatted_classes,TEST_PATH)
    elif(CLASSIFIER=="LDA-KNN"):
        #extraemos los vectores de caracteristicas
        ch_ext = CharacteristicsExtractor()
        ch_vectors_by_class = ch_ext.extract_characteristics_vectors(TRAIN_PATH)
    
        # Ordenamos por clase para que el vector de características coincida con su respectiva clase
        ch_vectors_by_class.sort(key=lambda obj: obj.get_belonging_class(), reverse=False)
    
        #formamos la matriz de vectores de caracteristicas
        formatted_classes = []
        formatted_vectors = []
        for obj_ch_vector in ch_vectors_by_class:
            
            formatted_classes.append(obj_ch_vector.get_belonging_class())
            formatted_vector=np.array(obj_ch_vector.get_characteristics_vector(),np.float64)
            formatted_vectors.append(formatted_vector)
    
        #clasifica las muestras de test
        signal_recognizer = SignalRecognizerKNN()
        signal_recognizer.calculate_lda(formatted_vectors, formatted_classes,TEST_PATH)
    elif(CLASSIFIER=="LBP-LDA-Bayes"):
        #extraemos los vectores de caracteristicas
        ch_ext = CharacteristicsExtractorLBP()
        ch_vectors_by_class = ch_ext.extract_characteristics_vectors(TRAIN_PATH)
    
        # Ordenamos por clase para que el vector de características coincida con su respectiva clase
        ch_vectors_by_class.sort(key=lambda obj: obj.get_belonging_class(), reverse=False)
    
        #formamos la matriz de vectores de caracteristicas
        formatted_classes = []
        formatted_vectors = []
        for obj_ch_vector in ch_vectors_by_class:
            
            formatted_classes.append(obj_ch_vector.get_belonging_class())
            formatted_vector=np.array(obj_ch_vector.get_characteristics_vector(),np.float64)
            formatted_vectors.append(formatted_vector)
    
        #clasifica las muestras de test
        signal_recognizer = SignalRecognizerLBP()
        signal_recognizer.calculate_lda(formatted_vectors, formatted_classes,TEST_PATH)

if __name__ == "__main__":
    main()