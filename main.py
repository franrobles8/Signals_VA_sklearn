from CharacteristicsExtractor import CharacteristicsExtractor
from SignalRecognizer import SignalRecognizer
from sklearn.preprocessing import StandardScaler
import numpy as np
def main():
    ch_ext = CharacteristicsExtractor()
    #while True:
    ch_vectors_by_class = ch_ext.extract_characteristics_vector("./train_recortadas")

    # Ordenamos por clase para que el vector de características coincida con su respectiva clase
    ch_vectors_by_class.sort(key=lambda obj: obj.get_belonging_class(), reverse=False)

    formatted_classes = []
    formatted_vectors = []
    standar_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    for obj_ch_vector in ch_vectors_by_class:
        if(not formatted_classes.__contains__(obj_ch_vector.get_belonging_class())):
            formatted_classes.append(obj_ch_vector.get_belonging_class())
        formatted_vector=np.array(obj_ch_vector.get_characteristics_vector(),np.float64)
        """
        formatted_vector=standar_scaler.fit_transform(np.array(obj_ch_vector.get_characteristics_vector(),np.float64))
        """
        formatted_vectors.append(formatted_vector)

    # print(formatted_classes)

    signal_recognizer = SignalRecognizer()
    signal_recognizer.calculate_lda(formatted_vectors, formatted_classes)


        # Cerramos la ventana presionando escape
"""
        if cv.waitKey(5) == 27:
            break
       
"""
    #cv.destroyAllWindows()

if __name__ == "__main__":
    main()