from CharacteristicsExtractor import CharacteristicsExtractor
from SignalRecognizer import SignalRecognizer

def main():
    ch_ext = CharacteristicsExtractor()
    #while True:
    ch_vectors_by_class = ch_ext.extract_characteristics_vector("./train_recortadas")

    # Ordenamos por clase para que el vector de características coincida con su respectiva clase
    ch_vectors_by_class.sort(key=lambda obj: obj.get_belonging_class(), reverse=False)

    formatted_classes = []
    formatted_vectors = []

    for obj_ch_vector in ch_vectors_by_class:
        formatted_classes.append(obj_ch_vector.get_belonging_class())
        formatted_vectors.append(obj_ch_vector.get_characteristics_vector())

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