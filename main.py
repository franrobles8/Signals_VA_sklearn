from CharacteristicsExtractor import CharacteristicsExtractor
import cv2 as cv

def main():
    ch_ext = CharacteristicsExtractor()
    #while True:
    ch_ext.extract_characteristics_vector("./train_recortadas")

        # Cerramos la ventana presionando escape
"""
        if cv.waitKey(5) == 27:
            break
"""
    #cv.destroyAllWindows()

if __name__ == "__main__":
    main()