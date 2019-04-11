import numpy as np
import cv2 as cv
import os

class CharacteristicsExtractor:



    def extract_characteristics_vector(self, path_training_images):
        # extraer características

        folders = [folder for folder in os.listdir(path_training_images) if not folder.endswith(".DS_Store")]

        for folder in folders:
            extensions = ['jpg', 'png', 'bmp', 'jpeg', 'ppm']
            file_names = [file for file in os.listdir(path_training_images + "/" + folder) if
                          not file.endswith(".DS_Store") and any(file.endswith(extension) for extension in extensions)]


            img_list = []

            for file in file_names:
                print(path_training_images + "/" + folder + "/" + file)
                img = cv.imread(path_training_images + "/" + folder + "/" + file, 1)

                imgCopy = img.copy()
                imgCopy  = cv.cvtColor(imgCopy, cv.COLOR_BGR2GRAY)
                cv.imshow("Img prueba", imgCopy)