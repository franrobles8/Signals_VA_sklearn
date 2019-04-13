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
                dst=img.copy()
                imgCopy  = cv.cvtColor(imgCopy, cv.COLOR_BGR2GRAY)
                cv.equalizeHist( imgCopy, dst );
                cv.resize(dst, (32, 32),interpolation=cv.INTER_LINEAR)
                
                hog = cv.HOGDescriptor(_winSize=(32,32),_blockSize=(16,16),_blockStride=(8,8),_cellSize=(8,8),_nbins=9)
                
                hist = hog.compute(dst)
                
                img_list.append(hist)
                """"""
                #cv.imshow("Img prueba", imgCopy)