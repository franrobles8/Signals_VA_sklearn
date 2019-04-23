import numpy as np
import cv2 as cv
import os
from ComputedImage import ComputedImage

class CharacteristicsExtractor:
    def extract_characteristics_vector(self, path_image):
        # extraer característica       
        print(path_image)
        img = cv.imread(path_image, 1)

        imgCopy = img.copy()
        imgCopy = cv.cvtColor(imgCopy, cv.COLOR_BGR2GRAY)
        dst = imgCopy.copy()
        cv.equalizeHist(dst, dst)
        dst= cv.resize(dst, (32, 32), interpolation=cv.INTER_LINEAR)
        
        hog = cv.HOGDescriptor(_winSize=(32, 32), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)
        #hog = cv.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)


        # winStride -> Se aconseja el doble del blockStride
        winStride = (16, 16)
        padding = (8, 8)
        #hist = hog.compute(dst, winStride, padding)
        hist = hog.compute(dst,winStride,padding)
        # print(hist)

        

        return hist

    def extract_characteristics_vectors(self, path_training_images):
        # extraer características

        folders = [folder for folder in os.listdir(path_training_images) if not folder.endswith(".DS_Store")]

        computed_list_by_folder = []

        for folder in folders:
            extensions = ['jpg', 'png', 'bmp', 'jpeg', 'ppm']
            file_names = [file for file in os.listdir(path_training_images + "/" + folder) if
                          not file.endswith(".DS_Store") and any(file.endswith(extension) for extension in extensions)]

            for file in file_names:
                print(path_training_images + "/" + folder + "/" + file)
                img = cv.imread(path_training_images + "/" + folder + "/" + file, 1)

                imgCopy = img.copy()
                imgCopy = cv.cvtColor(imgCopy, cv.COLOR_BGR2GRAY)
                dst = imgCopy.copy()
                cv.equalizeHist(dst, dst)
                dst= cv.resize(dst, (32, 32), interpolation=cv.INTER_LINEAR)
                
                hog = cv.HOGDescriptor(_winSize=(32, 32), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)
                #hog = cv.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)


                # winStride -> Se aconseja el doble del blockStride
                winStride = (16, 16)
                padding = (8, 8)
                #hist = hog.compute(dst, winStride, padding)
                hist = hog.compute(dst,winStride,padding)
                # print(hist)

                computed_list_by_folder.append(ComputedImage(folder, hist))

        return computed_list_by_folder
