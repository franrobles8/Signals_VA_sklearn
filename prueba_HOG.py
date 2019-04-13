import cv2
for i in range(100):
    img = cv2.imread("train_recortadas/00/00000.ppm",0)
    cv2.equalizeHist(img,img)
    img = cv2.resize(img,(32,32),interpolation=cv2.INTER_LINEAR) #Es importante alinear la imagen con los bloques de HOG
    
    hog = cv2.HOGDescriptor(_winSize=(32,32),_blockSize=(16,16),_blockStride=(8,8),_cellSize=(8,8),_nbins=9)
    hist = hog.compute(img)
    print(hist)
