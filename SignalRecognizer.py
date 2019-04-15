from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np


class SignalRecognizer:

    def calculate_lda(self, formatted_vectors, formatted_classes):

        # Error en la linea X_train = standar_scaler.fit_transform(X_train)
        # Ocurre porque los vectores dentro de X_train, tienen distinto número de elementos en su interior... hay que arreglarlo
        for t in formatted_vectors:
            print(len(t))

        X = formatted_vectors
        y = formatted_classes
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

        standar_scaler = StandardScaler()
        X_train = standar_scaler.fit_transform(X_train)
        X_test = standar_scaler.transform(X_test)

        # Creamos el objeto LDA
        lda = LinearDiscriminantAnalysis(n_components=1)
        # Creamos la matriz de proyección y
        X_train = lda.fit_transform(X_train, y_train)
        X_test = lda.transform(X_test)

        # Predecimos los resultados de Test
        classifier = LogisticRegression(random_state=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(y_pred)


    # def prepare_dataset(self, ch_vectors):


