class ComputedImage:
    def __init__(self, belonging_class, characteristics_vector):
        self.belonging_class = belonging_class
        self.characteristics_vector = characteristics_vector

    def get_belonging_class(self):
        return self.belonging_class

    def get_characteristics_vector(self):
        return self.characteristics_vector

    def __str__(self):
        return "belonging_class: {} , characteristics_vector: {}".format(self.belonging_class, self.characteristics_vector)

