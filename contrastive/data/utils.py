class Size:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self):
        return (self.width, self.height)

    def max_shape(self):
        return max(self.width, self.height)