class LesionMemory:

    def __init__(self):
        self.images = {}

    def put(self, key, image):
        self.images[key] = image

    def get(self, key):
        return self.images.get(key)

    def clear(self):
        self.images.clear()
