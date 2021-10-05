import numpy as np


class Extractor:
    def __init__(self):
        pass

    def extract(self, image_data, r):
        classes = np.zeros(3)
        for x in range(2 * r):
            for y in range(2 * r):
                rr = np.linalg.norm(np.array([r, r], dtype=np.uint8) - np.array([x, y]))
                if rr <= r:
                    c = image_data[y, x]  # opencv uses BGR convention!
                    if c[0] >= 200:  # Blue
                        classes[1] += 1
                    elif c[2] >= 200:  # Red
                        classes[0] += 1
                    elif c[0] == 0 and c[1] == 0 and c[2] == 0:
                        classes[2] += 1

        class_prob = np.around(classes / sum(classes), 2)

        return class_prob
