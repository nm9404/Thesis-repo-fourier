import numpy as np
class FourierUtils:
    def __init__(self, image):
        self.image=image

    def transform(self):
        transform = np.fft.fftshift(np.fft.fft2(self.image))
        return [np.abs(transform), np.angle(transform)]

        