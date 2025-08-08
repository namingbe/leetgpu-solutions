# Tinygrad 0.10.3
import tinygrad

# image is a tensor on the GPU
def solve(image: tinygrad.Tensor, width: int, height: int):
    # horrible, ik
    image[:] = 255 - image
    image[3::4] = 255 - image[3::4]
