import numpy as np
from numba import cuda
from PIL import Image


@cuda.jit
def mandelbrot_calc(iterations):
    i_start = cuda.grid(1)
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x
    for i in range(i_start, iterations.size, threads_per_grid):

        row = i // row_steps
        col = i % col_steps
        re = min_x + step * col
        im = min_y + step * row

        re_z = 0
        im_z = 0
        for j in range(depth):
            temp_re_z = re_z ** 2 - im_z ** 2 + re
            temp_im_z = 2 * re_z * im_z + im
            re_z = temp_re_z
            im_z = temp_im_z
            if re_z * re_z + im_z * im_z > 4:
                iterations[i] = j
                break
            elif j == depth - 1:
                iterations[i] = depth
                break


step = 0.002
max_x = 0.8
min_x = - 2.2
max_y = 1.5
min_y = -1.5
row_length = (max_x - min_x)
col_length = (max_y - min_y)
row_steps = int(row_length / step)
col_steps = int(col_length / step)
total_points = int((row_steps * col_steps))
depth = 10000

print("Creating arrays")
mandelbrot = np.zeros(total_points).astype(np.int8)

print("Loading arrays into GPU")
dev_iterations = cuda.device_array_like(mandelbrot)

print("Running The Mandelbrot Calculation Kernel")
mandelbrot_calc[1024, 1024](dev_iterations)
cuda.synchronize()
print("Done running Mandelbrot Calculation Kernel")


print("Loading arrays into CPU")
mandelbrot = dev_iterations.copy_to_host()


print("Processing Image")
mandelbrot = mandelbrot.reshape((row_steps, col_steps))
mandelbrot = mandelbrot / np.max(mandelbrot)
mandelbrot = mandelbrot * 255
mandelbrot = mandelbrot.astype(np.uint8)
mandelbrot = Image.fromarray(mandelbrot)
mandelbrot.save("mandelbrot.png")
print("Done saving image")
