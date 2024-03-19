import time

import numpy as np
from numba import cuda
from PIL import Image
import tqdm
import time


@cuda.jit
def mandelbrot_calc(mandelbrot, start_value):
    i_start = cuda.grid(1)
    if i_start >= mandelbrot.shape[0]:
        return
    for i in range(i_start, mandelbrot.shape[0], cuda.gridsize(1)):
        position = i + start_value
        row = position // row_steps
        col = position % col_steps
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
                mandelbrot[i] = j
                break
            elif j == depth - 1:
                mandelbrot[i] = depth



max_x = 0.8
min_x = - 2.2
max_y = 1.5
min_y = -1.5
resolution = 100000
step = (max_x - min_x) / resolution
row_length = (max_x - min_x)
col_length = (max_y - min_y)
row_steps = int(row_length / step)
col_steps = int(col_length / step)
total_points = int((row_steps * col_steps))
depth = 1000

blocks_per_grid = 1000
threads_per_block = 1000
amount_of_threads = blocks_per_grid * threads_per_block
parts = 100
part_size = total_points // parts
part_rows = row_steps // parts



print("Creating arrays")
mandelbrot_part = np.zeros(part_size).astype(np.int16)


print("Loading arrays into GPU")
dev_mandelbrot = cuda.device_array_like(mandelbrot_part)


progress = tqdm.tqdm(total=total_points, desc="Running Mandelbrot Calculation Kernel")
master_image = Image.new('L', (col_steps, row_steps))

for i in range(0, total_points, part_size):
    mandelbrot_calc[blocks_per_grid, threads_per_block](dev_mandelbrot, i)
    cuda.synchronize()
    current_part = dev_mandelbrot.copy_to_host()

    current_part_image = current_part.reshape((part_rows, col_steps))
    current_part_image = current_part_image / depth
    current_part_image = current_part_image * 255
    current_part_image = current_part_image.astype(np.uint8)
    current_part_image = Image.fromarray(current_part_image)
    current_part_image.save(f"mandelbrot_part_{i // part_size}.png")

    box = (0, i // part_size * part_rows, col_steps, (i // part_size + 1) * part_rows)
    master_image.paste(current_part_image, box)

    progress.update(part_size)



progress.close()
print("Done running Mandelbrot Calculation Kernel")


print("Saving master image")
master_image.save("mandelbrot.png")


