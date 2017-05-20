import numpy as np
from utils import Dataset, show_img

dataset = Dataset('./data')

max_row = 34
max_col = 50

test_x = dataset.testing_img
test_y = dataset.testing_label

test_x = test_x.reshape([-1, max_row, max_col, 1])

total_correct = 0

square_w = max_col / 2
square_h = max_row / 2

for idx, img in enumerate(test_x):
    least_dense_square = [9001, 9001] # index, density

    for square_index in range(0, 4):
        square_index = 3 - square_index

        square_density = 0
        w_offset = square_index % 2
        h_offset = int(square_index / 2)

        for row in range(h_offset*square_h, h_offset*square_h + square_h):
            for col in range(w_offset*square_w, w_offset*square_w + square_w):
                square_density += img[row][col]

        if square_density < least_dense_square[1]:
            least_dense_square[1] = square_density
            least_dense_square[0] = square_index

    result = [0.0, 0.0, 0.0, 0.0]
    result[least_dense_square[0]] = 1.0

    if all(result == test_y[idx]):
        total_correct += 1

print total_correct, len(test_x), total_correct/float(len(test_x))