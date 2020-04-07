import numpy as np


def parse(file_path, groups):
    with open(file_path, 'r') as f:
        str_list = f.readlines()

    idx = 0
    row = 0
    list_len = len(str_list)
    width = int(list_len / groups)
    arr = np.zeros([groups, width], np.float32)
    for s in str_list:
        if s[0].isdigit() or s[0] == "-":
            arr[idx, row] = float(s)
        idx += 1
        if idx % 1280 == 0:
            row += 1
            idx = 0

    return arr


def _is_float(num):
    float(num)
