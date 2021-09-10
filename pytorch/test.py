import torch
import numpy as np
import random
import math

def generate_data(num, x, b):
    res = []
    for i in range(num):
        y = 0
        for j in range(len(x)):
            y += x[j] * i
        y += b + random.random() * 0.5
        res.append(y)

    return res


def model(x, b):


    pass


def loss(y, y_):
    return math.pow(y - y_, 2)



if __name__ == '__main__':
    data = generate_data(100, [3, 4], 2)

    print(data)