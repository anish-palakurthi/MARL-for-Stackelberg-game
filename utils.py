import numpy as np
import random


def complexGaussian(row=1, col=1, scale=1, amp=1.0):
    real = np.random.normal(scale=scale, size=[row, col])[0] * np.sqrt(0.5)
    img = np.random.normal(scale=scale, size=[row, col])[0] * np.sqrt(0.5)
    return amp * (real + 1j * img)


def random_selection(relay_list, total_power):
    relay = random.randint(1, len(relay_list))
    power = random.random() * total_power
    return [relay, power]


def max_list(lt):
    temp = 0
    for i in lt:
        if lt.count(i) > temp:
            max_str = i
            temp = lt.count(i)
    return max_str
