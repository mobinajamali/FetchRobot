#! /usr/bin/env python

import numpy as np
import tensorflow as tf

def convert_to_tf(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float64)
    return arg

def change_shape(original,shape_x, shape_y):
    original_np = np.array(original)
    new = np.reshape(original_np, (shape_x, shape_y))
    return new

def change_list_to_tf(original_list, shape_x, shape_y):
    action_np = change_shape(original_list,shape_x, shape_y)
    action_tf = convert_to_tf(action_np)
    return action_tf

if __name__ == "__main__":
    action=[-1.12855124,
            -0.12091899,
            1.34841811,
            0.44896376,
            0.82030997,
            1.17669048,
            0.40064817,
            -0.88997421,
            1.71008155,
            -0.94205601,
            1.6523181,
            -0.44358072,
            0.17722137,
            0.04564171]
    action_tf = change_list_to_tf(action, 2, 7)
    print(action_tf)
    print(action_tf.shape[0])
    print(action_tf.shape[1])
    print(action_tf.dtype())
    