# -*- coding: utf-8 -*-
import tensorflow as tf
from pinn.layers import CellListNL


class repulsive(tf.keras.Model):
    """repulsive buckingham potential

    This is a simple implementation of repulsive potential
    for the purpose of testing distance/force calculations.

    Args:
        tensors: input data (nested tensor from dataset).
        rc: cutoff radius.
        B, A: repulsive parameters
    """
    def __init__(self, rc=3.0, B=1.0, A=1.0):
        super(LJ, self).__init__()
        self.rc = rc
        self.B = B
        self.A = A
        self.nl_layer = CellListNL(rc)

    def preprocess(self, tensors):
        if 'ind_2' not in tensors:
            tensors.update(self.nl_layer(tensors))
        return tensors

    def call(self, tensors):
        rc, B, A = self.rc, self.B, self.A
        tensors = self.preprocess(tensors)
        e0 = A * (tf.exp(-rc/ B))
        en = A * tf.exp(-tensors['dist'] / B)-e0
        natom = tf.shape(tensors['ind_1'])[0]
        nbatch = tf.reduce_max(tensors['ind_1'])+1
        en = tf.math.unsorted_segment_sum(en, tensors['ind_2'][:, 0], natom)
        return en/2.0
