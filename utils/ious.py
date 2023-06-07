import logging

import numpy as np

from utils.logging import set_logging


def ious_calc(bb_a, bb_b):
    """
    
    :param bb_a: shape -> n x 4
    :param bb_b: shape -> m x 4
    :return: 
    """
    bb_b = np.expand_dims(bb_b, 0)
    bb_a = np.expand_dims(bb_a, 1)

    xx1 = np.maximum(bb_a[..., 0], bb_b[..., 0])
    yy1 = np.maximum(bb_a[..., 1], bb_b[..., 1])
    xx2 = np.minimum(bb_a[..., 2], bb_b[..., 2])
    yy2 = np.minimum(bb_a[..., 3], bb_b[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_a[..., 2] - bb_a[..., 0]) * (bb_a[..., 3] - bb_a[..., 1])
              + (bb_b[..., 2] - bb_b[..., 0]) * (bb_b[..., 3] - bb_b[..., 1]) - wh)
    return o


def iogs_calc(bb_a, bb_b):
    """

    :param bb_a: shape -> n x 4
    :param bb_b: shape -> m x 4
    :return:
    """
    bb_b = np.expand_dims(bb_b, 0)
    bb_a = np.expand_dims(bb_a, 1)

    xx1 = np.maximum(bb_a[..., 0], bb_b[..., 0])
    yy1 = np.maximum(bb_a[..., 1], bb_b[..., 1])
    xx2 = np.minimum(bb_a[..., 2], bb_b[..., 2])
    yy2 = np.minimum(bb_a[..., 3], bb_b[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_a[..., 2] - bb_a[..., 0]) * (bb_a[..., 3] - bb_a[..., 1]))
    return o


if __name__ == '__main__':
    set_logging()
    rects_a = np.array([[0, 0, 100, 100],
                        [300, 300, 500, 500]])
    rects_b = np.array([[50, 50, 150, 150],
                        [400, 300, 500, 600]])

    res = iogs_calc(rects_a, rects_b)
    logging.info(res)
