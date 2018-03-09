import tensorflow as tf
import numpy as np

from os.path import join, dirname

# Hyper parameters for voice conversion
vc = tf.contrib.training.HParams(
    name="vc",

    # Acoustic features
    order=59,
    frame_period=5,
    windows=[
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ],
)
