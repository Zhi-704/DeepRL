import tensorflow as tf
import numpy as np
from tools import clip

class ActorLoss(tf.keras.losses.Loss):
    def __init__(self, ϵ):
        super().__init__()
        ϵ = ϵ
    def call(self, adv, old_log_prob, new_model, img, speed, steer):
        new_probs = tf.math.log(new_model.call(np.array([img]), np.array([np.array([speed])]), np.array([np.array([steer])])))
        ratio = tf.math.exp(new_probs / old_log_prob)
        return -1 * tf.reduce_mean(tf.minimum(ratio * adv, clip(ratio, self.ϵ) * adv))