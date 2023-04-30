import tensorflow as tf
import numpy as np
from tools import clip

class ActorLoss(tf.keras.losses.Loss):
    def __init__(self, ϵ):
        super().__init__()
        ϵ = ϵ
    def call(self, adv, old_model, new_model, img, speed, steer):
        new_probs = new_model.call(np.array([img]), np.array([np.array([speed])]), np.array([np.array([steer])]))
        old_probs = old_model.call(np.array([img]), np.array([np.array([speed])]), np.array([np.array([steer])]))
        ratio = new_probs / old_probs
        return -1 * tf.reduce_mean(tf.minimum(ratio * adv, clip(ratio, self.ϵ) * adv))