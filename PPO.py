from preprocess import process_img, get_speed, get_steer
from tools import GAE, standardize, clip
from loss import ActorLoss
import tensorflow as tf
import numpy as np
import gymnasium as gym

class ActorModel(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        self.conv_8_3 = tf.keras.layers.Conv2D(8, (3,3), padding="same")
        self.conv_16_5 = tf.keras.layers.Conv2D(16, (5,5), padding="valid")
        self.max_pool = tf.keras.layers.MaxPool2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense_32_sig = tf.keras.layers.Dense(32, activation="sigmoid")
        self.dense_16_1 = tf.keras.layers.Dense(16, activation="relu")
        self.dense_16_2 = tf.keras.layers.Dense(16, activation="relu")
        self.dense_5 = tf.keras.layers.Dense(5, activation="softmax")
        
    def call(self, input_img, speed=None, steer=None):
        x = self.conv_8_3(input_img)
        x = self.max_pool(x)
        x = self.conv_16_5(x)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.dense_32_sig(x)
        x = tf.experimental.numpy.append(x, speed, axis=1)
        x = tf.experimental.numpy.append(x, steer, axis=1)
        x = self.dense_16_1(x)
        x = self.dense_16_2(x)
        return self.dense_5(x)
    


class CriticModel(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        self.conv_8_3 = tf.keras.layers.Conv2D(8, (3,3), padding="same")
        self.conv_16_5 = tf.keras.layers.Conv2D(16, (5,5), padding="valid")
        self.max_pool = tf.keras.layers.MaxPool2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense_32_sig = tf.keras.layers.Dense(32, activation="sigmoid")
        self.dense_16_1 = tf.keras.layers.Dense(16, activation="relu")
        self.dense_16_2 = tf.keras.layers.Dense(16, activation="relu")
        self.dense_1 = tf.keras.layers.Dense(1, activation="linear")
        
    def call(self, input_img, speed=None, steer=None, action=None):
        x = self.conv_8_3(input_img)
        x = self.max_pool(x)
        x = self.conv_16_5(x)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.dense_32_sig(x)
        x = tf.experimental.numpy.append(x, speed, axis=1)
        x = tf.experimental.numpy.append(x, steer, axis=1)
        x = tf.experimental.numpy.append(x, action, axis=1)
        x = self.dense_16_1(x)
        x = self.dense_16_2(x)
        return self.dense_1(x)
    


class PPO(object):
    def __init__(self, actor_lr, critic_lr, ε, buffersize):
        self.actor_model = ActorModel()
        self.actor_lr = actor_lr
        self.actor_model.compile(
            optimizer = tf.optimizers.Adam(actor_lr),
            loss = ActorLoss()
        )
        self.critic_model = CriticModel()
        self.critic_model.compile(
            optimizer = tf.optimizers.Adam(critic_lr),
            loss = "mse"
        )
        self.buffer = Buffer(buffersize)
        self.env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False)
        self.env.reset()
        for i in range(50):
            observation, reward, terminated, truncated, info = self.env.step(0)
        self.speed = get_steer(observation)
        self.steer = get_steer(observation)
        self.img = process_img(observation)
        
    def calculate_loss(self, ϵ, adv, old_prob, img, speed, steer, action):
        new_probs = self.actor_model.call(np.array([img]), np.array([np.array([speed])]), np.array([np.array([steer])]))
        ratio = new_probs / old_prob
        print(f"new_probs = {new_probs}")
        print(f"ratio = {ratio}")
        print(f"ratio * adv = {ratio * adv}")
        print(f"clip(ratio, ϵ) * adv = {clip(ratio, ϵ) * adv}")
        print(f"tf.minimum(ratio * adv, clip(ratio, ϵ) * adv) = {tf.minimum(ratio * adv, clip(ratio, ϵ) * adv)}")
        print(f"tf.reduce_mean(tf.minimum(ratio * adv, clip(ratio, ϵ) * adv)) = {tf.reduce_mean(tf.minimum(ratio * adv, clip(ratio, ϵ) * adv))}")
        return -1 * tf.reduce_mean(tf.minimum(ratio * adv, clip(ratio, ϵ) * adv))
    
    def learn(self, epochs, batchsize):
        """Train the agent."""
        for epoch in range(epochs):
            print(f"epoch {epoch+1}")
            self.buffer.reset()
            self.env.reset()
            for _ in range(49):
                observation, reward, terminated, truncated, info = self.env.step(0)
            while not self.buffer.is_full():
                if self.buffer.index % 1000 == 0:
                    print(f"Step {self.buffer.index}")
                self.step()
#             for _ in range(3):
#                 losses = np.zeros(self.buffer.capacity)
#                 for i in range(self.buffer.capacity):
#                     losses[i] = self.calculate_loss(
#                         0.02,
#                         self.buffer.advantage_buffer[i],
#                         self.buffer.prob_buffer[i],
#                         self.buffer.images_buffer[i],
#                         self.buffer.speeds_buffer[i],
#                         self.buffer.steering_buffer[i],
#                         self.buffer.actions_buffer[i]
#                     )
#                 print(f"avg loss = {np.mean(losses)}")
#                 with tf.GradientTape() as tape:
#                     a_gard = tape.gradient(tf.convert_to_tensor(losses, np.float32), self.actor_model.trainable_weights)
#                     self.actor_model.optimizer.apply_gradients(zip(a_gard, self.actor_model.trainable_weights))
#             history = self.actor_model.fit(losses,
#                                         losses,
#                                         batch_size=64,
#                                         epochs=3
#                                      )
#             tf.saved_model.save(mod,'./actor')
            history = self.critic_model.fit(self.buffer.images_buffer,
                                            self.buffer.speeds_buffer,
                                            self.buffer.steering_buffer,
                                            self.buffer.actions_buffer,
                                            self.buffer.values_buffer,
                                            batch_size=16,
                                            epochs=3
                                      )
            tf.saved_model.save('./critic')
            


    def step(self):
        """Perform the given action in the environment."""
        probs = self.actor_model.call(np.array([self.img]), np.array([np.array([self.speed])]), np.array([np.array([self.steer])]))[0].numpy()
        difference = 1.0 - sum(probs)
        probs /= probs.sum().astype(float)
        action = np.random.choice(5, p=probs)
        prob = probs[action]
        value = self.critic_model.call(np.array([self.img]), np.array([np.array([self.speed])]), np.array([np.array([self.steer])]), np.array([np.array([action])]))
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.buffer.collect(self.img, self.speed, self.steer, action, reward, value, prob)
        self.speed = get_steer(observation)
        self.steer = get_steer(observation)
        self.img = process_img(observation)

        if terminated or self.buffer.is_full():
            if terminated:
                last_val = 0
            else:
                probs = self.actor_model.call(np.array([self.img]), np.array([np.array([self.speed])]), np.array([np.array([self.steer])]))[0].numpy()
                probs /= probs.sum().astype(float)
                action = np.random.choice(5, p=probs)
                prob = probs[action]
                last_val = self.critic_model.call(np.array([self.img]), np.array([np.array([self.speed])]), np.array([np.array([self.steer])]), np.array([np.array([action])]))
            self.buffer.finish_path(last_val, terminated)
            self.env.reset()
            for i in range(50):
                observation, reward, terminated, truncated, info = self.env.step(0)
                self.speed = get_steer(observation)
                self.steer = get_steer(observation)
                self.img = process_img(observation)

    def save():
        """Save current agent networks."""
        
    def load():
        """Load saved agent networks."""
        
    def store_transition():
        """Store transitions to the replay buffer each step."""
        
    def finish_path():
        """Calculate cumulative rewards."""

class Buffer(object):

    def __init__(self, size):
        self.images_buffer = np.zeros(size, dtype = object)
        self.speeds_buffer = np.zeros(size, dtype = np.float32)
        self.steering_buffer = np.zeros(size, dtype = np.float32)
        self.actions_buffer = np.zeros(size, dtype = np.float32)
        self.rewards_buffer = np.zeros(size, dtype = np.float32)
        self.values_buffer = np.zeros(size, dtype = np.float32)
        self.advantage_buffer = np.zeros(size, dtype = np.float32)
        self.rtg_buffer = np.zeros(size, dtype = np.float32)
        self.prob_buffer = np.zeros(size, dtype = np.float32)
        self.lam = 0.99
        self.gamma = 0.99
        self.capacity = size
        self.path_start = 0
        self.index = 0

    def collect(self, img, speed, steering, action, reward, value, prob):

        assert self.index < self.capacity
        self.images_buffer[self.index] = img
        self.speeds_buffer[self.index] = speed
        self.steering_buffer[self.index] = steering
        self.actions_buffer[self.index] = action
        self.rewards_buffer[self.index] = reward
        self.values_buffer[self.index] = value
        self.prob_buffer[self.index] = prob
        self.index += 1

    def reset(self):
        self.__init__(self.capacity)

    def is_full(self):
        return self.index >= self.capacity

    def data(self):
        assert self.index == self.capacity
        self.index = 0
        self.advantage_buffer = standardize(self.advantage_buffer)
        return [self.images_buffer, self.speeds_buffer, self.steering_buffer, self.actions_buffer, self.rewards_buffer, self.advantage_buffer, self.rtg_buffer, self.prob_buffer]
    
    def finish_path(self, last_val, terminated):
        
        if terminated:
            rewards = np.append(self.rewards_buffer[self.index], last_val)
            values = np.append(self.values_buffer[self.index], last_val)
        else:
            rewards = np.append(self.rewards_buffer, last_val)
            values = np.append(self.values_buffer, last_val)
            
        deltas = rewards[:-1] + self.gamma * values[1:] - values [:-1]

        self.advantage_buffer[self.path_start : self.index] = GAE(deltas[self.path_start : self.index], self.gamma, self.lam)

        self.rtg_buffer[self.path_start : self.index-1] = GAE(rewards[self.path_start : self.index], self.gamma,1)[:-1]

        self.path_start = self.index


# if __name__ == "__main__":
#     actor_model = ActorModel()
#     actor_model.compile(
#         optimizer = tf.optimizers.Adam(0.01),
#         loss="mse", 
#         metrics=["mae"]
#     )
#     critic_model = CriticModel()
#     critic_model.compile(
#         optimizer = tf.optimizers.Adam(0.01),
#         loss="mse", 
#         metrics=["mae"]
#     )
