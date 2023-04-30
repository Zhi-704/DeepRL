from preprocess import process_img, get_speed, get_steer
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
        self.dense_5 = tf.keras.layers.Dense(5, activation="softmax")
        
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
        return self.dense_5(x)
    


class PPO(object):
    def __init__():
        actor_model = ActorModel()
        critic_model = CriticModel()
        env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False)
        for i in range(50):
            observation, reward, terminated, truncated, info = env.step(0)
        speed = get_steer(observation)
        steer = get_steer(observation)
        img = process_img(observation)

    def train_actor():
        """Actor training function."""
        
    def train_critic():
        """Critic training function."""
        
    def update():
        """Main training function."""
    
    def get_action(self):
        """Action selection."""
        return np.argmax(self.actor_model.call(np.array([self.img]), np.array([np.array([self.speed])]), np.array([np.array([self.steer])])))

    def step(self, action):
        """Perform the given action in the environment."""
        observation, reward, terminated, truncated, info = self.env.step(action)
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
#states
#actions
#rewards
    def __init__(self,size):
        self.states_buffer = np.zeros(size, dtype = np.float32)
        self.actions_buffer = np.zeros(size, dtype = np.float32)
        self.rewards_buffer = np.zeros(size, dtype = np.float32)
        self.capacity = size
        self.index = 0

    def collect(self, state, action, reward):

        assert self.index < self.capacity
        self.states_buffer[self.index] = state
        self.actions_buffer[self.index] = state
        self.rewards_buffer[self.index] = state
        self.index += 1

    def data(self):
        assert self.index == self.capacity
        self.index = 0
        return [self.states_buffer, self.actions_buffer, self.rewards_buffer]

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
