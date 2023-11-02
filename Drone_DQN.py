import tensorflow as tf
from Drone_Sim import Simulation as Sim
from collections import deque
import random
import numpy as np
import os

class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

# Initialize the constants
total_episodes = 100
num_actions = 4
learning_rate = 0.01
gamma = 0.99
batch_size = 64
epsilon = 4
num_generations = 10  # Define the number of generations

# Initialize Replay Buffer
replay_buffer = deque(maxlen=10000)

game = Sim()
model = DQN(num_actions)
def argmax(list):
    max = 0
    for number in list:
        if number > max:
            max = number
    return max

def arglist(max, list):
    i = 0
    for number in list:
        if number != max:
            i+=1
        else:
            return i
movements = [0,1,2,3]
def choose_action(state, epsilon):
    option = random.randint(1,epsilon)
    if option > 1:
        state_tensors = [tf.convert_to_tensor([[s]], dtype=tf.float32) for s in state]
        combined_state = tf.concat(state_tensors, axis=0)
        q_values = model(combined_state)
        max = argmax(list(q_values[0][0].numpy()))
        lista = list(q_values[0][0].numpy())
        return arglist(max,lista)
    else:
        return (random.choice(movements))


# Function to perform DQN training steps
def train_dqn(model, optimizer, training_batch, gamma):
    states, actions, rewards, next_states, dones = zip(*training_batch)
    
    dones = [1 if value else -1 for value in dones]
    
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    
    # Filter out None values from the actions list
    filtered_actions = [action for action in actions if action is not None]
    filtered_actions = np.array(filtered_actions, dtype=np.int32)
    
    actions = tf.convert_to_tensor(filtered_actions, dtype=tf.int32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)



    with tf.GradientTape() as tape:
        q_values = model(states)
        action_masks = tf.one_hot(actions, num_actions)
        q_values = tf.reduce_sum(tf.multiply(q_values, action_masks), axis=1)

        rewards = tf.reshape(rewards, [-1, 1])  # Reshape rewards to [batch_size, 1]
        """
        print("next states: ",tf.reduce_max(next_states,axis=1))
        print("reduce max *2",tf.reduce_max(model(next_states), axis=1))
        """
        #print(rewards)
        
        max_values = tf.reduce_max(model(next_states), axis=1)
        max_values_each_row = tf.reshape(tf.reduce_max(max_values, axis=1), (64,1))
        dones = tf.reshape(dones, (64,1))
        #print(max_values_each_row)

        target_q_values = rewards + (1 - dones) * gamma * max_values_each_row
        loss = tf.reduce_mean(tf.square(target_q_values - q_values))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))



# Define the main training loop or steps for the DQN to interact with the game environment
# Training involves actions, collecting experiences, updating the DQN, and more
# Here is a high-level outline of the training loop:



"""
    replay_buffer.append((state, action, reward, next_state, done))
    batch = random.sample(replay_buffer, batch_size)

    Later on, we will sample a small batch for training 
        -We will calculate the Q values for the sample and use them to compute the loss function
        -The optimiser will try to minimise the loss function and will optimise the network to predict better Q values
        -This will happen during training
"""

# Define other necessary elements for the training loop
# For instance, experience replay, epsilon-greedy policy, optimizer, etc.

# Training loop
# Implement steps where the agent (DQN) interacts with the environment (Game)
# Collect experiences, update the DQN, and iterate for a number of episodes or steps
# Use epsilon-greedy policy to balance exploration and exploitation

# Training loop
for generation in range(num_generations):
    for episode in range(total_episodes):
        state = game.get_state()
        done = False

        while not done:
            action = choose_action(state, epsilon)
            new_state, reward, done = game.step(action)

            replay_buffer.append((state, action, reward, new_state, done))
            state = new_state
            print(reward)

            if len(replay_buffer) >= batch_size:
                training_batch = random.sample(replay_buffer, batch_size)
                train_dqn(model, model.optimizer, training_batch, gamma)  # Pass optimizer to the training function

            if episode % 10 == 0:
                # Evaluate, save checkpoints, or monitor training progress here
                pass