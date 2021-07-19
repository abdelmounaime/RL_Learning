import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

env_name = "Pendulum-v0"
env = gym.make(env_name)

num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (self.x_prev + self.theta*(self.mean-self.x_prev)*self.dt
                + self.std_dev*np.sqrt(self.dt)*np.random.normal(size=self.mean.shape))
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.min_experience_treshold = 5

        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))  
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    @tf.function
    def update(self, model, state_batch, action_batch, reward_batch, next_state_batch):
        with tf.GradientTape() as tape:
            target_actions = model.target_actor(next_state_batch, training=True)
            y = reward_batch + model.gamma*model.target_critic([next_state_batch, target_actions], training = True)
            critic_value = model.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y-critic_value))
        critic_grad = tape.gradient(critic_loss, model.critic_model.trainable_variables)
        model.critic_optimizer.apply_gradients(zip(critic_grad, model.critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = model.actor_model(state_batch, training=True)
            critic_value = model.critic_model([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, model.actor_model.trainable_variables)
        model.actor_optimizer.apply_gradients(zip(actor_grad, model.actor_model.trainable_variables))


    def learn(self, model):
        # Check if we have enough experience in buffer
        if self.buffer_counter < self.min_experience_treshold 
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(model, state_batch, action_batch, reward_batch, next_state_batch)

class UCB1:
    def __init__(self, algos, epsilon):
        self.n = 0
        self.epsilon = epsilon
        self.algos = algos
        self.nk = np.zeros(len(algos))
        self.xk = np.zeros(len(algos))

    def select_best_algo(self):
        #check if nk == 0 or if we are in the first iterations
        for i in range(len(self.algos)):
            if self.nk[i] < 5:
                return np.random.randint(len(self.algos))

        return np.argmax([ self.xk[i] + np.sqrt(self.epsilon * np.log(self.n)/self.nk[i]) for i in len(algos)])

    def update_UCB1(self, algo_index, rew):
        self.xk[algo_index] = (self.nk[algo_index]*self.xk[algo_index] + rew)/(self.nk[algo_index]+1)
        self.nk[algo_index] += 1
        self.n += 1

class ddpg:
    def __init__(self, actor_layers, critic_state_layers, critic_action_layers, gamma, tau, critic_lr, actor_lr):
        self.actor_model = self.get_actor(actor_layers)
        self.critic_model = self.get_critic(critic_state_layers, critic_action_layers)
        self.target_actor = self.get_actor(actor_layers)
        self.target_critic = self.get_critic(critic_state_layers, critic_action_layers)

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.tau = tau
        self.gamma = gamma
        


    @tf.function
    def update_targets(self):
        for (a, b) in zip(self.target_actor.variables, self.actor_model.variables):
            a.assign(b * self.tau + a * (1-self.tau))

        for (a, b) in zip(self.target_critic.variables, self.critic_model.variables):
            a.assign(b * self.tau + a * (1-self.tau))


    def get_actor(self, h_layers):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(num_states,))
        out = layers.Dense(h_layers[0], activation="relu")(inputs)
        for l in h_layers[1:]:
            out = layers.Dense(l, activation="relu")(out)
        outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic(self, h_state_layers, h_action_layers):
        # State as input
        state_input = layers.Input(shape=(num_states))
        state_out = layers.Dense(h_state_layers[0],activation="relu")(state_input)
        for l in h_state_layers[1:]:
            state_out = layers.Dense(l, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(num_actions))
        action_out = layers.Dense(h_action_layers[0], activation="relu")(action_input)
        for l in h_action_layers[1:]:
            action_out = layers.Dense(32, activation="relu")(action_out)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def policy(self, state, noise_object):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

        return [np.squeeze(legal_action)]

std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

total_runs = 10

buffer_ = Buffer(50000, 64)
#Create portfolio of algorithms
portfolio_layers = [
        [[128,128],[8,16],[8,16]],
        [[256,256],[16,32],[16,32]],
        [256,256],[16,16,32],[16,32,32]
        ]

algos = []

for l in portfolio_layers:
    algos.append(ddpg(l[0],l[1],l[2],0.99,0.005,0.002,0.001))

#algo = ddpg([256,256],[16,32],[16,32],0.99,0.005,0.002,0.001)

ep_reward_list = []

avg_reward_list = []

for beta in range(runs):

    #Update algorithms in portfolio
    for algo in algos:

    #for t in 2**beta to 2**(beta+1) - 1
    # run off-line algorithms selected by ucb1
    # update running algorithm ucb1 params


    prev_state = env.reset()
    episodic_reward = 0

    done = False

    while not done:

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        action = algo.policy(tf_prev_state, ou_noise)

        state, reward, done, info = env.step(action)

        buffer_.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer_.learn(algo)
        algo.update_targets()

        prev_state = state

    ep_reward_list.append(episodic_reward)

    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()
        

