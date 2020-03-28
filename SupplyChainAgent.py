import numpy as np
from collections import defaultdict
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import dill
from ReplayBuffer import ReplayBuffer
from time import clock


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = Sequential([
        Dense(fc1_dims, input_shape=(input_dims, )),
        Activation('relu'),
        Dense(fc2_dims),
        Activation('relu'),
        Dense(n_actions)])

    model.compile(optimizer=Adam(lr=lr), loss='mse')

    return model


class MonteCarloAgent:

    def __init__(self, nA, num_episodes=52, discount_factor=0.99, epsilon=0.5, fname='mc_model.npy'):
        self.return_sum = defaultdict(float)
        self.return_count = defaultdict(float)
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.file_name = fname
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(nA))
        self.episode_memory = []
        self.policy = make_epsilon_greedy_policy(self.Q, self.epsilon, self.nA)
        return

    def get_next_action(self, state):
        probs = self.policy(tuple(state))
        action = np.random.choice(np.arange(len(probs)), p=probs)
        return action

    def remember(self, state, action, reward):
        self.episode_memory.append((tuple(state), action, reward))

    def learn(self):
        episode = self.episode_memory
        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])

        for state, action in sa_in_episode:

            sa_pair = (state, action)
            first_occurence_idx = next(i for i, x in enumerate(episode)
                                       if x[0] == state and x[1] == action)

            # sum up all rewards for first occurences
            G = sum([x[2] * (self.discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])

            # calculate average return for this state ovr all sampled episodes
            self.return_sum[sa_pair] += G
            self.return_count[sa_pair] += 1.0
            self.Q[state][action] = self.return_sum[sa_pair] / self.return_count[sa_pair]

        # update policy
        self.policy = make_epsilon_greedy_policy(self.Q, self.epsilon, self.nA)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def save_model(self):
        with open(self.file_name, 'wb') as dill_file:
            dill.dump(self.Q, dill_file)

    def load_model(self):
        with open(self.file_name, 'rb') as dill_file:
            self.Q = dill.load(self.file_name)

    def reset(self):
        self.episode_memory = []


class DQNAgent:
    """
    Adapted from Phil Tabor's code.
    https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code
    """
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims, mem_size=1000000, fname='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.model_file = fname

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)

        self.q_eval = build_dqn(alpha, n_actions, input_dims, 16, 16)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def get_next_action(self, state):
        state = np.reshape(state, (1, len(state)))
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_next, axis=1) * done

        _ = self.q_eval.fit(state, q_target, verbose=False)

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
