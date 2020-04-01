import numpy as np
from collections import defaultdict
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import dill
from ReplayBuffer import ReplayBufferonteCarloAgent:

class DeterministicAgent:
    def __init__(self, nA, num_episodes=52, discount_factor=0.99, epsilon=0.5, fname='deterministic_model.npy'):
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

