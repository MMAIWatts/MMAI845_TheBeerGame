import numpy as np
from collections import defaultdict


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


class MonteCarloAgent:

    def __init__(self, nA, num_episodes=52, discount_factor=1.0, epsilon=0.5):
        self.return_sum = defaultdict(float)
        self.return_count = defaultdict(float)
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(nA))

        self.policy = make_epsilon_greedy_policy(self.Q, self.epsilon, self.nA)
        return

    def get_next_action(self, state):
        probs = self.policy(state)
        action = np.random.choice(np.arange(len(probs)), p=probs)
        return action

    def update_Q(self, episode):
        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            first_occurence_idx = next(i for i, x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            # sum up all rewards for first occurences
            # for i, x in enumerate(episode[first_occurence_idx]):
            #     print(x)

            G = sum([x[2] * (self.discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])

            # calculate average return for this state ovr all sampled episodes
            self.return_sum[sa_pair] += G
            self.return_count[sa_pair] += 1.0
            self.Q[state][action] = self.return_sum[sa_pair] / self.return_count[sa_pair]

            # update policy
            self.policy = make_epsilon_greedy_policy(self.Q, self.epsilon, self.nA)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
