from env import DeterministicOpponentEnv
import numpy as np
from collections import defaultdict
import pickle
import sys

from test import test
from test import get_strategy


def simple_strategy(*args):
    return 4


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def mc_control_epsilon_greedy(env,
                              Q,
                              num_episodes,
                              discount_factor=1.0,
                              epsilon=0.1):

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_n)

    for i_episode in range(1, num_episodes + 1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        episode = []
        state = env.reset()
        for t in range(100):
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            first_occurence_idx = next(i for i, x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            G = sum([
                x[2] * (discount_factor**i)
                for i, x in enumerate(episode[first_occurence_idx:])
            ])
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]

        # The policy is improved implicitly by changing the Q dictionary

    return Q


if __name__ == '__main__':

    Q = defaultdict(lambda: np.zeros(env.action_n))
    env = DeterministicOpponentEnv(simple_strategy)
    for i in range(10):
        Q = mc_control_epsilon_greedy(env, Q, 500000, epsilon=0.1)
        with open(f'save/mc_v{i}.pkl', 'wb') as f:
            pickle.dump(dict(Q), f)

        strategy = get_strategy(f'save/mc_v{i}.pkl')
        env = DeterministicOpponentEnv(strategy)
