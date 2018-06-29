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


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_n))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 1000 == 0:
            print(
                "\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()

        # One step in the environment
        # total_reward = 0.0
        while True:

            # Take a step
            action_probs = policy(state)
            action = np.random.choice(
                np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            state = next_state

    return Q


env = DeterministicOpponentEnv(simple_strategy)
Q = q_learning(env, num_episodes=500000, epsilon=0.1, alpha=0.05)

with open('save/ql_v1.pkl', 'wb') as f:
    pickle.dump(dict(Q), f)

test(get_strategy('save/ql_v1.pkl'))
