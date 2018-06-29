import time
from env import Env
import numpy as np
from collections import defaultdict

ACTIONSPACE = list(range(11))


class Node:
    def __init__(self):
        self.untried = ACTIONSPACE.copy()
        self.tried = {}
        self.visits = 0

    def update(self, action, result):
        self.visits += 1
        if action not in self.tried:
            self.tried[action] = {'visits': 0, 'wins': 0}
            self.untried.remove(action)
        self.tried[action]['visits'] += 1
        self.tried[action]['wins'] += result

    def get_weights(self, action):
        d = self.tried[action]
        return d['wins'] / d['visits'] + np.sqrt(
            2 * np.log(self.visits) / d['visits'])

    def UCTselect(self):
        assert self.untried == []
        return max(ACTIONSPACE, key=lambda a: self.get_weights(a))


def wrapper_step(env, action):
    s, p, w = env.step(action)
    done = False
    score = None
    if w is not None:
        done = True
        score = 1 - w
    s_key = tuple(reversed(s)) if p == 1 else s
    return s, p, done, score, s_key


def UCT(values, env, iters=500000):
    for i in range(iters):
        s, p = env.reset()
        s_key = tuple(reversed(s)) if p == 1 else s

        trace = []
        done = False
        while values[s_key].untried == [] and not done:
            a = values[s_key].UCTselect()
            trace.append((s, p, a))
            s, p, done, score, s_key = wrapper_step(env, a)

        if values[s_key].untried != [] and not done:
            a = np.random.choice(values[s_key].untried)
            trace.append((s, p, a))
            s, p, done, score, s_key = wrapper_step(env, a)

        while not done:
            _, _, done, score, _ = wrapper_step(env,
                                                np.random.choice(ACTIONSPACE))

        for s, p, a in trace:
            if p == 0:
                values[s].update(a, score)
            if p == 1:
                values[tuple(reversed(s))].update(a, 1 - score)

    return values


def UCT_test(values, env, time_limit=2):
    start = time.time()
    count = 0
    while True:
        count += 1
        s, p = env.reset()
        s_key = tuple(reversed(s)) if p == 1 else s

        trace = []
        done = False
        while values[s_key].untried == [] and not done:
            a = values[s_key].UCTselect()
            trace.append((s, p, a))
            s, p, done, score, s_key = wrapper_step(env, a)

        if values[s_key].untried != [] and not done:
            a = np.random.choice(values[s_key].untried)
            trace.append((s, p, a))
            s, p, done, score, s_key = wrapper_step(env, a)

        while not done:
            _, _, done, score, _ = wrapper_step(env,
                                                np.random.choice(ACTIONSPACE))

        for s, p, a in trace:
            if p == 0:
                values[s].update(a, score)
            if p == 1:
                values[tuple(reversed(s))].update(a, 1 - score)

        if time.time() - start > time_limit:
            print(count)
            break

    return values


if __name__ == '__main__':
    values = defaultdict(Node)
    env = Env()

    values = UCT(values, env, 200000)
    values = UCT_test(values, env, 1)

    # v = {}
    # for s in values:
    #     d = values[s].tried
    #     v[s] = max(ACTIONSPACE, key=lambda a: d[a]['wins'] / d[a]['visits'])
    #
    # import pickle
    # with open('save/mcts.pkl', 'wb') as f:
    #     pickle.dump(v, f)
