import pickle
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

    def restore(self, d):
        self.tried = d
        self.visits = sum(d[a]['visits'] for a in d)
        self.untried = [a for a in ACTIONSPACE if a not in d]


def wrapper_step(env, action):
    s, p, w = env.step(action)
    done = False
    score = None
    if w is not None:
        done = True
        score = 1 - w
    s_key = tuple(reversed(s)) if p == 1 else s
    return s, p, done, score, s_key


def get_a_trace(values, env):
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
        _, _, done, score, _ = wrapper_step(env, np.random.choice(ACTIONSPACE))

    return trace, score


def UCT(values, env, iters=500000):
    for i in range(iters):
        if i % 1000 == 0:
            print('{}th-iter / {}iters'.format(i, iters))

        trace, score = get_a_trace(values, env)

        for s, p, a in trace:
            if p == 0:
                values[s].update(a, score)
            if p == 1:
                values[tuple(reversed(s))].update(a, 1 - score)

    return values


def save_values(values, filename):
    v = {}
    for s in values:
        d = values[s].tried
        v[s] = d
    with open('save/' + filename, 'wb') as f:
        pickle.dump(v, f)

    q = {}
    for s in v:
        q[s] = []
        for i in range(11):
            d = v[s].get(i, {'visits': 2, 'wins': 1})
            q[s].append(d['wins'] / d['visits'])
    with open('save/strategy-' + filename, 'wb') as f:
        pickle.dump(q, f)


def load_values(filename):
    with open(filename, 'rb') as f:
        v = pickle.load(f)

    values = defaultdict(Node)
    for s in v:
        values[s].restore(v[s])

    return values


if __name__ == '__main__':
    env = Env()

    values = load_values('save/mcts_base.pkl')

    step = 0
    while True:
        values = UCT(values, env, 500000)
        save_values(values, 'mcts_v{}.pkl'.format(step))
        step += 1
