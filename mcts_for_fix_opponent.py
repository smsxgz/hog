import pickle
from env import DeterministicOpponentEnv as Env
import numpy as np
from collections import defaultdict

ACTIONSPACE = list(range(11))

from mcts import Node


def get_a_trace(values, env):
    s = env.reset()

    trace = []
    done = False
    while values[s].untried == [] and not done:
        a = values[s].UCTselect()
        trace.append((s, a))
        s, r, done, _ = env.step(a)

    if values[s].untried != [] and not done:
        a = np.random.choice(values[s].untried)
        trace.append((s, a))
        s, r, done, _ = env.step(a)

    while not done:
        _, r, done, _ = env.step(a)

    return trace, (r + 1) // 2


def UCT(values, env, iters=500000):
    for i in range(iters):
        if i % 1000 == 0:
            print('{}th-iter / {}iters'.format(i, iters))

        trace, score = get_a_trace(values, env)

        for s, a in trace:
            values[s].update(a, score)

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

    def simple_strategy(*args):
        return 4

    env = Env(simple_strategy)

    values = load_values('save/mcts_base.pkl')

    step = 0
    while True:
        values = UCT(values, env, 500000)
        save_values(values, 'fix-opponent-mcts_v{}.pkl'.format(step))
        step += 1
