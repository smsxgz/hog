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


def get_state_key(state, player):
    if player == 0:
        return state
    elif player == 1:
        return tuple(reversed(state))


def UCT(values, env, iters=500000):
    for i in range(iters):
        s, p = env.reset()
        s_key = get_state_key(s, p)

        trace = []
        done = False
        while values[s_key].untried == [] and not done:
            a = values[s_key].UCTselect()
            trace.append((s, p, a))

            s, p, w = env.step(a)
            s_key = get_state_key(s, p)
            if w is not None:
                done = True
                score = 1 - w

        if values[s_key].untried != [] and not done:
            a = np.random.choice(values[s_key].untried)
            trace.append((s, p, a))

            s, p, w = env.step(a)
            s_key = get_state_key(s, p)
            if w is not None:
                done = True
                score = 1 - w

        while not done:
            *_, w = env.step(np.random.choice(ACTIONSPACE))
            if w is not None:
                done = True
                score = 1 - w

        for s, p, a in trace:
            if p == 0:
                values[s].update(a, score)
            if p == 1:
                values[tuple(reversed(s))].update(a, 1 - score)

    return values


if __name__ == '__main__':
    values = defaultdict(Node)
    env = Env()

    values = UCT(values, env, 50000000)

    v = {}
    for s in values:
        d = values[s].tried
        v[s] = max(ACTIONSPACE, key=lambda a: d[a]['wins'] / d[a]['visits'])

    import pickle
    with open('save/mcts.pkl', 'wb') as f:
        pickle.dump(v, f)
