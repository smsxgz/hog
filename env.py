import time
import numpy as np


def roll_dice(num_rolls, ns, dice=6):
    points = 0
    for _ in range(num_rolls):
        d = ns.randint(1, dice + 1)
        if d == 1:
            points = 1
            break
        else:
            points += d
    return points


def free_bacon(score):
    assert score < 100
    d1, d2 = divmod(score, 10)
    return (d1 * d2) % 10 + 1


def take_turn(num_rolls, score, opponent_score, ns, dice=6):
    if num_rolls == 0:
        score += free_bacon(opponent_score)
    else:
        score += roll_dice(num_rolls, ns, dice)

    if (score % 10) == (opponent_score // 10 % 10):
        score, opponent_score = opponent_score, score

    return score, opponent_score


class Env:
    def __init__(self):
        self.action_n = 11

    def _get_state(self):
        return (self.score_0, self.score_1)

    def _get_winner(self):
        if self.score_0 >= 100:
            winner = 0
        elif self.score_1 >= 100:
            winner = 1
        else:
            winner = None

        return winner

    def get_player(self):
        return self.player

    def reset(self):
        self.ns = np.random.RandomState(int(time.time() * 1000) % 2147483647)
        self.score_0 = 0
        self.score_1 = 0
        self.turn = -1
        self.player = self.ns.randint(2)
        self._pre_player = 1 - self.player
        return self._get_state(), self.player

    def step(self, action):
        assert 0 <= action < self.action_n
        assert self.score_0 < 100
        assert self.score_1 < 100

        self.turn += 1

        if self.player == 0:
            self.score_0, self.score_1 = take_turn(action, self.score_0,
                                                   self.score_1, self.ns)
        if self.player == 1:
            self.score_1, self.score_0 = take_turn(action, self.score_1,
                                                   self.score_0, self.ns)

        if self.turn % 8 != action or self._pre_player == self.player:
            self._pre_player, self.player = self.player, 1 - self.player
        else:
            self._pre_player = self.player

        return self._get_state(), self.player, self._get_winner()


class SimpleEnv(Env):
    def reset(self):
        self.ns = np.random.RandomState(int(time.time() * 1000) % 2147483647)
        self.score_0 = 0
        self.score_1 = 0
        self.player = self.ns.randint(2)
        return self._get_state(), self.player

    def step(self, action):
        assert 0 <= action < self.action_n
        assert self.score_0 < 100
        assert self.score_1 < 100

        if self.player == 0:
            self.score_0, self.score_1 = take_turn(action, self.score_0,
                                                   self.score_1, self.ns)
        if self.player == 1:
            self.score_1, self.score_0 = take_turn(action, self.score_1,
                                                   self.score_0, self.ns)
        self.player = 1 - self.player

        return self._get_state(), self.player, self._get_winner()


class DeterministicOpponentEnv:
    def __init__(self, opponent_strategy_func, simple=False):
        if simple:
            self.env = SimpleEnv()
        else:
            self.env = Env()
        self.opponent = opponent_strategy_func
        self.action_n = self.env.action_n

    def reset(self):
        state, player = self.env.reset()
        if player == 0:
            return state

        while True:
            assert self.env.get_player() == 1
            state, player, winner = self.env.step(
                self.opponent(state[1], state[0]))
            if winner is not None:
                return self.reset()
            if player == 0:
                return state

    def step(self, action):
        assert self.env.get_player() == 0
        state, player, winner = self.env.step(action)
        if winner is not None:
            return state, 1 - 2 * winner, True, None
        if player == 0:
            return state, 0, False, None

        while True:
            assert self.env.get_player() == 1
            state, player, winner = self.env.step(
                self.opponent(state[1], state[0]))
            if winner is not None:
                return state, 1 - 2 * winner, True, None
            if player == 0:
                return state, 0, False, None
