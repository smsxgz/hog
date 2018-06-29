"""
This is a minimal contest submission file. You may also submit the full
hog.py from Project 1 as your contest entry.

Only this file will be submitted. Make sure to include any helper functions
from `hog.py` that you'll need here! For example, if you have a function to
calculate Free Bacon points, you should make sure it's added to this file
as well.

Don't forget: your strategy must be deterministic and pure.
"""

PLAYER_NAME = 'Yizhuo Miao' # Change this line!
from dice import six_sided, four_sided, make_test_dice
from ucb import main, trace, interact

GOAL_SCORE = 100  # The goal of Hog is to score 100 points.

def roll_dice(num_rolls, dice=six_sided):
    """Simulate rolling the DICE exactly NUM_ROLLS > 0 times. Return the sum of
    the outcomes unless any of the outcomes is 1. In that case, return 1.

    num_rolls:  The number of dice rolls that will be made.
    dice:       A function that simulates a single dice roll outcome.
    """
    # These assert statements ensure that num_rolls is a positive integer.
    assert type(num_rolls) == int, 'num_rolls must be an integer.'
    assert num_rolls > 0, 'Must roll at least once.'
    # BEGIN PROBLEM 1
    "*** YOUR CODE HERE ***"
    points = 0
    i=0
    while i < num_rolls:
        d = dice()
        if d == 1:
            points = 1
        elif points != 1:
            points += d
        i += 1
    return points
    # END PROBLEM 1

def free_bacon(score):
    """Return the points scored from rolling 0 dice (Free Bacon).

    score:  The opponent's current score.
    """
    assert score < 100, 'The game should be over.'
    # BEGIN PROBLEM 2
    "*** YOUR CODE HERE ***"
    first_digit = score // 10
    second_digit = score % 10
    t = first_digit * second_digit
    return t % 10 + 1
    # END PROBLEM 2

def is_swap(score0, score1):
    """Return whether the current player's score has a ones digit
    equal to the opponent's score's tens digit."""
    # BEGIN PROBLEM 4
    "*** YOUR CODE HERE ***"
    a = score0 % 10
    b = score1//10
    if a == b:
        return True
    else:
        return False
    # END PROBLEM 4

def other(player):
    """Return the other player, for a player PLAYER numbered 0 or 1.

    >>> other(0)
    1
    >>> other(1)
    0
    """
    return 1 - player

def silence(score0, score1):
    """Announce nothing (see Phase 2)."""
    return silence

def final_strategy(score, opponent_score):
    """Write a brief description of your final strategy.

    *** YOUR DESCRIPTION HERE ***
    This strategy aim to maximize the difference d = score - opponent_score.
    It improves some details of swap_strategy(margin=8, num_rolls=6)
    (num_rolls=6 maximize the average points gained per turn).
    It includes:
    1) When gain 1 points result of a beneficial swap, return 10;
    2) When close to the Goal, reduce num_rolls to reduce risk;
    3) Avoid adverse swap.
    4) Chage the paramet
    """
    # Your hog final win rate: 0.6776497595321793
    # BEGIN PROBLEM 12
    free = free_bacon(opponent_score)
    d_previous = score - opponent_score
    d_bacon = d_previous + free
    def bad_swap_number(score0, score1):
        number_list = []
        for i in range(1,11):
            if is_swap(score0 + i, score1) and score0 + i > score1:
                number_list.append(i)
        return number_list
    bad = bad_swap_number(score, opponent_score)
    if is_swap(score + free, opponent_score):
        d_bacon = - d_bacon
    d_1 = d_previous+1
    if is_swap(score+1, opponent_score):
        d_1 = - d_1
    d_normal = d_previous + 8
    d = max(d_normal, d_bacon, d_1)
    if score == 0 and opponent_score == 0:
        result = 0    #turn 0
    elif score == 0:
        return 1    #turn 1
    elif (score == 1 or score ==2) and (opponent_score == 1 or opponent_score == 2):   #turn 2
        return 2
    elif (score == 1 or score ==2):   #turn 3
        return 3
    # win in 1 turn:
    elif score + free >= 100 and not is_swap(score+free, opponent_score):
        return 0
    elif score + 1 == 100 and not is_swap(score+1, opponent_score):
        return 10
    elif d == d_bacon:
        result = 0
    elif d == d_1:
        return 7
    #Avoid adverse swap by get 1 point:
    elif 1 in bad or ((7 in bad or 8 in bad or 9 in bad or 10 in bad)
        and (score > GOAL_SCORE - 7 or score - opponent_score > 23)):
        if not (free in bad):
            result = 0
        else:
            return 1
    elif 3 in bad:
        if not free in bad and free > 5:
            result = 0
        else:
            return 2
    #Avoid risk:
    elif score > GOAL_SCORE - 4:
        if not free in bad and free > 3:
            result = 0
        else:
            return 1
    elif score > GOAL_SCORE - 9:
        if not free in bad and free > 5:
            result = 0
        else:
            return 2
    elif score > GOAL_SCORE - 13:
        if not free in bad and free > 7:
            result = 0
        else:
            return 3
    elif score > GOAL_SCORE - 16:
        return 4
    elif score < opponent_score - 7:
        return 7
    elif score < 57:
        return 6
    else:
        return 5
    if result == 0:
        if (is_swap(opponent_score+free_bacon(score+free),score+free)
        and score + free > opponent_score+free_bacon(score+free) + 5):
            if d == d_1:
                return 7
            elif 1 in bad or ((7 in bad or 8 in bad or 9 in bad or 10 in bad)
                and (score > GOAL_SCORE - 7 or score - opponent_score > 23)):
                return 1
            elif 3 in bad:
                return 2
            elif score > GOAL_SCORE - 16:
                return 4
            elif score < opponent_score - 7:
                return 7
            elif score < 57:
                return 6
            else:
                return 5
        else:
            return 0
    # END PROBLEM 12
