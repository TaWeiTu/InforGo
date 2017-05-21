"""utilitiy function and object"""
import numpy as np

from InforGo.environment.global_var import LOGGER, DEBUG


def plot_state(state):
    for r in range(4):
        output = ""
        for h in range(4):
            for c in range(4): output += str(int(state[h][r][c])) if type(state) == 'list' or type(state).__module__ == np.__name__ else str(state[h, r, c])
            output += " | "
        print(output)


def log_state(state, logfile):
    for r in range(4):
        output = ""
        for h in range(4):
            for c in range(4): output += str(int(state[h][r][c]))
            output += " | "
        logfile.write(output + "\n")


def get_pattern(state, player):
    """Calculate corner position, two in a line, three in a line for both player"""
    opponent = -player
    corner = [0, 0]
    two = [0, 0]
    three = [0, 0]
    four = [0, 0]
    f = lambda x: 0 if x == player else 1
    for i in range(4):
        if state[i, i, i] == player: corner[0] += 1
        elif state[i, i, i] == opponent: corner[1] += 1
        if state[i, 3 - i, 3 - i] == player: corner[0] += 1
        elif state[i, 3 - i, 3 - i] == opponent: corner[1] += 1
        if state[i, 3 - i, i] == player: corner[0] += 1
        elif state[i, 3 - i, i] == opponent: corner[1] += 1
        if state[i, i, 3 - i] == player: corner[0] += 1
        elif state[i, i, 3 - i] == opponent: corner[1] += 1

    for h in range(4):
        for r in range(4):
            cnt = [0, 0]
            for c in range(4):
                if state[h, r, c]: cnt[f(state[h, r, c])] += 1
            if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
            if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
            if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
            if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
            if cnt[0] == 4: four[0] += 1
            if cnt[1] == 4: four[1] += 1
        for c in range(4):
            cnt = [0, 0]
            for r in range(4):
                if state[h, r, c]: cnt[f(state[h, r, c])] += 1
            if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
            if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
            if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
            if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
            if cnt[0] == 4: four[0] += 1
            if cnt[1] == 4: four[1] += 1
        cnt = [0, 0]
        for i in range(4):
            if state[h, i, i]: cnt[f(state[h, i, i])] += 1
        if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
        if cnt[0] == 4: four[0] += 1
        if cnt[1] == 4: four[1] += 1
        cnt = [0, 0]
        for i in range(4):
            if state[h, i, 3 - i]: cnt[f(state[h, i, 3 - i])] += 1
        if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
        if cnt[0] == 4: four[0] += 1
        if cnt[1] == 4: four[1] += 1

    for r in range(4):
        for c in range(4):
            cnt = [0, 0]
            for h in range(4):
                if state[h, r, c]: cnt[f(state[h, r, c])] += 1
            if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
            if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
            if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
            if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
            if cnt[0] == 4: four[0] += 1
            if cnt[1] == 4: four[1] += 1
        cnt = [0, 0]
        for i in range(4):
            if state[i, r, i]: cnt[f(state[i, r, i])] += 1
        if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
        if cnt[0] == 4: four[0] += 1
        if cnt[1] == 4: four[1] += 1
        cnt = [0, 0]
        for i in range(4):
            if state[i, r, 3 - i]: cnt[f(state[i, r, 3 - i])] += 1
        if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
        if cnt[0] == 4: four[0] += 1
        if cnt[1] == 4: four[1] += 1
    for c in range(4):
        cnt = [0, 0]
        for i in range(4):
            if state[i, i, c]: cnt[f(state[i, i, c])] += 1
        if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
        if cnt[0] == 4: four[0] += 1
        if cnt[1] == 4: four[1] += 1
        cnt = [0, 0]
        for i in range(4):
            if state[i, 3 - i, c]: cnt[f(state[i, 3 - i, c])] += 1
        if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
        if cnt[0] == 4: four[0] += 1
        if cnt[1] == 4: four[1] += 1
    cnt = [0, 0]
    for i in range(4):
        if state[i, i, i]: cnt[f(state[i, i, i])] += 1
    if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
    if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
    if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
    if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
    if cnt[0] == 4: four[0] += 1
    if cnt[1] == 4: four[1] += 1
    cnt = [0, 0]
    for i in range(4):
        if state[i, i, 3 - i]: cnt[f(state[i, i, 3 - i])] += 1
    if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
    if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
    if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
    if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
    if cnt[0] == 4: four[0] += 1
    if cnt[1] == 4: four[1] += 1
    cnt = [0, 0]
    for i in range(4):
        if state[3 - i, i, i]: cnt[f(state[3 - i, i, i])] += 1
    if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
    if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
    if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
    if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
    if cnt[0] == 4: four[0] += 1
    if cnt[1] == 4: four[1] += 1
    cnt = [0, 0]
    for i in range(4):
        if state[i, 3 - i, i]: cnt[f(state[i, 3 - i, i])] += 1
    if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
    if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
    if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
    if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
    if cnt[0] == 4: four[0] += 1
    if cnt[1] == 4: four[1] += 1
    pattern = np.reshape(np.array([corner[0], corner[1], two[0], two[1], three[0], three[1], four[0], four[1]]), [1, 8])
    return pattern


def get_winning_move(state, player):
    """Calculate corner position, two in a line, three in a line for both player"""
    three = [0, 0]
    f = lambda x: 0 if x == player else 1
    position = []
    for h in range(4):
        for r in range(4):
            cnt = [0, 0]
            for c in range(4):
                if state[h, r, c]: cnt[f(state[h, r, c])] += 1
            if cnt[0] == 3 and cnt[1] == 0:
                for c in range(4):
                    if state[h, r, c] == 0: position.append((h, r, c))
        for c in range(4):
            cnt = [0, 0]
            for r in range(4):
                if state[h, r, c]: cnt[f(state[h, r, c])] += 1
            if cnt[0] == 3 and cnt[1] == 0:
                for r in range(4):
                    if state[h, r, c] == 0: position.append((h, r, c))
        cnt = [0, 0]
        for i in range(4):
            if state[h, i, i]: cnt[f(state[h, i, i])] += 1
        if cnt[0] == 3 and cnt[1] == 0:
            for i in range(4):
                if state[h, i, i] == 0: position.append((h, i, i))
        cnt = [0, 0]
        for i in range(4):
            if state[h, i, 3 - i]: cnt[f(state[h, i, 3 - i])] += 1
        if cnt[0] == 3 and cnt[1] == 0:
            for i in range(4):
                if state[h, i, 3 - i] == 0: position.append((h, i, 3 - i))

    for r in range(4):
        for c in range(4):
            cnt = [0, 0]
            for h in range(4):
                if state[h, r, c]: cnt[f(state[h, r, c])] += 1
            if cnt[0] == 3 and cnt[1] == 0: 
                for h in range(4):
                    if state[h, r, c] == 0: position.append((h, r, c))
        cnt = [0, 0]
        for i in range(4):
            if state[i, r, i]: cnt[f(state[i, r, i])] += 1
        if cnt[0] == 3 and cnt[1] == 0:
            for i in range(4):
                if state[i, r, i] == 0: position.append((i, r, i))
        cnt = [0, 0]
        for i in range(4):
            if state[i, r, 3 - i]: cnt[f(state[i, r, 3 - i])] += 1
        if cnt[0] == 3 and cnt[1] == 0:
            for i in range(4):
                if state[i, r, 3 - i] == 0: position.append((i, r, 3 - i))
    for c in range(4):
        cnt = [0, 0]
        for i in range(4):
            if state[i, i, c]: cnt[f(state[i, i, c])] += 1
        if cnt[0] == 3 and cnt[1] == 0: 
            for i in range(4):
                if state[i, i, c] == 0: position.append((i, i, c))
        cnt = [0, 0]
        for i in range(4):
            if state[i, 3 - i, c]: cnt[f(state[i, 3 - i, c])] += 1
        if cnt[0] == 3 and cnt[1] == 0: 
            for i in range(4):
                if state[i, 3 - i, c] == 0: position.append((i, 3 - i, c))
    cnt = [0, 0]
    for i in range(4):
        if state[i, i, i]: cnt[f(state[i, i, i])] += 1
    if cnt[0] == 3 and cnt[1] == 0:
        for i in range(4):
            if state[i, i, i] == 0: position.append((i, i, i))
    cnt = [0, 0]
    for i in range(4):
        if state[i, i, 3 - i]: cnt[f(state[i, i, 3 - i])] += 1
    if cnt[0] == 3 and cnt[1] == 0: 
        for i in range(4):
            if state[i, i, 3 - i] == 0: position.append((i, i, 3 - i))
    cnt = [0, 0]
    for i in range(4):
        if state[3 - i, i, i]: cnt[f(state[3 - i, i, i])] += 1
    if cnt[0] == 3 and cnt[1] == 0:
        for i in range(4):
            if state[3 - i, i, i] == 0: position.append((3 - i, i, i))
    cnt = [0, 0]
    for i in range(4):
        if state[i, 3 - i, i]: cnt[f(state[i, 3 - i, i])] += 1
    if cnt[0] == 3 and cnt[1] == 0:
        for i in range(4):
            if state[i, 3 - i, i] == 0: position.append((i, 3 - i, i))
    return position

def decode_action(action_num):
    action = [0, 0]
    for i in range(2):
        action[i] = action_num % 4
        action_num //= 4
    return action

def encode_action(action_pair):
    row, col = action_pair
    return row + col * 4


def TD(v, v_, R, alpha, gamma):
    """TD(0)"""
    td = v + alpha * (R + gamma * v_ - v)
    return min(td, 1.0) if td >= 0 else max(td, -1.0)


def emit_action(action):
    row, col = action
    if not DEBUG: print(row, col)


class Logger(object):
    def __init__(self):
        pass
    def info(self, message):
        if DEBUG: LOGGER.info(message)
    def error(self, message):
        if DEBUG: LOGGER.error(message)
    def debug(self, message):
        if DEBUG: LOGGER.debug(message)
    def verbose(self, message):
        if VERBOSE: LOGGER.verbose(message)

logger = Logger()
