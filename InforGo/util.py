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
    position = []
    for i in range(16):
        if state.line_scoring[i] == player*3:
            row, col = i//4, i%4
            for j in range(4):
                if not state[j, row, col]:
                    position.append((j, row, col))
                    break
    for i in range(16):
        if state.line_scoring[i+16] == player*3:
            hei, col = i//4, i%4
            for j in range(4):
                if not state[hei, j, col]:
                    position.append((hei, j, col))
                    break
    for i in range(16):
        if state.line_scoring[i+32] == player*3:
            hei, row = i//4, i%4
            for j in range(4):
                if not state[hei, row, j]:
                    position.append((hei, row, j))
                    break
    for i in range(4):
        if state.line_scoring[i+48] == player*3:
            for j in range(4):
                if not state[i, j, j]:
                    position.append((i, j, j))
                    break
    for i in range(4):
        if state.line_scoring[i+52] == player*3:
            for j in range(4):
                if not state[i, j, 3-j]:
                    position.append((i, j, 3-j))
                    break
    for i in range(4):
        if state.line_scoring[i+56] == player*3:
            for j in range(4):
                if not state[j, i, j]:
                    position.append((j, i, j))
                    break
    for i in range(4):
        if state.line_scoring[i+60] == player*3:
            for j in range(4):
                if not state[j, i, 3-j]:
                    position.append((j, i, 3-j))
                    break
    for i in range(4):
        if state.line_scoring[i+64] == player*3:
            for j in range(4):
                if not state[j, j, i]:
                    position.append((j, j, i))
                    break
    for i in range(4):
        if state.line_scoring[i+68] == player*3:
            for j in range(4):
                if not state[j, 3-j, i]:
                    position.append((j, 3-j, i))
                    break
    if state.line_scoring[72] == player*3:
        for j in range(4):
            if not state[j, j, j]:
                position.append((j, j, j))
                break
    if state.line_scoring[73] == player*3:
        for j in range(4):
            if not state[j, j, 3-j]:
                position.append((j, j, 3-j))
                break
    if state.line_scoring[74] == player*3:
        for j in range(4):
            if not state[j, 3-j, j]:
                position.append((j, 3-j, j))
                break
    if state.line_scoring[75] == player*3:
        for j in range(4):
            if not state[j, 3-j, 3-j]:
                position.append((j, 3-j, 3-j))
                break
    # print(state.line_scoring)
    # print(player, position)
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
