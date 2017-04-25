import numpy as np


def plot_state(state):
    for r in range(4):
        output = ""
        for h in range(4):
            for c in range(4): output += str(int(state[h][r][c][0][0]))
            output += " | "
        print(output)


def log_state(state, logfile):
    for r in range(4):
        output = ""
        for h in range(4):
            for c in range(4): output += str(int(state[h][r][c][0][0]))
            output += " | "
        logfile.write(output + "\n")


def get_pattern(state, player):
    opponent = -player
    corner = [0, 0]
    two = [0, 0]
    three = [0, 0]
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
        for c in range(4):
            cnt = [0, 0]
            for r in range(4):
                if state[h, r, c]: cnt[f(state[h, r, c])] += 1
            if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
            if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
            if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
            if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
        cnt = [0, 0]
        for i in range(4):
            if state[h, i, i]: cnt[f(state[h, i, i])] += 1
        if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
        cnt = [0, 0]
        for i in range(4):
            if state[h, i, 3 - i]: cnt[f(state[h, i, 3 - i])] += 1
        if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0: three[1] += 1

    for r in range(4):
        for c in range(4):
            cnt = [0, 0]
            for h in range(4):
                if state[h, r, c]: cnt[f(state[h, r, c])] += 1
            if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
            if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
            if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
            if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
        cnt = [0, 0]
        for i in range(4):
            if state[i, r, i]: cnt[f(state[i, r, i])] += 1
        if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
        cnt = [0, 0]
        for i in range(4):
            if state[i, r, 3 - i]: cnt[f(state[i, r, 3 - i])] += 1
        if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
    for c in range(4):
        cnt = [0, 0]
        for i in range(4):
            if state[i, i, c]: cnt[f(state[i, i, c])] += 1
        if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
        cnt = [0, 0]
        for i in range(4):
            if state[i, 3 - i, c]: cnt[f(state[i, 3 - i, c])] += 1
        if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
    cnt = [0, 0]
    for i in range(4):
        if state[i, i, i]: cnt[f(state[i, i, i])] += 1
    if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
    if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
    if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
    if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
    cnt = [0, 0]
    for i in range(4):
        if state[i, i, 3 - i]: cnt[f(state[i, i, 3 - i])] += 1
    if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
    if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
    if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
    if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
    cnt = [0, 0]
    for i in range(4):
        if state[3 - i, i, i]: cnt[f(state[3 - i, i, i])] += 1
    if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
    if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
    if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
    if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
    cnt = [0, 0]
    for i in range(4):
        if state[i, 3 - i, i]: cnt[f(state[i, 3 - i, i])] += 1
    if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
    if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
    if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
    if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
    pattern = np.reshape(np.array([corner[0], corner[1], two[0], two[1], three[0], three[1]]), [1, 6])
    return pattern


def decode_action(action_num):
    action = [0, 0]
    for i in range(2):
        action[i] = action_num % 4
        action_num //= 4
    return action


def TD(v, v_, R, alpha, gamma):
    return v + alpha * (R + gamma * v_ - v)
