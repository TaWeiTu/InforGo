import random
import numpy as np


def plot_state(state):
    for r in range(4):
        output = ""
        for h in range(4):
            for c in range(4): output += str(int(state[h][r][c][0][0]))
            output += " | "
        print(output)

def get_pattern(state, player):
    opponent = -player
    corner = [0, 0]
    two = [0, 0]
    three = [0, 0]
    for i in range(4):
        if state[i][i][i][0][0] == player: corner[0] += 1
        elif state[i][i][i][0][0] == opponent: corner[1] += 1
        if state[i][3 - i][3 - i][0][0] == player: corner[0] += 1
        elif state[i][3 - i][3 - i][0][0] == opponent: corner[1] += 1
        if state[i][3 - i][i][0][0] == player: corner[0] += 1
        elif state[i][3 - i][i][0][0] == opponent: corner[1] += 1
        if state[i][i][3 - i][0][0] == player: corner[0] += 1
        elif state[i][i][3 - i][0][0] == opponent: corner[1] += 1

    for h in range(4):
        for r in range(4):
            cnt = [0, 0]
            for c in range(4):
                if state[h][r][c][0][0]: cnt[int(state[h][r][c][0][0]) - 1] += 1
            if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
            if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
            if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
            if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
        for c in range(4):
            cnt = [0, 0]
            for r in range(4):
                if state[h][r][c][0][0]: cnt[int(state[h][r][c][0][0]) - 1] += 1
            if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
            if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
            if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
            if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
        cnt = [0, 0]
        for i in range(4):
            if state[h][i][i][0][0]: cnt[int(state[h][i][i][0][0]) - 1] += 1
        if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
        cnt = [0, 0]
        for i in range(4):
            if state[h][i][3 - i][0][0]: cnt[int(state[h][i][3 - i][0][0]) - 1] += 1
        if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0: three[1] += 1

    for r in range(4):
        for c in range(4):
            cnt = [0, 0]
            for h in range(4):
                if state[h][r][c][0][0]: cnt[int(state[h][r][c][0][0]) - 1] += 1
            if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
            if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
            if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
            if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
        cnt = [0, 0]
        for i in range(4):
            if state[i][r][i][0][0]: cnt[int(state[i][r][i][0][0]) - 1] += 1
        if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
        cnt = [0, 0]
        for i in range(4):
            if state[i][r][3 - i][0][0]: cnt[int(state[i][r][3 - i][0][0]) - 1] += 1
        if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
    for c in range(4):
        cnt = [0, 0]
        for i in range(4):
            if state[i][i][c][0][0]: cnt[int(state[i][i][c][0][0]) - 1] += 1
        if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
        cnt = [0, 0]
        for i in range(4):
            if state[i][3 - i][c][0][0]: cnt[int(state[i][3 - i][c][0][0]) - 1] += 1
        if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
    cnt = [0, 0]
    for i in range(4):
        if state[i][i][i][0][0]: cnt[int(state[i][i][i][0][0]) - 1] += 1
    if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
    if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
    if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
    if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
    cnt = [0, 0]
    for i in range(4):
        if state[i][i][3 - i][0][0]: cnt[int(state[i][i][3 - i][0][0]) - 1] += 1
    if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
    if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
    if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
    if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
    cnt = [0, 0]
    for i in range(4):
        if state[3 - i][i][i][0][0]: cnt[int(state[3 - i][i][i][0][0]) - 1] += 1
    if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
    if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
    if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
    if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
    cnt = [0, 0]
    for i in range(4):
        if state[i][3 - i][i][0][0]: cnt[int(state[i][3 - i][i][0][0]) - 1] += 1
    if cnt[0] == 2 and cnt[1] == 0: two[0] += 1
    if cnt[1] == 2 and cnt[0] == 0: two[1] += 1
    if cnt[0] == 3 and cnt[1] == 0: three[0] += 1
    if cnt[1] == 3 and cnt[0] == 0: three[1] += 1
    pattern = np.reshape(np.array([corner[0], corner[1], two[0], two[1], three[0], three[1]]), [1, 6])
    return pattern
