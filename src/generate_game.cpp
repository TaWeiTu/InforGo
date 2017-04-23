#include <bits/stdc++.h>
using namespace std;

class Bingo {
    private:
        vector<vector<vector<int>>> board;
        vector<vector<int>> height;
        void resize_board() {
            board.resize(4);
            for (int i = 0; i < 4; ++i) {
                board[i].resize(4);
                for (int j = 0; j < 4; ++j) {
                    board[i][j].resize(4);
                    for (int k = 0; k < 4; ++k) board[i][j][k] = 0;
                }
            }
        }
        void resize_height() {
            height.resize(4);
            for (int i = 0; i < 4; ++i) {
                height[i].resize(4);
                for (int j = 0; j < 4; ++j) height[i][j] = 0;
            }
        }
    public:
        Bingo() {
            resize_board();
            resize_height();
        }
        Bingo(vector<vector<vector<int>>> s) {
            resize_board(); resize_height();
            for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 4; ++k) board[i][j][k] = s[i][j][k];
            }
            for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 4; ++k) if (board[k][i][j]) height[i][j] = k + 1;
            }
        }
        vector<vector<vector<int>>> get_state() { return board; }
        vector<vector<int>> get_height() { return height; }
        bool valid_action(int row, int col) {
            return height[row][col] < 4;
        }
        void place(int row, int col, int player) {
            if (!valid_action(row, col)) return;
            int h = height[row][col];
            board[h][row][col] = player;
            height[row][col]++;
        }
        bool win(int player) {
            bool ret;
            for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
                ret = true;
                for (int k = 0; k < 4; ++k) if (board[i][j][k] != player) ret = false;
                if (ret) return true;
            }
            for (int i = 0; i < 4; ++i) for (int k = 0; k < 4; ++k) {
                ret = true;
                for (int j = 0; j < 4; ++j) if (board[i][j][k] != player) ret = false;
                if (ret) return true;
            }
            for (int j = 0; j < 4; ++j) for (int k = 0; k < 4; ++k) {
                ret = true;
                for (int i = 0; i < 4; ++i) if (board[i][j][k] != player) ret = false;
                if (ret) return true;
            }
            for (int i = 0; i < 4; ++i) {
                ret = true;
                for (int j = 0; j < 4; ++j) if (board[i][j][j] != player) ret = false;
                if (ret) return true;
                ret = true;
                for (int j = 0; j < 4; ++j) if (board[i][j][3 - j] != player) ret = false;
                if (ret) return true;
            }
            for (int j = 0; j < 4; ++j) {
                ret = true;
                for (int i = 0; i < 4; ++i) if (board[i][j][i] != player) ret = false;
                if (ret) return true;
                ret = true;
                for (int i = 0; i < 4; ++i) if (board[i][j][3 - i] != player) ret = false;
                if (ret) return true;
            }
            for (int k = 0; k < 4; ++k) {
                ret = true;
                for (int i = 0; i < 4; ++i) if (board[i][i][k] != player) ret = false;
                if (ret) return true;
                ret = true;
                for (int i = 0; i < 4; ++i) if (board[i][3 - i][k] != player) ret = false;
                if (ret) return true;
            }
            ret = true;
            for (int i = 0; i < 4; ++i) if (board[i][i][i] != player) ret = false;
            if (ret) return true;
            ret = true;
            for (int i = 0; i < 4; ++i) if (board[i][i][3 - i] != player) ret = false;
            if (ret) return true;
            ret = true;
            for (int i = 0; i < 4; ++i) if (board[i][3 - i][i] != player) ret = false;
            if (ret) return true;
            ret = true;
            for (int i = 0; i < 4; ++i) if (board[3 - i][i][i] != player) ret = false;
            if (ret) return true;
            return false;
        }
        bool full() {
            for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) if (height[i][j] < 4) return false;
            return true;
        }
};

long long id = 0;

void generate_game(Bingo, int, string);
pair<int, int> decode(int);

int main() {
    generate_game(Bingo(), 1, "");
    return 0;
}

void generate_game(Bingo bingo, int player, string record) {
    if (bingo.full() || bingo.win(1) || bingo.win(2)) {
        fstream file;
        string name;
        stringstream ss; ss << id; ss >> name;
        ++id;
        file.open("../Data/record/generator/" + name, ios::out);
        record += "-1 -1 -1\n";
        file << record;
        file.close();
        return;
    }
    vector<vector<vector<int>>> board = bingo.get_state();
    vector<vector<int>> height = bingo.get_height();
    vector<int> vec;
    for (int i = 0; i < 16; ++i) vec.push_back(i);
    random_shuffle(vec.begin(), vec.end());
    for (int i : vec) {
        pair<int, int> p = decode(i);
        if (!bingo.valid_action(p.first, p.second)) continue;
        Bingo new_bingo = Bingo(board);
        string new_record = record;
        new_record += (char)('0' + height[p.first][p.second]); new_record += ' ';
        new_record += (char)('0' + p.first); new_record += ' '; new_record += (char)('0' + p.second); new_record += '\n';
        new_bingo.place(p.first, p.second, player);
        generate_game(new_bingo, (player == 1 ? 2 : 1), new_record);
    }
}

pair<int, int> decode(int action) {
    pair<int, int> ret;
    ret.first = action % 4;
    action /= 4;
    ret.second = action % 4;
    return ret;
}
