from InforGo.process.schema import Schema as schema
from InforGo.bingo import Bingo as State


class Runner(schema):

    def __init__(self, **kwargs):
        super().__init__(kwargs['n_epoch'], kwargs['player_len'], kwargs['pattern_len'], kwargs['n_hidden_layer'], kwargs['n_node_hidden'],
                         kwargs['activation_fn'], kwargs['learning_rate'], kwargs['directory'], kwargs['alpha'], kwargs['gamma'], kwargs['lamda'],
                         kwargs['search_depth'], kwargs['play_first'])
        self.player = 1 if kwargs['play_first'] else -1

    def run(self):
        env = State()
        player = 1
        s = env.get_initial_state()
        while True:
            action = self.get_action(s, player)
            flag, s, R = env.take_action(*action, player)
            if flag == player: break
            player = -player
        return player

    def get_action(self, state, player):
        if player == self.player: return self.AI.get_action(state)
        return self.read_action()

    def read_action(self):
        print("input")
        height, row, col = map(int, input().split())
        return row, col

