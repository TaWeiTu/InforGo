from InforGo.process.schema import Schema as schema
from InforGo.bingo import Bingo as State
from InforGo.util import decode_action


class Bot(object):

    def __init__(self, player):
        self.player = player
        self.opponent = -player

    def generate_action(self, state):
        for i in range(4):
            for j in range(4):
                env = State(state)
                if env.valid_action(i, j):
                    env.place(i, j)
                    if env.win(self.player): return i, j
        for i in range(4):
            for j in range(4):
                env = State(state)
                env.player *= -1
                if env.valid_action(i, j):
                    env.place(i, j)
                    if env.win(self.opponent): return i, j
        return random.randint(0, 3), random.randint(0, 3)


class Tester(schema):
    
    def __init__(self, **kwargs):
        super().__init__(kwargs['n_epoch'], kwargs['player_len'], kwargs['pattern_len'], kwargs['n_hidden_layer'], kwargs['n_node_hidden'],
                         kwargs['activation_fn'], kwargs['learning_rate'], kwargs['directory'], kwargs['alpha'], kwargs['gamma'], kwargs['lamda'],
                         kwargs['search_depth'], kwargs['play_first'])
        self.player = 1 if kwargs['play_first'] else -1
        self.bot = Bot(-self.player)

    def test(self):
        env = State()
        victory = 1
        for epoch in range(self.n_epoch):
            s = env.get_initial_state()
            player = 1
            while True:
                action = self.get_action(s, player)
                flag, s, R = env.take_action(*action, player)
                if flag == player: break
                player = -player
            if player == self.player: victory += 1
        return victory / self.n_epoch

    def get_action(self, state, player):
        if player == self.player: return self.AI.get_action(state)
        return self.bot.generate_action(state)
