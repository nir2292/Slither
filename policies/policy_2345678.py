from policies import base_policy as bp
import numpy as np
import pickle

FRAME_SIZE = 8

class Policy2354678(bp.Policy):


    def cast_string_args(self, policy_args):
        policy_args['Q'] = float(policy_args['Q']) if 'Q' in policy_args else np.zeros([len(self.ACTIONS), self.board_size[0], self.board_size[1]])
        return policy_args

    def init_run(self):
        #initiate Q table actions*states
        self.statesSize = self.board_size[0]*self.board_size[1]
        #self.Q = np.zeros([len(self.ACTIONS), self.board_size[0], self.board_size[1]])
        self.turnCount = 0
        self.CurrState = []
        self.LearnRate = 0.8
        self.discount = 0.9
        self.CurrState = (0,0)
        self.CurrMove = 0


        try:
            self.Q = pickle.load(open(self.load_from))
        except IOError:
            self.Q = np.zeros([len(self.ACTIONS), self.board_size[0], self.board_size[1]])
        print('d')
        #self.state = state

    def learn(self, reward, t):
        currQ = self.Q[self.CurrMove, self.CurrState[0], self.CurrState[1]]
        temp = currQ + self.LearnRate*(reward + self.discount*np.max(self.Q[:, self.CurrState[0], self.CurrState[1]]) - currQ)
        self.Q[self.CurrMove, self.CurrState[0], self.CurrState[1]] = temp


    def act(self, t, state, player_state):
        head_pos = player_state['chain'][-1]
        self.CurrState = head_pos
        #na = 0
        #a = bp.Policy.ACTIONS[na]
        na = np.argmax(self.Q[:, head_pos[0], head_pos[1]] + np.random.randn(1, len(self.ACTIONS)))# * (1. / (self.turnCount++ + 1)))
        a = bp.Policy.ACTIONS[na]
        self.CurrState = head_pos.move(bp.Policy.TURNS[player_state['dir']][a])
        self.CurrMove = na
        self.turnCount += 1
        return a

    def get_state(self):
        return self.Q