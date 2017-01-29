from policies import base_policy as bp
import numpy as np
import pickle


class myPolicy(bp.Policy):

    def cast_string_args(self, policy_args):
        return policy_args

    def init_run(self):
        pass

    def learn(self, reward, t):
        pass

    def act(self, t, state, player_state):
        raise ValueError()

    def get_state(self):
        return self.state