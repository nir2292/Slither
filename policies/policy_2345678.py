from policies import base_policy as bp
import numpy as np
import pickle
from queue import *


FRAME_SIZE = 8

class Policy2354678(bp.Policy):


    def cast_string_args(self, policy_args):
        self.Q = {}

        if 'load_from' in policy_args:
            try:
                self.Q = pickle.load(open(policy_args['load_from'], 'rb'))
            except IOError:
                self.Q = {}

        #policy_args['Q'] = policy_args['Q'] if 'Q' in policy_args else {}

        #  -m2 -P Policy2354678(load_from=states/140631070615816.1.Policy2354678.state.pkl);Policy2354678(load_from=states/140631070615816.2.Policy2354678.state.pkl) -D 3000
        return policy_args

    def init_run(self):
        #initiate Q table actions*states
        self.statesSize = self.board_size[0]*self.board_size[1]
        #self.Q = np.zeros([len(self.ACTIONS), self.board_size[0], self.board_size[1]])
        #self.Q = {}

        #-m2 -P Policy2354678(load_from=states/139713257744536.2.Policy2354678.state.pkl) -D 3000

        self.turnCount = 0

        self.factor = 10
        self.curr_state = 0
        self.curr_intention = 0
        self.last_state = 0
        self.last_intention = 0

        self.intentions = [0, 1]

        self.epsilon = 0.5
        self.alpha = 0.3
        self.gamma = 0.9

        self.LearnRate = 0.8
        self.discount = 0.9
        # self.CurrState = (0,0)
        self.CurrMove = 0




    # def learn(self, reward, t):
    #     currQ = self.Q[self.CurrMove, self.CurrState[0], self.CurrState[1]]
    #     temp = currQ + self.LearnRate*(reward + self.discount*np.max(self.Q[:, self.CurrState[0], self.CurrState[1]]) - currQ)
    #     self.Q[self.CurrMove, self.CurrState[0], self.CurrState[1]] = temp



    def calc_distance(self, x1, x2, y1, y2):
        dist = np.math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist


    def get_act(self, towards, obj_x, obj_y, head_pos, reduced_head_pos, player_state, reduced_state):
        """
        if towards == true : calc the next step towards the object
        else: randomely choose one of the 2 other ways
        :return:
        """
        dis = self.calc_distance(obj_x, reduced_head_pos[0], obj_y, reduced_head_pos[1])

        zero_flag, one_flag, two_flag = False, False, False


        while True:
            i = np.random.randint(3)
            if i == 0: zero_flag = True
            if i == 1: one_flag = True
            if i == 2: two_flag = True

            r, c = head_pos.move(bp.Policy.TURNS[player_state['dir']][bp.Policy.ACTIONS[i]])

            x_fact = (1 if (head_pos[0] < r) else 0) + (-1 if (head_pos[0] > r) else 0)
            y_fact = (1 if (head_pos[1] < c) else 0) + (-1 if (head_pos[1] > c) else 0)

            if towards:
                #TODO CHECK THIS OUT
                if self.calc_distance(obj_x, reduced_head_pos[0] + x_fact, obj_y, reduced_head_pos[1] + y_fact) < dis:# and \
                                # reduced_state[reduced_head_pos[0] + x_fact][reduced_head_pos[1] + y_fact] != reduced_state[self.factor][self.factor]:
                    return i
                elif zero_flag and one_flag and two_flag: return 2
            else:
                if self.calc_distance(obj_x, reduced_head_pos[0] + x_fact, obj_y, reduced_head_pos[1] + y_fact) > dis:# and \
                                # reduced_state[reduced_head_pos[0] + x_fact][reduced_head_pos[1] + y_fact] != reduced_state[self.factor][self.factor]:
                    return i
                elif zero_flag and one_flag and two_flag: return 2


    def getQ(self, state_obj, intention):
        return self.Q.get((state_obj, intention), 0.0)
    # DONE

    def learnQ(self, state_obj, intention, reward, value):
        oldv = self.Q.get((state_obj, intention), None)
        if oldv is None:
            self.Q[(state_obj, intention)] = reward
        else:
            self.Q[(state_obj, intention)] = oldv + self.alpha * (value - oldv)

    # DONE
    def learn(self, reward, t):

        q_lohpatli = self.getQ(self.curr_state, self.curr_intention)
        self.learnQ(self.last_state, self.last_intention, reward, reward + self.gamma * q_lohpatli)

    # def act(self, t, state, player_state):
    #     head_pos = player_state['chain'][-1]
    #     self.CurrState = head_pos
    #     #na = 0
    #     #a = bp.Policy.ACTIONS[na]
    #     na = np.argmax(self.Q[:, head_pos[0], head_pos[1]] + np.random.randn(1, len(self.ACTIONS)))# * (1. / (self.turnCount++ + 1)))
    #     a = bp.Policy.ACTIONS[na]
    #     self.CurrState = head_pos.move(bp.Policy.TURNS[player_state['dir']][a])
    #     self.CurrMove = na
    #     self.turnCount += 1
    #     return a

        # DONE
    # def get_act(self, towards, reduced_state, obj_x, obj_y, head_pos, player_state):
    #     """
    #     if towards == true : calc the next step towards the object
    #     else: randomely choose one of the 2 other ways
    #     :return:
    #     """
    #     dis = self.calc_distance(obj_x, head_pos[0], obj_y, head_pos[1])
    #     while True:
    #         i = np.random.randint(2)
    #         r, c = head_pos.move(bp.Policy.TURNS[player_state['dir']][i])
    #         if towards:
    #             if self.calc_distance(obj_x, r, obj_y, c) < dis:
    #                 return i
    #         else:
    #             if self.calc_distance(obj_x, r, obj_y, c) > dis:
    #                 return i

    # Half Done - logic for other players
    def get_closest_object_number(self, state, head_pos):
        """
        :return:
        """
        # TODO - put logic for encountering other players


        closest_obj_num, indx_x, indx_y = 0, 0, 0
        min_dis = abs(state.shape[0] * state.shape[1])

        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i][j] != 0 and state[i][j] != state[head_pos[0]][head_pos[1]]:
                    dis = self.calc_distance(i, head_pos[0], j, head_pos[1])
                    if dis < min_dis:
                        min_dis = dis
                        closest_obj_num, indx_x, indx_y = state[i][j], i, j
        return closest_obj_num, indx_x, indx_y

    # DONE
    def update_state_act(self, curr_state, curr_act, t):
        """
        :param state:
        :return:
        """
        self.last_state = self.curr_state
        self.last_intention = self.curr_intention
        self.curr_state = curr_state
        self.curr_intention = curr_act

    def get_reduced_state(self, state, head_pos):
        """

        :param state: matrix
        :param head_pos: (x,y)
        :return: a 5 * 5 or smaller board around the agents head
        """
        x_flag = False
        y_flag = False

        factor = self.factor

        padState = np.concatenate((state[:, -factor:state.shape[1]], state, state[:, 0:factor]), axis=1)
        padState = np.concatenate((padState[-factor:padState.shape[0] , :], padState, padState[0:factor , :]), axis=0)

        reduced_state = {}
        # TODO - change it to get out of board
        lower_x = head_pos[0] - factor
        # if (lower_x < 0): lower_x = 0
        lower_y = head_pos[1] - factor
        # if (lower_y < 0): lower_y = 0

        # TODO - insert logic for getting out of board
        higher_x = head_pos[0] + factor
        # if (higher_x > state.shape[0]):
        #     higher_x = state.shape[0]
        higher_y = head_pos[1] + factor
        # if (higher_y > state.shape[1]): higher_y = state.shape[1]

        # TODO check, maybe the other way around
        #reduced_state = state[lower_x:higher_x, lower_y:higher_y]
        reduced_state = padState[(head_pos[0]%state.shape[0]):((head_pos[0]%state.shape[0])+2*factor+1), (head_pos[1]%state.shape[1]):((head_pos[1]%state.shape[1])+2*factor+1)]

        return reduced_state

    def act(self, t, state, player_state):

        head_pos = player_state['chain'][-1]
        # look only on the 10 * 10 (or smaller) patch around agents head
        reduced_state = self.get_reduced_state(state, head_pos)
        reduced_head_pos = (self.factor,self.factor)
        # state = state

        # state of board will be defined as an id of the closest object number
        state_obj, indx_x, indx_y = self.get_closest_object_number(reduced_state, reduced_head_pos)
        chosen = 0
        if state_obj==0:
            return bp.Policy.ACTIONS[2]
        # Acting will be devided to 2 : move towards an object or m
        # Explore or exploit:
        if self.epsilon > 0.1: self.epsilon -= 0.001

        #TODO left right instead of closer further
        if np.random.rand(1) < self.epsilon:
            # go towards the object or avoid it : randomly
            chosen = 1
            action = self.get_act(chosen, indx_x, indx_y, head_pos, reduced_head_pos, player_state, reduced_state)

            #choose random turn
            #action = np.random.randint(3)
        else:
            q = [self.getQ(state_obj, intent) for intent in self.intentions]
            maxQ = max(q)
            count = q.count(maxQ)

            #i = 0
            if count > 1:
                i = np.random.randint(1)
            else:
                i = q.index(maxQ)
            # Choose weather to go towards the object or avoid it
            chosen = self.intentions[i]
            action = self.get_act(chosen, indx_x, indx_y, head_pos, reduced_head_pos, player_state, reduced_state)
        self.update_state_act(state_obj, chosen, t)
        return bp.Policy.ACTIONS[action]

    def get_state(self):
        return self.Q