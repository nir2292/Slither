from policies import base_policy as bp

import random
import numpy as np

############################################
#  TODO Assumption about the ex:
#
#
#
#
#
############################################


class AvoidCollisions(bp.Policy):

    def cast_string_args(self, policy_args):
        policy_args['example'] = int(policy_args['example']) if 'example' in policy_args else 0
        return policy_args

    def init_run(self):
        #print(self.example)
        self.r_sum = 0

        self.firstLearn = True

        self.q = {}

        self.epsilon = 0.1
        self.alpha = 0.2
        self.gamma = 0.9
        self.actions = {1,0}
        self.last_state = {}
        self.last_action = 0
        self.curr_state = {}
        self.curr_action = 0

    # DONE
    def getQ(self, state_obj, action):
        return self.q.get((state_obj, action), 0.0)

    # DONE
    def learnQ(self, state_obj, action, reward, value):
        oldv = self.q.get((state_obj, action), None)
        if oldv is None:
            self.q[(state_obj, action)] = reward
        else:
            self.q[(state_obj, action)] = oldv + self.alpha * (value - oldv)

    # DONE
    def learn(self, reward, t):

        if t % 100 == 0:
            self.log(str(self.r_sum), 'value')
            self.r_sum = 0
        else:
            self.r_sum += reward

        if not self.firstLearn:
            qnext = self.getQ(self.curr_state, self.curr_action)
            self.learnQ(self.last_state, self.last_action, reward, reward + self.gamma * qnext)
        if self.firstLearn: self.firstLearn = False

    # DONE
    def calc_distance(self, x1, x2, y1, y2):
        dist = np.math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist

    # DONE
    def get_act(self, towards, reduced_state, obj_x, obj_y, head_pos, player_state):
        """
        if towards == true : calc the next step towards the object
        else: randomely choose one of the 2 other ways
        :return:
        """
        dis = self.calc_distance(obj_x, head_pos[0], obj_y, head_pos[1])
        while True:
            i = np.random.randint(2)
            r, c = head_pos.move(bp.Policy.TURNS[player_state['dir']][i])
            if towards:
                if self.calc_distance(obj_x, r, obj_y, c) < dis:
                    return i
            else:
                if self.calc_distance(obj_x, r, obj_y, c) > dis:
                    return i

    # Half Done - logic for other players
    def get_closest_object_number(self, state, head_pos):
        """
        :return:
        """
        # TODO - put logic for encountering other players


        closest_obj_num, indx_x, idx_y = 0, 0, 0
        min_dis = state.shape[0] * state.shape[1]

        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i][j] < 0 :
                    dis = self.calc_distance(i , head_pos[0], j, head_pos[1])
                    if dis < min_dis:
                        min_dis = dis
                        closest_obj_num, indx_x, indx_y = state[i][j], i, j
        return closest_obj_num, indx_x, indx_y


    # DONE
    def update_state_act(self, curr_state, curr_act):
        """
        :param state:
        :return:
        """
        self.last_state = self.curr_state
        self.last_action =  self.curr_action
        self.curr_state = curr_state
        self.curr_action = curr_act


    def get_reduced_state(self, state, head_pos):
        """

        :param state: matrix
        :param head_pos: (x,y)
        :return: a 5 * 5 or smaller board around the agents head
        """
        x_flag = False
        y_flag = False

        reduced_state = {}
        # TODO - change it to get out of board
        lower_x = head_pos[0] - 5
        if(lower_x < 0): lower_x = 0
        lower_y = head_pos[1] - 5
        if(lower_y < 0): lower_y = 0

        # TODO - insert logic for getting out of board
        higher_x = head_pos[0] + 5
        if(higher_x > state.shape[0]):
            higher_x = state.shape[0]
        higher_y = head_pos[1] + 5
        if(higher_y > state.shape[1]): higher_y = state.shape[1]

        # TODO check, maybe the other way around
        reduced_state = state[lower_x:higher_x][lower_y:higher_y]
        return reduced_state

    def act(self, t, state, player_state):

        head_pos = player_state['chain'][-1]
        # look only on the 10 * 10 (or smaller) patch around agents head
        reduced_state = self.get_reduced_state(state, head_pos)
        hed_pos = (5,5)
        state = state

        # state of board will be defined as an id of the closest object number
        state_obj, indx_x, indx_y = self.get_closest_object_number(reduced_state, head_pos)


        # Acting will be devided to 2 : move towards an object or m
        # Explore or exploit:
        if np.random.rand(1) < self.epsilon:
            # go towards the object or avoid it : randomly
            action = self.get_act(np.random.randint(1), indx_x, indx_y, head_pos, player_state)
            self.update_state_act(state_obj, action)
            return action
        else:
            q = [self.getQ(state_obj, ac) for ac in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)
            # Choose weather to go towards tthe object or avoid it
            action = self.get_act(self.actions[i], indx_x, indx_y, head_pos, player_state)
        self.update_state_act(state_obj, action)
        return action


        # Maybe epsilon -= 0.001 or something

    def get_state(self):
        return None
