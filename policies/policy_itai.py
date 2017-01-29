from policies import base_policy as bp

import random
import numpy as np



class AvoidCollisions(bp.Policy):

    def cast_string_args(self, policy_args):
        policy_args['example'] = int(policy_args['example']) if 'example' in policy_args else 0
        return policy_args

    def init_run(self):
        #print(self.example)
        self.r_sum = 0

        self.q = {}

        self.epsilon = 0.1
        self.alpha = 0.2
        self.gamma = 0.9
        self.actions = bp.Policy.ACTIONS
        self.last_state = {}
        self.last_action = 0
        self.curr_state = {}
        self.curr_action = 0

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def learn(self, reward, t):

        if t % 100 == 0:
            self.log(str(self.r_sum), 'value')
            self.r_sum = 0
        else:
            self.r_sum += reward

        qnext = self.getQ(self.curr_state, self.curr_action)
        self.learnQ(self.last_state, self.last_action, reward, reward + self.gamma * qnext)


    def get_act(self, towards):
        """
        if towards == true : calc the next step towards the object
        else: randomely choose one of the 2 other ways
        :return:
        """
        # TODO implement

    def get_closest_object_number(state, player_state):
        """
        :return:
        """
        for ()
            return 0

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
        reduced_state = {}

        lower_x = head_pos[0] - 5
        if(lower_x < 0): lower_x = 0
        lower_y = head_pos[1] - 5
        if(lower_y < 0): lower_y = 0

        higher_x = head_pos[0] + 5
        if(higher_x > state.shape[0]): higher_x = state.shape[0]
        higher_y = head_pos[1] + 5
        if(higher_y > state.shape[1]): higher_y = state.shape[1]

        # TODO check, maybe the other way around
        reduced_state = state[lower_x:higher_x][lower_y:higher_y]
        return reduced_state

    def act(self, t, state, player_state):

        head_pos = player_state['chain'][-1]
        # look only on the 10 * 10 (or smaller) patch around agents head
        reduced_state = self.get_reduced_state(state, head_pos)
        state = state


        # state of board will be defined as an id of the closest object number
        state_obj = self.get_closest_object_number(reduced_state, player_state)


        # Acting will be devided to 2 : move towards an object or m
        # Explore or exploit:
        if np.random.rand(1) < self.epsilon:
            # go towards the object or avoid it : randomely
            action = self.getAct(np.random.randint(1))
            self.update_state_act(state_obj, action)
            return action
        else:
            q = [self.getQ(state, ac) for ac in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        self.update_state_act(state_obj, action)
        return action


        #head_pos = player_state['chain'][-1]
        #action = getAct()

        """
        #a = bp.Policy.ACTIONS['CC']
        for a in [a]: #+ list(np.random.permutation(bp.Policy.ACTIONS)):
            r, c = head_pos.move(bp.Policy.TURNS[player_state['dir']][a]) % state.shape
            if state[r, c] <= 0: return a
        return a
        """


        # Maybe epsilon -= 0.001 or something

    def get_state(self):
        return None
