# DONE
def update_state_act(self, curr_state, curr_intention, t):
    """
    :param state:
    :return:
    """
    # Append state and intention to t_table dict. reward will be updated in learn.
    self.t_table[t%100] = (curr_state,curr_intention, 0)



# DONE
def learn(self, reward, t):

    last_state, last_intention, last_reward = 0, 0, 0
    curr_state = self.t_table.get(t%100)[0]
    curr_intention = self.t_table.get(t%100)[1]

    if t > 0 :
        last_state = self.t_table.get((t-1)%100)[0]
        last_intention = self.t_table.get((t-1)%100)[1]
        # Update t table reward
        self.t_table[(t-1)%100][2] = reward
        last_reward = self.t_table.get((t-1)%100)[2]

    q_lohpatli = self.getQ(curr_state, curr_intention)
    self.learnQ(last_state, last_intention, last_reward, last_reward + self.gamma * q_lohpatli)
