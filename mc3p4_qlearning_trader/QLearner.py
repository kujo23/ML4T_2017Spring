"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand
import time
import math

#3

class QLearner(object):
    def author(self):
        return 'llee81'

    def __init__(self, \
                 num_states=100, \
                 num_actions=4, \
                 alpha=0.2, \
                 gamma=0.9, \
                 rar=0.5, \
                 radr=0.99, \
                 dyna=0, \
                 verbose=False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0
        self.a = 0
        self.q = np.random.rand(num_states, num_actions)
        self.q = (self.q * 2.) - 1.
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        # dyna
        self.t_group_count = np.zeros((num_states, num_actions))
        self.t_count = np.zeros((num_states, num_actions, num_states))
        self.t_prob = np.zeros((num_states, num_actions, num_states))
        self.r_prime = np.zeros((num_states, num_actions))
        self.dyna_ct = 0

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = np.argmax(self.q[s, :]) if rand.random() > self.rar else  rand.randint(0, self.num_actions - 1)
        if self.verbose: print "querysetstate s =", s, "a =", action, ' q:', self.q[self.s, self.a]

        self.a = action
        return action

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        action = np.argmax(self.q[s_prime, :]) if rand.random() > self.rar else  rand.randint(0,
                                                                                              self.num_actions - 1)
        if self.verbose: print "query s =", self.s, " a =", action, " r =", r, " s_prime =", s_prime, ' q:', self.q[
            self.s, self.a]

        # updates
        update1 = (1 - self.alpha) * self.q[self.s, self.a]
        update2 = self.alpha * (r + self.gamma * np.max(self.q[s_prime, :]))
        self.rar = self.rar * self.radr
        self.q[self.s, self.a] = update1 + update2

        # dyna
        if self.dyna > 0:
            self.t_group_count[self.s, self.a] = self.t_group_count[self.s, self.a] + 1
            self.t_count[self.s, self.a, s_prime] = self.t_count[self.s, self.a, s_prime] + 1
            self.t_prob[self.s, self.a, s_prime] = self.t_count[self.s, self.a, s_prime] / self.t_group_count[
                self.s, self.a]

            # train R
            self.r_prime[self.s, self.a] = (1 - self.alpha) * self.r_prime[self.s, self.a] + self.alpha * r

            for i in range(self.dyna):
                a_dyna = rand.randint(0, self.num_actions - 1)
                s_dyna = rand.randint(0, self.num_states - 1)
                if self.t_group_count[s_dyna, a_dyna] > 0:  # has data to update
                    prob_s_prime = np.random.rand()
                    s_prime_dyna = 0
                    cumm_prob = 0.
                    s_prime_dyna = np.argmax(self.t_prob[s_dyna, a_dyna, :])
                    '''
                    for prob in self.t_prob[s_dyna,a_dyna,:]:
                        cumm_prob = cumm_prob + prob
                        if prob_s_prime<=cumm_prob:
                            #print 'BREAK s_prime_dyna: ', s_prime_dyna , ' prob_s_prime:',prob_s_prime,' cumm_prob:',cumm_prob,' prob: ',prob
                            break
                        s_prime_dyna = s_prime_dyna + 1
                    '''
                    update1 = (1 - self.alpha) * self.q[s_dyna, a_dyna]
                    update2 = self.alpha * (
                    self.r_prime[s_dyna, a_dyna] + self.gamma * np.max(self.q[s_prime_dyna, :]))
                    self.q[s_dyna, a_dyna] = update1 + update2

        # dyna end

        self.s = s_prime
        self.a = action
        return action

        # dyna end

        self.s = s_prime
        self.a = action
        return action


if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
