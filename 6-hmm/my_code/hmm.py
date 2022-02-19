from __future__ import print_function
import json
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
            ==> [0.7 0.3]
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
            ==> [[0.8 0.2]
                [0.4 0.6]]
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
            ==> [[0.5 0.  0.4 0.1]
                [0.5 0.1 0.2 0.2]]
        - obs_dict: A dictionary mapping each observation symbol to its index
            ==> {'A': 0, 'C': 1, 'T': 3, 'G': 2}
            ==> index each value
            ==> convert states back to this dict

        - state_dict: A dictionary mapping each state to its index
            ==> {'1': 0, '2': 1}
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L
            ==> loop through
            ==> ['A' 'G' 'C' 'C' 'T' 'A' 'C'] (7)

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
                 ==> alpha_s (t) = b_s,xt * summation (a_s',s * alpha(t-1))
        """
        S = len(self.pi)
            # ==> 2
        L = len(Osequence)
            # ==> 7
        O = self.find_item(Osequence)
            # ==> [0, 2, 1, 1, 3, 0, 1] (7)
        alpha = np.zeros([S, L])
            # ==> (2, 7)
        ######################################################
        # TODO: compute and return the forward messages alpha
        ######################################################
        # [L10 | P28]
        # print("self.pi : ", self.pi)
        # print("self.A : ", self.A)
        # print("self.B : ", self.B)
        # print(">>>self.obs_dict : ", self.obs_dict)
        # print(">>>self.state_dict : ", self.state_dict)
        # print(">>>Osequence : ", Osequence)
        # print(">>>S : ", S)
        # print(">>>L : ", L)
        # print(">>>O : ", O)
        # print(">>>alpha[:,0] : ", alpha[:,0])

        # get observation A
        get_ob = Osequence[0]
        # print(">>>get_ob : ", get_ob)

        # map key of obs_dict to observation
        k_o_map = self.obs_dict[get_ob]
        # print(">>>self.obs_dict[get_ob] : ", k_o_map)

        # get each column of B
        get_B = self.B[:, k_o_map]
        # print("get_b : ", get_b)

            # assign alpha columns to pi * B
        alpha[:,0] = self.pi * get_B
        # print("alpha[:,0] : ", alpha[:,0])

        # loop over each obeservation
        for t in range(1, L):

            # loop over each state
            for s in range(S):
                # get last column of alpha
                last_alpha = alpha[:, t - 1]
                # print("last_alpha : ", last_alpha)
                # get last column of A @s
                get_A_s = self.A[:, s]
                # print("get_A_s : ", get_A_s)

                A_dot_alp = np.dot(get_A_s, last_alpha)

                # self.B is broken down above
                get_B_s_x = self.B[s, self.obs_dict[Osequence[t]]]

                alpha[s,t] = get_B_s_x * A_dot_alp
        return alpha



    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        beta = np.zeros([S, L])
        #######################################################
        # TODO: compute and return the backward messages beta
        #######################################################
        # [L10 | P30]
        for s in range(S):
            beta[s, L - 1] = 1

        for t in reversed(range(L - 1)) :
            for st in range(S) :
                beta_s_t = beta[s, t + 1]
                A_s_s = self.A[st, s]
                # get_B =
                # beta[st, t] = sum([beta_s_t * A_s_s * self.B[s, self.obs_dict[Osequence[t + 1]]] for s in range(S)])
                beta[st, t] = sum([beta[s, t + 1] * self.A[st, s] * self.B[s, self.obs_dict[Osequence[t + 1]]] for s in range(S)])
        return beta


    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """

        #####################################################
        # TODO: compute and return prob = P(X_{1:T}=x_{1:T})
        #   using the forward/backward messages
        #####################################################
        # set pr() = 0
        get_probabiliy = 0
        # update alpha by calling the forward method
        get_alpha = self.forward(Osequence)

        # take the 2nd to last column & sum over them for the pr()
        get_probabiliy = sum(get_alpha[:, -1])
        return get_probabiliy

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
		           (note that this is gamma[i, t-1] instead of gamma[i, t])
        """
        ######################################################################
        # TODO: compute and return gamma using the forward/backward messages
        ######################################################################
        # [L10 | P31]
        S = len(self.pi)
        L = len(Osequence)
        get_probability = np.zeros([S, L])

        # call backward message
        get_beta = self.backward(Osequence)
        # call forward message
        get_alpha = self.forward(Osequence)

        # take the 2nd to last column & sum over them for the pr()
        denom = sum(get_alpha[:, -1])

        # loop over observations
        for t in range(L):
            for s in range(S):

                # a_s,t * b_s,t
                get_a_b = np.multiply(get_alpha[s ,t], get_beta[s, t])

                # pr() of observatng a sequence
                get_probability[s , t] = np.divide(get_a_b, denom)

        return get_probability
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array where prob[i, j, t-1] =
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        S = len(self.pi)
        L = len(Osequence)
        probability = np.zeros([S, S, L - 1])
        #####################################################################
        # TODO: compute and return prob using the forward/backward messages
        #####################################################################
        get_beta = self.backward(Osequence)
        get_alpha = self.forward(Osequence)
        denom = np.sum(get_alpha[:, -1])

        for t in range(L - 1):
            for i in range(S):
                for j in range(S):
                    probability[i , j, t] = get_alpha[i, t] * self.A[i, j] * get_beta[j, t + 1] * self.B[j, self.obs_dict[Osequence[t + 1]]] / denom
        return probability

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        # path = []
        states = []
        ################################################################################
        # TODO: implement the Viterbi algorithm and return the most likely state path
        ################################################################################
        S = len(self.pi)
        N = len(Osequence)
        get_del = np.zeros([S, N])
        Delta = np.zeros([S, N], dtype="int")

        # get observation
        get_ob = Osequence[0]
        # print(">>>get_ob : ", get_ob)

        # map key of obs_dict to observation
        k_o_map = self.obs_dict[get_ob]
        # print(">>>self.obs_dict[get_ob] : ", k_o_map)

        get_B = self.B[:, k_o_map]

        get_del[:, 0] = np.multiply(self.pi, get_B)

        # loop over len of sequence
        for t in range(1, N) :

            for s in range(S) :
                get_A = self.A[:, s]

                # B @ s, observations t * max value of )A @ column s * delta @ column 1)
                get_del[s, t] = self.B[s, self.obs_dict[Osequence[t]]] * np.max(get_A * get_del[:, t - 1])

                # delta_s,t = argmax (A @ column s) * delta
                Delta[s, t] = np.argmax(self.A[:, s] * get_del[:, t - 1])

        max_z = np.argmax(get_del[:, N - 1])

        # append max of z to the path
        states.append(max_z)


        for t in range(N - 1, 0, -1) :
            max_z = Delta[max_z, t]

            # append max of z of delta @ t
            states.append(max_z)

        states = states[: : -1]

        path = [0] * len(states)

        for s_d in self.state_dict :
            # loop over each state in list to get specific state
            for state in range(len(states)) :
                # set this current state to the value of the s_d in the state_dict
                if states[state] == self.state_dict[s_d] :
                    # update path
                    path[state] = s_d

        return path


    #DO NOT MODIFY CODE BELOW
    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
