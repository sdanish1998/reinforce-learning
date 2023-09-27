import numpy as np


class DynamicProgramming:
    def __init__(self, MDP):
        self.R = MDP.R  # |A|x|S|
        self.P = MDP.P  # |A|x|S|x|S|
        self.discount = MDP.discount
        self.nStates = MDP.nStates
        self.nActions = MDP.nActions

    ####Helpers####
    def extractRpi(self, pi):
        '''
        Returns R(s, pi(s)) for all states. Thus, the output will be an array of |S| entries.
        This should be used in policy evaluation and policy iteration.

        Parameter pi: a deterministic policy
        Precondition: An array of |S| integers, each of which specifies an action (row) for a given state s.
        '''
        return self.R[pi, np.arange(len(self.R[0]))]

    def extractPpi(self, pi):
        '''
        Returns P^pi: This is a |S|x|S| matrix where the (i,j) entry corresponds to
        P(j|i, pi(i))

        Parameter pi: a deterministic policy
        Precondition: An array of |S| integers
        '''
        return self.P[pi, np.arange(len(self.P[0]))]

    ####Value Iteration###
    def computeVfromQ(self, Q, pi):
        '''
        Returns the V function for a given Q function corresponding to a deterministic policy pi. Remember that

        V^pi(s) = Q^pi(s, pi(s))

        Parameter Q: Q function
        Precondition: An array of |S|x|A| numbers

        Parameter pi: Policy
        Preconditoin: An array of |S| integers
        '''
        return np.array([Q[s,pi[s]] for s in range(self.nStates)])


    def computeQfromV(self, V):
        '''
        Returns the Q function given a V function corresponding to a policy pi. The output is an |S|x|A| array.

        Use the bellman equation for Q-function to compute Q from V.

        Parameter V: value function
        Precondition: An array of |S| numbers
        '''
        return np.array([[self.R[a][s]
                + self.discount*np.sum([self.P[a][s][s1]*V[s1]
                for s1 in range(self.nStates)])
                for a in range(self.nActions)]
                for s in range(self.nStates)])


    def extractMaxPifromQ(self, Q):
        '''
        Returns the policy pi corresponding to the Q-function determined by

        pi(s) = argmax_a Q(s,a)


        Parameter Q: Q function
        Precondition: An array of |S|x|A| numbers
        '''
        return np.argmax(Q, axis = 1)

    def extractMaxPifromV(self, V):
        '''
        Returns the policy corresponding to the V-function. Compute the Q-function
        from the given V-function and then extract the policy following

        pi(s) = argmax_a Q(s,a)

        Parameter V: V function
        Precondition: An array of |S| numbers
        '''
        return self.extractMaxPifromQ(self.computeQfromV(V))


    def valueIterationStep(self, Q):
        '''
        Returns the Q function after one step of value iteration. The input
        Q can be thought of as the Q-value at iteration t. Return Q^{t+1}.

        Parameter Q: value function
        Precondition: An array of |S|x|A| numbers
        '''
        return np.array([[self.R[a][s]
                + self.discount * np.sum(self.P[a][s] * np.max(Q, axis=1))
                for a in range(self.nActions)]
                for s in range(self.nStates)])

    def valueIteration(self, initialQ, tolerance=0.01):
        '''
        This function runs value iteration on the input initial Q-function until
        a certain tolerance is met. Specifically, value iteration should continue to run until
        ||Q^t-Q^{t+1}||_inf <= tolerance. Recall that for a vector v, ||v||_inf is the maximum
        absolute element of v.


        This function should return the policy, value function, number
        of iterations required for convergence, and the end epsilon where the epsilon is
        ||Q^t-Q^{t+1}||_inf.

        Parameter initialQ:  Initial value function
        Precondition: array of |S|x|A| entries

        Parameter tolerance: threshold threshold on ||Q^t-Q^{t+1}||_inf
        Precondition: Float >= 0 (default: 0.01)
        '''
        currQ, epsilon, iterId = initialQ, np.inf, 0
        while True:
            newQ = self.valueIterationStep(currQ)
            epsilon = np.max(np.abs(currQ - newQ))
            currQ = newQ
            iterId = iterId + 1
            if epsilon <= tolerance: break
        pi = self.extractMaxPifromQ(currQ)
        V = self.computeVfromQ(currQ, pi)
        return pi, V, iterId, epsilon

    ### EXACT POLICY EVALUATION  ###
    def exactPolicyEvaluation(self, pi):
        '''
        Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma P^pi V^pi

        Return the value function

        Parameter pi: Deterministic policy
        Precondition: array of |S| integers
        '''
        return np.linalg.solve(np.eye(self.nStates) - self.discount*self.extractPpi(pi), self.extractRpi(pi))


    ### APPROXIMATE POLICY EVALUATION ###
    def approxPolicyEvaluation(self, pi, tolerance=0.01):
        '''
        Evaluate a policy using approximate policy evaluation. Like value iteration, approximate
        policy evaluation should continue until ||V_n - V_{n+1}||_inf <= tolerance.

        Return the value function, number of iterations required to get to exactness criterion, and final epsilon value.

        Parameter pi: Deterministic policy
        Precondition: array of |S| integers

        Parameter tolerance: threshold threshold on ||V^n-V^n+1||_inf
        Precondition: Float >= 0 (default: 0.01)
        '''
        V, epsilon, n_iters = np.zeros(self.nStates), np.inf, 0
        Ppi, Rpi = self.extractPpi(pi), self.extractRpi(pi)
        while True:
            newV = Rpi + self.discount * (Ppi @ V)
            epsilon = np.max(np.abs(V - newV))
            V = newV
            n_iters = n_iters + 1
            if epsilon <= tolerance: break
        return V, n_iters, epsilon

    def policyIterationStep(self, pi, exact):
        '''
        This function runs one step of policy evaluation, followed by one step of policy improvement. Return
        pi^{t+1} as a new numpy array. Do not modify pi^t.

        Parameter pi: Current policy pi^t
        Precondition: array of |S| integers

        Parameter exact: Indicate whether to use exact policy evaluation
        Precondition: boolean
        '''
        if exact: return self.extractMaxPifromV(self.exactPolicyEvaluation(pi))
        return self.extractMaxPifromV(self.approxPolicyEvaluation(pi)[0])

    def policyIteration(self, initial_pi, exact):
        '''

        Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a Q^pi(s,a)).


        This function should run policyIteration until convergence where convergence
        is defined as pi^{t+1}(s) == pi^t(s) for all states s.

        Return the final policy, value-function for that policy, and number of iterations
        required until convergence.

        Parameter initial_pi:  Initial policy
        Precondition: array of |S| entries

        Parameter exact: Indicate whether to use exact policy evaluation
        Precondition: boolean

        '''
        pi, iterId = initial_pi, 0
        while True:
            iterId = iterId + 1
            newPi = self.policyIterationStep(pi, exact)
            if np.array_equal(pi, newPi): break
            pi = newPi
        V = self.exactPolicyEvaluation(pi) if exact else self.approxPolicyEvaluation(pi)[0]
        return pi, V, iterId
