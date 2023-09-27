import numpy as np
from finite_difference_method import gradient, jacobian, hessian
from lqr import lqr

class LocalLinearizationController:
    def __init__(self, env):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset
                 the state to any state
        """
        self.env = env

    def c(self, s, a):
        """
        Cost function of the env.
        It sets the state of environment to `s` and then execute the action `a`, and
        then return the cost.
        Parameter:
            s (1D numpy array) with shape (4,)
            a (1D numpy array) with shape (1,)
        Returns:
            cost (double)
        """
        assert s.shape == (4,)
        assert a.shape == (1,)
        env = self.env
        env.reset(state=s)
        observation, cost, done, info = env.step(a)
        return cost

    def f(self, s, a):
        """
        State transition function of the environment.
        Return the next state by executing action `a` at the state `s`
        Parameter:
            s (1D numpy array) with shape (4,)
            a (1D numpy array) with shape (1,)
        Returns:
            next_observation (1D numpy array) with shape (4,)
        """
        assert s.shape == (4,)
        assert a.shape == (1,)
        env = self.env
        env.reset(state=s)
        next_observation, cost, done, info = env.step(a)
        return next_observation


    def compute_local_policy(self, s_star, a_star, T):
        """
        This function perform a first order taylar expansion function f and
        second order taylor expansion of cost function around (s_star, a_star). Then
        compute the optimal polices using lqr.
        outputs:
        Parameters:
            T (int) maximum number of steps
            s_star (numpy array) with shape (4,)
            a_star (numpy array) with shape (1,)
        return
            Ks(List of tuples (K_i,k_i)): A list [(K_0,k_0), (K_1, k_1),...,(K_T,k_T)] with length T
                                          Each K_i is 2D numpy array with shape (1,4) and k_i is 1D numpy
                                          array with shape (1,)
                                          such that the optimial policies at time are i is K_i * x_i + k_i
                                          where x_i is the state
        """
        n_a, = a_star.shape
        n_s, = s_star.shape
        
        # Helper functions
        def partial(f, fixed, loc=1):
            assert loc == 1 or loc == 2
            if loc == 2:
                def new_f(arg):
                    return f(arg, fixed)
            else:
                  def new_f(arg):
                    return f(fixed, arg)
            return new_f

        def extra(arr):
            return self.c(arr[:n_s], arr[-n_a:])

        # Easy ones
        A = jacobian(partial(self.f, a_star, 2), s_star)
        B = jacobian(partial(self.f, s_star, 1), a_star)

        q = gradient(partial(self.c, a_star, 2), s_star).reshape(n_s, 1)
        r = gradient(partial(self.c, s_star, 1), a_star).reshape(n_a, 1)

        # Search for H
        H = hessian(extra, np.concatenate((s_star, a_star)))

        # Eigen decomposition of H
        sigmas,vs = np.linalg.eig(H)
        pos_sigmas = sigmas[sigmas > 0]
        pos_vs = vs[:, sigmas > 0]
        D = np.diag(pos_sigmas)

        l = 0.725
        I = np.identity(n_a + n_s)

        H_approx = pos_vs @ D @ pos_vs.T + l * I

        # Finally our missing params:
        Q = H_approx[:n_s, :n_s]
        R = H_approx[-n_a:, -n_a:]
        M = H_approx[:n_s, -n_a:]

        b = self.c(s_star, a_star) + 1/2 * s_star.T @ Q/2@ s_star + 1/2* a_star.T @ R @ a_star + s_star.T @ M @ a_star - q.T @ s_star - r.T @ a_star
        m = (self.f(s_star, a_star) - A @ s_star - b * a_star).reshape(n_s,1)

        return lqr(A, B, m, Q, R, M, q, r, b, T)

class PIDController:
    """
    Parameters:
        P, I, D: Controller gains
    """

    def __init__(self, P, I, D):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset
                 the state to any state
        """
        self.P, self.I, self.D = P, I, D
        self.err_sum = 1.8
        self.err_prev = 1.8

    def get_action(self, err):
        self.err_sum += err
        a = self.P * err + self.I * self.err_sum + self.D * (err - self.err_prev)
        self.err_prev = err
        return a
