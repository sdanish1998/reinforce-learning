import numpy as np


def compute_Q_params_(A, B, m, Q, R, M, q, r, b, P, y, p):
    # Using equations in section 3.2.1
    C = Q + A.T @ P @ A
    D = R + B.T @ P @ B
    E = M + 2 * (A.T @ P @ B)
    f = (q.T + 2* (m.T @ P @ A) + y.T @ A).T
    g = (r.T + 2* (m.T @ P @ B) + y.T @ B).T
    h = b + m.T @ P @ m + y.T@ m + p
    return C, D, E, f, g, h


def compute_Q_params(A, B, m, Q, R, M, q, r, b, P, y, p):
    """
    Compute the Q function parameters for time step t.
    Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)
        Parameters:
        A (2d numpy array): A numpy array with shape (n_s, n_s)
        B (2d numpy array): A numpy array with shape (n_s, n_a)
        m (2d numpy array): A numpy array with shape (n_s, 1)
        Q (2d numpy array): A numpy array with shape (n_s, n_s). Q is PD
        R (2d numpy array): A numpy array with shape (n_a, n_a). R is PD.
        M (2d numpy array): A numpy array with shape (n_s, n_a)
        q (2d numpy array): A numpy array with shape (n_s, 1)
        r (2d numpy array): A numpy array with shape (n_a, 1)
        b (1d numpy array): A numpy array with shape (1,)
        P (2d numpy array): A numpy array with shape (n_s, n_s). This is the quadratic term of the
            value function equation from time step t+1. P is PSD.
        y (2d numpy array): A numpy array with shape (n_s, 1).  This is the linear term
            of the value function equation from time step t+1
        p (1d numpy array): A numpy array with shape (1,).  This is the constant term of the
            value function equation from time step t+1
    Returns:
        C (2d numpy array): A numpy array with shape (n_s, n_s)
        D (2d numpy array): A numpy array with shape (n_s, n_a)
        E (2d numpy array): A numpy array with shape (n_s, n_a)
        f (2d numpy array): A numpy array with shape (n_s,1)
        g (2d numpy array): A numpy array with shape (n_a,1)
        h (1d numpy array): A numpy array with shape (1,)

        where the following equation should hold
        Q_t^*(s) = s^T C s + a^T D s + s^T E a + f^T s  + g^T a + h
    """
    n_s, n_a = B.shape
    assert A.shape == (n_s, n_s)
    assert B.shape == (n_s, n_a)
    assert m.shape == (n_s, 1)
    assert Q.shape == (n_s, n_s)
    assert R.shape == (n_a, n_a)
    assert M.shape == (n_s, n_a)
    assert q.shape == (n_s, 1)
    assert r.shape == (n_a, 1)
    assert b.shape == (1, )
    assert P.shape == (n_s, n_s)
    assert y.shape == (n_s, 1)
    assert p.shape == (1, )
    return compute_Q_params_(A, B, m, Q, R, M, q, r, b, P, y, p)


def compute_policy_(D, E, g):
    # Using equations in section 3.2.2
    K_t = -1/2 * (np.linalg.inv(D) @ E.T)
    k_t = -1/2 * (np.linalg.inv(D) @ g)
    return K_t, k_t


def compute_policy(A, B, m, C, D, E, f, g, h):
    """
    Compute the optimal policy at the current time step t
    Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)
    Let Q_t^*(s) = s^T C s + a^T D a + s^T E a + f^T s  + g^T a  + h
    Parameters:
        A (2d numpy array): A numpy array with shape (n_s, n_s)
        B (2d numpy array): A numpy array with shape (n_s, n_a)
        m (2d numpy array): A numpy array with shape (n_s, 1)
        C (2d numpy array): A numpy array with shape (n_s, n_s). C is PD.
        D (2d numpy array): A numpy array with shape (n_a, n_a). D is PD.
        E (2d numpy array): A numpy array with shape (n_s, n_a)
        f (2d numpy array): A numpy array with shape (n_s, 1)
        g (2d numpy array): A numpy array with shape (n_a, 1)
        h (1d numpy array): A numpy array with shape (1, )
    Returns:
        K_t (2d numpy array): A numpy array with shape (n_a, n_s)
        k_t (2d numpy array): A numpy array with shape (n_a, 1)

        where the following holds
        \pi*_t(s) = K_t s + k_t
    """
    n_s, n_a = B.shape
    assert A.shape == (n_s, n_s)
    assert B.shape == (n_s, n_a)
    assert m.shape == (n_s, 1)
    assert C.shape == (n_s, n_s)
    assert D.shape == (n_a, n_a)
    assert E.shape == (n_s, n_a)
    assert f.shape == (n_s, 1)
    assert g.shape == (n_a, 1)
    assert h.shape == (1, )
    return compute_policy_(D, E, g)


def compute_V_params_(C, D, E, f, g, h, K, k):
    # Using equations in section 3.2.3
    P_h = C + K.T @ D @ K + E @ K
    y_h = (f.T + 2*(k.T @ D @ K) + k.T @ E.T + g.T @ K).T
    p_h = k.T @ D @ k + g.T @ k + h
    return P_h, y_h, p_h


def compute_V_params(A, B, m, C, D, E, f, g, h, K, k):
    """
    Compute the V function parameters for the next time step
    Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)
    Let V_t^*(s) = s^TP_ts + y_t^Ts + p_t
    Parameters:
        A (2d numpy array): A numpy array with shape (n_s, n_s)
        B (2d numpy array): A numpy array with shape (n_s, n_a)
        m (2d numpy array): A numpy array with shape (n_s, 1)
        C (2d numpy array): A numpy array with shape (n_s, n_s). C is PD.
        D (2d numpy array): A numpy array with shape (n_a, n_a). D is PD.
        E (2d numpy array): A numpy array with shape (n_s, n_a)
        f (2d numpy array): A numpy array with shape (n_s, 1)
        g (2d numpy array): A numpy array with shape (n_a, 1)
        h (1d numpy array): A numpy array with shape (1, )
        K (2d numpy array): A numpy array with shape (n_a, n_s)
        k (2d numpy array): A numpy array with shape (n_a, 1)

    Returns:
        P_h (2d numpy array): A numpy array with shape (n_s, n_s)
        y_h (2d numpy array): A numpy array with shape (n_s, 1)
        p_h (1d numpy array): A numpy array with shape (1,)
    """
    n_s, n_a = B.shape
    assert A.shape == (n_s, n_s)
    assert B.shape == (n_s, n_a)
    assert m.shape == (n_s, 1)
    assert C.shape == (n_s, n_s)
    assert D.shape == (n_a, n_a)
    assert E.shape == (n_s, n_a)
    assert f.shape == (n_s, 1)
    assert g.shape == (n_a, 1)
    assert h.shape == (1, )
    assert K.shape == (n_a, n_s)
    assert k.shape == (n_a, 1)
    return compute_V_params_(C, D, E, f, g, h, K, k)


def lqr(A, B, m, Q, R, M, q, r, b, T):
    """
    Compute optimal policies by solving
    argmin_{\pi_0,...\pi_{T-1}} \sum_{t=0}^{T-1} s_t^T Q s_t + a_t^T R a_t + s_t^T M a_t + q^T s_t + r^T a_t
    subject to s_{t+1} = A s_t + B a_t + m, a_t = \pi_t(s_t)

    Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)
    Let optimal \pi*_t(s) = K_t s + k_t

    Parameters:
    A (2d numpy array): A numpy array with shape (n_s, n_s)
    B (2d numpy array): A numpy array with shape (n_s, n_a)
    m (2d numpy array): A numpy array with shape (n_s, 1)
    Q (2d numpy array): A numpy array with shape (n_s, n_s). Q is PD.
    R (2d numpy array): A numpy array with shape (n_a, n_a). R is PD.
    M (2d numpy array): A numpy array with shape (n_s, n_a)
    q (2d numpy array): A numpy array with shape (n_s, 1)
    r (2d numpy array): A numpy array with shape (n_a, 1)
    b (1d numpy array): A numpy array with shape (1,)
    T (int): The number of total steps in finite horizon settings

    Returns:
        ret (list): A list, [(K_0, k_0), (K_1, k_1), ..., (K_{T-1}, k_{T-1})]
        and the shape of K_t is (n_a, n_s), the shape of k_t is (n_a,)
    """
    n_s, n_a = B.shape
    assert A.shape == (n_s, n_s)
    assert B.shape == (n_s, n_a)
    assert m.shape == (n_s, 1)
    assert Q.shape == (n_s, n_s)
    assert R.shape == (n_a, n_a)
    assert M.shape == (n_s, n_a)
    assert q.shape == (n_s, 1)
    assert r.shape == (n_a, 1)
    assert b.shape == (1, )
    pies = []

    # T-1 policy
    K_last, k_last = compute_policy_(R, M, r)
    Kt, kt = K_last[0], k_last[0]
    pies.append((Kt, kt))

    # T-1 value function
    Pt, yt, _ = compute_V_params_(Q, R, M, q, r, np.zeros((1,)), K_last, k_last)
    pt = b - 1/4 * r.T @ np.linalg.inv(R) @ r

    for i in range(T-1):
        C, D, E, f, g, h =  compute_Q_params(A, B, m, Q, R, M, q, r, b, Pt, yt, pt[0])
        Kt, kt =  compute_policy(A, B, m, C, D, E, f, g, h[0])
        pies.insert(0, (Kt[0], kt[0]))
        Pt, yt, pt = compute_V_params(A, B, m, C, D, E, f, g, h[0], Kt, kt)

    return pies
