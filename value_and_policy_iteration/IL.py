## Code for Imitation Learning
from sklearn.linear_model import LogisticRegression
import numpy as np

def collectData(expert_policy, dp, sample_size=500):
    """
    Uses expert policy pi to collect a dataset [(s_i, a_i)] of size sample_size.
    Assumes that all trajectories start from state 0.

    Input:
        expert_policy: expert policy (array of |S| integers)
        dp: an instance of the DynamicProgramming class
        sample_size: the number of samples to collect

    Output:
        states: An array of the form [s_1, s_2, ..., s_n], where s_i is a one-hot encoding of the state at timestep i,
                    and n=sample_size
        actions: An array of the form [a_1, a_2, ..., a_n], where a_i is the action taken at timestep i,
                    and n=sample_size
    """
    #initialize dataset
    states, actions = [], []
    #initialize state
    s = 0
    #compute pi and Ppi using policy iteration
    pi, _, _ = dp.policyIteration(np.random.randint(dp.nActions, size=dp.nStates), True)
    Pip = dp.extractPpi(pi)

    # Main data collection loop
    for i in range(sample_size):
        # Choose the next action and state
        a = pi[s]
        s1 = np.random.choice(dp.nStates, 1, p=Pip[s])[0]
        # Update the state and action data
        states.append(s)
        actions.append(a)
        s = s1
        # Reset state to 0 if we reach the terminal state
        if s == dp.nStates-1:
            s = 0
    # Return actions and one-hot encoding of states
    return np.eye(dp.nStates)[states], np.array(actions)

def trainModel(states, actions):
    """
    Uses the dataset to train a policy pi using behavior cloning, using
    scikit-learn's logistic regression module as our policy class.

    Input:
        states: An array of the form [s_1, s_2, ..., s_n], where s_i is a one-hot encoding of the state at timestep i
                Note: n>=1
        actions: An array of the form [a_1, a_2, ..., a_n], where a_i is the action taken at timestep i
                Note: n>=1
    Output:
        pi: the learned policy (array of |S| integers)
    """
    # Learn policy using logistic regression
    clf = LogisticRegression(random_state=0).fit(states, actions)
    # Convert policy to vector form
    return clf.predict(np.eye(len(states[0])))
