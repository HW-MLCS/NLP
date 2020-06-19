import numpy as np
from scipy.sparse import lil_matrix, identity, vstack, block_diag
from scipy.sparse.linalg import spsolve
from scipy.optimize import linprog



class LinearProgramming(object):
    """
    LinearProgramming(num_states, num_actions, rewards, state_transition_probs, discount)

    Finds an optimal value and a policy of a Markov decision process.
    Parameters
    ----------
    num_states : int
        Number of elements in the set of states.
    num_actions : int
        Number of elements in the set of actions.
    rewards : numpy.ndarray
        Reward values in given states and actions.
        $r(s, a)$.
    state_transition_probs : numpy.ndarray
        Probability in transion to a next state $s'$ given state $s$ and action $a$.
    """

    def __init__(self, num_states, num_actions, rewards, state_transition_probs, discount):

        self.rewards = rewards
        self.discount = discount
        self.state_transition_probs = state_transition_probs
        self.num_states = num_states
        self.num_actions = num_actions
        self.policy = np.empty((num_states), dtype=int)

        # Setup matrix form
        self.c = np.ones((num_states),dtype=np.float32) / num_states
        self.A = discount * state_transition_probs - block_diag(
            (np.ones((num_actions, 1), dtype=np.float32),) * num_states,
            format='csc'
        )
        self.b = - np.reshape(self.rewards, -1)
    

    def fit(self, max_iteration=1e3, tolerance=1e-3, verbose=False, logging=False, dual=False):

        if dual:
            res = linprog(
                self.b,
                A_eq = self.A.T,
                b_eq = -self.c,
                bounds = (0, None),
                method = 'interior-point',
                options = {
                    'maxiter':max_iteration,
                    'tol':tolerance,
                    'disp':verbose,
                    'sparse':False
                }
            )
            self.policy = np.argmax(np.reshape(res.x, (self.num_states, -1)), axis=1)
            _one_hot_policy =  np.eye(self.num_actions)[self.policy].astype(bool)
            _I_s = identity(self.num_states, dtype=np.float32, format='csc')
            _A = _I_s - self.discount * vstack(
                [self.state_transition_probs[s*self.num_actions+a, :] for s, a in enumerate(self.policy)],
                format='csc'
            )
            self.values = spsolve(_A, self.rewards[_one_hot_policy])
        else:
            res = linprog(
                self.c,
                A_ub = self.A,
                b_ub = self.b,
                bounds = None,
                method = 'interior-point',
                options = {
                    'maxiter':max_iteration,
                    'tol':tolerance,
                    'disp':verbose,
                    'sparse':False
                }
            )
            self.values = res.x
            for s in range(self.num_states):
                _q = self.rewards[s, :] + self.discount * self.state_transition_probs[s*self.num_actions:(s+1)*self.num_actions, :] * self.values
                # Evaluate the deterministic policy $\pi(s)$
                self.policy[s] = np.argmax(_q)
        
        return res



def test(Model):
    """
    Test code for debugging
    """

    from scipy.sparse import dok_matrix

    # Setup the number of states and actions
    n_states = 11
    n_actions = 3

    # Setup the reward $r(s,a)$
    rewards = np.zeros([n_states, n_actions])
    rewards[-1, -1] = 1

    # Setup the random state transition probability $P(s'|s,a)$
    p = np.float32(0.5)
    state_transition_probs = np.empty((n_states), dtype=np.object)
    for i in range(n_states):
        P = dok_matrix(
            (1.0 - p) * np.ones((n_actions, n_states), dtype=np.float32) / np.float32(n_states - 1)
        )
        for j in range(n_actions):
            P[j, np.random.randint(low=0, high=n_states)] = p
        state_transition_probs[i] = P
    state_transition_probs = vstack(state_transition_probs, format='csr')


    # Test model
    model = Model(
        n_states, n_actions, rewards, state_transition_probs,
        0.99 # - np.finfo(np.float32).eps
    )
    model.fit(max_iteration=100, tolerance=0.001, verbose=True, dual=False)



if __name__=='__main__':

    test(LinearProgramming)
    
