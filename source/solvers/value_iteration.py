import numpy as np



class ValueIteration(object):
    """
    ValueIteration(num_states, num_actions, rewards, state_transition_probs, discount)

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

        self.num_states = num_states
        self.num_actions = num_actions
        self.rewards = rewards
        self.discount = discount
        self.state_transition_probs = state_transition_probs
        self.policy = np.empty((num_states), dtype=int)

        # Initialize the current value estimate with zeros
        self.values = np.max(rewards, axis=1)
    

    def update(self):
        
        # Compute the action values $Q(s,a)$
        values = np.empty_like(self.values, dtype=np.float32)
        for s in range(self.num_states):
            _q = self.rewards[s, :] + self.discount * self.state_transition_probs[s*self.num_actions:(s+1)*self.num_actions, :] * self.values
            # Evaluate the deterministic policy $\pi(s)$
            self.policy[s] = np.argmax(_q)
            # Compute the values $V(s)$
            values[s] = np.max(_q)

        # Compute the value difference $|\V_{k}-V_{k+1}|\$ for check the convergence
        diff = np.linalg.norm(
            (self.values[:] - values[:]) / self.num_states
        )

        # Update the current value estimate
        self.values = values

        return diff


    def fit(self, max_iteration=1e3, tolerance=1e-3, verbose=False, logging=False):
        
        if logging:
            history=[]

        # Value iteration loop
        for _iter in range(1, int(max_iteration+1)):

            # Update the value estimate
            diff = self.update()
            if logging:
                history.append(diff)
            if verbose:
                print('Iteration: {0}\tValue difference: {1}'.format(_iter, diff))

            # Check the convergence
            if diff < tolerance:
                if verbose:
                    print('Converged at iteration {0}.'.format(_iter))
                break

        if logging:
            return diff, history
        else:
            return diff


def test(Model):
    """
    Test code for debugging
    """

    from scipy.sparse import dok_matrix

    # Setup the number of states and actions
    n_states = 11
    n_actions = 3

    # Setup the reward $r(s,a)$
    rewards = np.zeros((n_states, n_actions), dtype=np.float32)
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
    model = Model(n_states, n_actions, rewards, state_transition_probs, 0.99 - np.finfo(np.float32).eps)
    model.fit(max_iteration=1000, tolerance=0.01, verbose=True)



if __name__=='__main__':

    test(ValueIteration)
    
