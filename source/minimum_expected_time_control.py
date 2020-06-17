import numpy as np
from scipy.sparse import dok_matrix, lil_matrix
from solvers.value_iteration import ValueIteration
from solvers.policy_iteration import PolicyIteration
from solvers.linear_programming import LinearProgramming
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches 

def solve(Model):

    r_max = 9
    delta_r = 0.045

    alpha_max = np.pi
    alpha_min = - alpha_max
    delta_alpha = 2 * np.pi / 361

    sigma = 0.1

    stochastic_term = sigma ** 2 / delta_alpha ** 2 / 2

    r_list = np.linspace(0, r_max, int(r_max / delta_r) + 1)
    alpha_list = np.arange(alpha_min, alpha_max, delta_alpha)
    n_r = len(r_list)
    n_alpha = len(alpha_list)

    n_states = n_r * n_alpha

    action_list = [-1.0, 1.0]
    n_actions = len(action_list)

    rewards = np.zeros([n_states, n_actions])
    P = dok_matrix((n_states*n_actions, n_states), dtype=np.float32)
    for j, alpha in enumerate(alpha_list):
        b1 = -np.cos(alpha)
        for i, r in enumerate(r_list):
            s = i * n_alpha + j
            if i == 0:
                P[s*n_actions:(s+1)*n_actions, s] = 1
            else:
                for a, u in enumerate(action_list):
                    b2 = np.sin(alpha) / r - u
                    p1_p = max(0, b1) / delta_r  # \frac{b_1^+}{\Delta r} 
                    p1_m = max(0, -b1) /delta_r  # \frac{b_1^-}{\Delta r}
                    p2_p = max(0, b2) / delta_alpha + stochastic_term  # \frac{b_2^+}{\Delta r} + \frac{\sigma_r^2}{2\Delta\alpha^2}
                    p2_m = max(0, -b2) / delta_alpha + stochastic_term  # \frac{b_2^-}{\Delta r} + \frac{\sigma_r^2}{2\Delta\alpha^2}
                    delta_t = p1_m + p1_p + p2_m + p2_p  # \Delta t
                    if i == n_r-1:
                        P[s*n_actions+a, s] += p1_p / delta_t
                    else:
                        P[s*n_actions+a, s + n_alpha] += p1_p / delta_t
                    P[s*n_actions+a, s - n_alpha] += p1_m / delta_t
                    if j == n_alpha-1:
                        P[s*n_actions+a, s - j] += p2_p /delta_t
                    else:
                        P[s*n_actions+a, s + 1] += p2_p /delta_t
                    if j == 0:
                        P[s*n_actions+a, s + n_alpha - 1] += p2_m / delta_t
                    else:
                        P[s*n_actions+a, s - 1] += p2_m / delta_t
                    rewards[s, a] -= 1.0 / delta_t
    state_transition_probs = P.tocsr()

    model = Model(
        n_states,
        n_actions,
        rewards,
        state_transition_probs,
        1-np.finfo(np.float32).eps
    )
    model.fit(max_iteration=1000000, tolerance=1e-4, verbose=True, logging=False)

    np.save('values.npy', model.values)
    np.save('policy.npy', model.policy)
    
    return model.values.copy(), model.policy.copy()


def simulate(initial_pose, target, dt, max_time):

    r_max = 9
    r_min = 0.01
    delta_r = 0.045

    alpha_max = np.pi
    alpha_min = - alpha_max
    delta_alpha = 2 * np.pi / 361

    sigma = 0.1

    r_list = np.linspace(0, r_max, int(r_max / delta_r) + 1)
    alpha_list = np.arange(alpha_min, alpha_max, delta_alpha)
    n_alpha = len(alpha_list)

    policy = np.load('policy.npy')

    trajectory = []
    target = np.array(target, dtype=np.float32)
    pose = np.array(initial_pose, dtype=np.float32)
    for t in np.arange(0, max_time, dt):
        r = np.linalg.norm(pose[:2]-target)
        alpha = np.arctan2(
            target[1] - pose[1], target[0] - pose[0]
        ) - pose[2]
        if alpha >= np.pi:
            alpha -= 2 * np.pi
        elif alpha < np.pi:
            alpha += 2 * np.pi
        r_idx = np.argmax(np.abs(r_list - r))
        alpha_idx = np.argmax(np.abs(alpha_list - alpha))
        a = policy[r_idx * n_alpha + alpha_idx]

        pose[0] += np.cos(pose[2]) * dt
        pose[1] += np.sin(pose[2]) * dt
        if a==0:
            pose[2] -= dt
        else:
            pose[2] += dt
        pose[2] += np.random.normal(scale=sigma) * dt
        if pose[2] >= np.pi:
            pose[2] -= 2 * np.pi
        elif pose[2] < np.pi:
            pose[2] += 2 * np.pi
        trajectory.append([t]+list(pose))

        if r < r_min:
            break

    return np.array(trajectory)


def showValue():
    value = np.load('values.npy')
    print(value.shape)
    print(value)

    plt.plot(value)
    # plt.plot(value)

    plt.show()
    print("end")


def showTraject():
    traj = np.load('trajectory.npy')
    verts = traj[:,1:3]

    codes = [
    Path.MOVETO,
    Path.CURVE4,
    Path.CURVE4,
    Path.CURVE4,
    ]
    path = Path(verts)

    fig, ax = plt.subplots()
    patch = patches.PathPatch(path, facecolor='none', lw=2)
    ax.add_patch(patch)
    
    # plt.plot(value)

    plt.show()
    

if __name__=='__main__':

    # values, policy = solve(PolicyIteration)
    # values, policy = solve(ValueIteration)
    # values, policy = solve(LinearProgramming)
    # showValue()
    # trajectory = simulate(
    #     initial_pose = [0, 0, 0],
    #     target = [1, 1],
    #     dt = 0.001,
    #     max_time = 10
    # )
    # np.save('trajectory.npy', trajectory)
    showTraject()
