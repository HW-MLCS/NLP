import numpy as np
from scipy.sparse import dok_matrix, lil_matrix
from solvers.value_iteration import ValueIteration
from solvers.policy_iteration import PolicyIteration
from solvers.linear_programming import LinearProgramming
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches 
from numpy import linalg as LA
import math
import pandas

def solve(Model):

    r_max = 9
    delta_r = 0.15  #0.045

    alpha_max = np.pi
    alpha_min = - alpha_max
    delta_alpha = 2 * np.pi / 91 # 361

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
        0.99 #1-np.finfo(np.float32).eps
    )
    model.fit(max_iteration=1000000, tolerance=1e-6, verbose=True, logging=False, dual = False)

    np.save('values.npy', model.values)
    np.save('policy.npy', model.policy)

    plt.imshow(model.policy.reshape((-1,91)))
    # plt.imshow(model.values.reshape((-1,181)))
    plt.show()
    
    return model.values.copy(), model.policy.copy()


def simulate(initial_pose, target, dt, max_time):

    print(initial_pose[2]*180/math.pi)
    r_max = 9
    r_min = 0.01
    delta_r = 0.045 #0.045

    alpha_max = np.pi
    alpha_min = - alpha_max
    delta_alpha = 2 * np.pi / 361 #361

    sigma = 0.1

    r_list = np.linspace(0, r_max, int(r_max / delta_r) + 1)
    alpha_list = np.arange(alpha_min, alpha_max, delta_alpha)
    n_alpha = len(alpha_list)
    # print(n_alpha)
    
    
    policy, value_LP =  csvToNpy()
    
    # policy = np.load('policy_value.npy')
    # policy = np.load('policy_policy.npy')
    # print(policy)
    # print(policy.shape)

    trajectory = []
    target = np.array(target, dtype=np.float32)
    pose = np.array(initial_pose, dtype=np.float32)
    for t in np.arange(0, max_time, dt):
        r = np.linalg.norm(pose[:2]-target) # distance from current position to the target
        alpha = np.arctan2(
            target[1] - pose[1], target[0] - pose[0]
        ) - pose[2] # arctan(y,x) - heading
        if alpha >= np.pi:
            alpha -= 2 * np.pi
        elif alpha < -np.pi:
            alpha += 2 * np.pi
        r_idx = np.argmin(np.abs(r_list - r))
        alpha_idx = np.argmin(np.abs(alpha_list - alpha))
        # print(alpha_idx) # 0 or 360 only
        a = policy[r_idx * n_alpha + alpha_idx]
        # print(a) # there is only 1
        # print(r_idx * n_alpha + alpha_idx)
        pose[0] += np.cos(pose[2]) * dt
        pose[1] += np.sin(pose[2]) * dt
        if a==0:
            pose[2] -= dt
        else:
            pose[2] += dt
        # np.random.seed(seed=100)
        pose[2] += np.random.normal(scale=sigma) * dt
        # print(np.random.normal(scale=sigma) * dt)
        if pose[2] >= np.pi:
            pose[2] -= 2 * np.pi
        elif pose[2] < - np.pi:
            pose[2] += 2 * np.pi
        trajectory.append([t]+list(pose))

        if r < r_min:
            break
    plt.figure(figsize=(10,8))
    # plt.xlim()
    traj = np.array(trajectory)
    head_angle = traj[:,3]
    coord = traj[:,1:3]
    x = coord[:,0]
    y = coord[:,1]
    print(y.shape)
    plt.plot(x,y)
    plt.arrow(0,0, x[40]-x[0],y[40]-y[0],head_width=0.07, head_length=0.09,width=0.01, overhang=0.5)
    plt.arrow(x[-1],y[-1], 1-x[-50],1-y[-50],head_width=0.07, head_length=0.09,width=0.01, overhang=0.5)
    # plt.arrow(0,0, x[40]-x[0],y[40]-y[0],head_width=0.07/2, head_length=0.09/2,width=0.01/2, overhang=0.5)
    # plt.arrow(x[-50],y[-50], 1-x[-50],1-y[-50],head_width=0.07/2, head_length=0.09/2,width=0.01/2, overhang=0.5)
    plt.text(0.05,0, "Initial point",fontsize = 14)
    plt.text(0.5,1, "Target point",fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel("X (m)", fontsize = 20)
    plt.ylabel("Y (m)", fontsize = 20, rotation=0, labelpad=30)
    # plt.title("Robot trajectory", fontsize=20)


    plt.show()

    return np.array(trajectory)


def csvToNpy():
    policy_dataframe = pandas.read_csv("policy_LP.csv",header=None)
    policy_array = np.array(policy_dataframe)
    policy_array = np.ravel(policy_array.T)

    value_dataframe = pandas.read_csv("value_LP.csv",header=None)
    value_array = np.array(value_dataframe)
    value_array = np.ravel(value_array.T)


    return policy_array, value_array


def comparePolicy():
    
    policy_LP, value_LP =  csvToNpy()

    # policy_LP = np.load('policy_LP.npy')
    policy_value = np.load('policy_value.npy')
    policy_policy = np.load('policy_policy.npy')

    value_value = np.load('values_value.npy')
    value_policy = np.load('values_policy.npy')

    # value_LP = np.load('values.npy')
    # each of them has 72561 component which consists of 0 and 1

    plt.figure(figsize=(10,18))
    plt.subplot(2,1,1)
    # plt.imshow(policy_LP.reshape((-1,361)))
    plt.imshow(policy_value.reshape((-1,361)))
    # plt.imshow(policy_policy.reshape((-1,361)))
    plt.colorbar()
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel(r"$\alpha$", fontsize = 25)
    plt.ylabel("R", fontsize = 25, rotation=0, labelpad=10)
    # plt.title('Policy')
    
    plt.subplot(2,1,2)
    # plt.imshow(value_LP.reshape((-1,361)))
    plt.imshow(value_value.reshape((-1,361)))
    # plt.imshow(value_policy.reshape((-1,361)))
    plt.colorbar()
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel(r"$\alpha$", fontsize = 25)
    plt.ylabel("R", fontsize = 25, rotation=0, labelpad=10)
    # plt.title('Value')

    plt.show()

    diff_LP = np.linalg.norm(policy_policy - policy_LP)
    diff_val = np.linalg.norm(policy_policy - policy_value)
    print("diff_LP=",diff_LP)
    print("diff_val=",diff_val)
    

def showTraject():
    traj = np.load('trajectory_value.npy')
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
    trajectory = simulate(
        initial_pose = [0, 0, 5/4*math.pi], #[0,0,0]
        target = [1, 1], #[1,1]
        dt = 0.001, #0.001
        max_time = 10 #10
    )
    # np.save('trajectory_value.npy', trajectory)

    # comparePolicy()
    
    # showTraject()
