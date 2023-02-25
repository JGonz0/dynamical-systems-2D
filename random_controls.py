import numpy as np
import dynamical_propagation as dp
import env_tools as etools
import random

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import argparse


def main(args):
    ''' 
        States in order:
        - X position (Angle 1 for Acrobot)
        - Y position (Angle 2 for Acrobot)
        - Theta value
        - WL, left wheel velocity magnitude
        - WR, right wheel velocity magnitude
    '''

    # Frame bound
    STATE_X_LOWER = -10
    STATE_X_UPPER = 10
    STATE_Y_LOWER = -10
    STATE_Y_UPPER = 10

    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    iters = abs(args.iters)

    if iters > 10000:
        # Iteration threshold
        iters = 10000
        print('Only the first 10k iterations are shown')


    # History for plotting
    history_x = [state[0]]
    history_y = [state[1]]
    full_history_x = [state[0]]
    full_history_y = [state[1]]

    for iteration in range(iters):
        # Random application time
        delta_t = random.uniform(0.0, args.atime)

        match args.system:
            case "omni":    
                ux = random.uniform(-args.velx, args.velx)
                uy = random.uniform(-args.vely, args.vely)
                
                kyno = dp.HOLOCONTROLS(ux, uy, state[0], state[1])
                new_state, sub_x, sub_y = kyno.propagation(delta_t)

                x, y = new_state[0], new_state[1]

                # YOUR FAVORITE COLLISION CHECKER HERE

                x, y = etools.bound_position(
                    x, y, STATE_X_LOWER, STATE_X_UPPER, STATE_Y_LOWER, STATE_Y_UPPER)

                state[0], state[1] = x, y
                history_x.append(x)
                history_y.append(y)

                full_history_x += sub_x
                full_history_y += sub_y


    # Plot settings
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_xlim([STATE_X_LOWER - 5, STATE_X_UPPER + 5])
    ax.set_ylim([STATE_Y_LOWER - 5, STATE_Y_UPPER + 5])

    # Background
    img = plt.imread("./imgs/background.jpg")
    ax.imshow(img, extent=[STATE_X_LOWER - 5, STATE_X_UPPER + 5,
                           STATE_Y_LOWER - 5, STATE_Y_UPPER + 5])

    path_walking = './imgs/tile00'
    paths = [path_walking+str(x)+'.png' for x in range(6)]
    
    def getImage(path, zoom=1):
            return OffsetImage(plt.imread(path), zoom=zoom)
    
    def update_animation(iter):
        iter = int(iter)
        x, y = full_history_x[iter], full_history_y[iter]

        # Bounds for substate
        x, y = etools.bound_position(
            x, y, STATE_X_LOWER, STATE_X_UPPER, STATE_Y_LOWER, STATE_Y_UPPER)

        ab = AnnotationBbox(getImage(paths[iter % 6]), (x, y), frameon=False)
        point = ax.add_artist(ab)

        #point, = ax.plot(history_x[0], history_y[0], c='r', marker="o")
        #point.set_data([x], [y])

        return [point]
    

    anime = FuncAnimation(fig, update_animation, interval=50, blit=True, 
                          frames=np.linspace(0, iters, iters, endpoint=False))

    plt.show()



parser = argparse.ArgumentParser()
parser.add_argument('--system', type=str, default='omni', help='dynamical system selection (omni, first, second, etc)')
parser.add_argument('--velx', type=float, default=-1.0, help='Linear X velocity bound (Omni) [-a, a]')
parser.add_argument('--vely', type=float, default=-1.0, help='Linear Y velocity bound (Omni) [-a, a]')
parser.add_argument('--vel', type=float, default=-1.0, help='Linear velocity bound (DDR, Car-like) [-a, a]')
parser.add_argument('--accel', type=float, default=1.0, help='Aceleration bound (DDR 2nd) [-a, a]')
parser.add_argument('--ang', type=float, default=1.0, help='Angular velocity bound (Acrobot, DDR, Car-like) [-a, a]')
parser.add_argument('--torque', type=float, default=1.0, help='Torque bound (Acrobot) [-a, a]')
parser.add_argument('--atime', type=float, default=1.0, help='Application time range [0, a]')
parser.add_argument('--iters', type=int, default=1000, help='Amount of iterations')

args = parser.parse_args()
main(args)