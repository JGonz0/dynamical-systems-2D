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
    full_history_x = [state[0]]
    full_history_y = [state[1]]
    history_control = []

    for iter in range(iters):
        # Random application time
        delta_t = random.uniform(0.0, args.atime)

        match args.system:
            case "omni":    
                u1 = random.uniform(-args.velx, args.velx)
                u2 = random.uniform(-args.vely, args.vely)
                
                kyno = dp.HOLOCONTROLS(u1, u2, state[0], state[1])
            
            case "first":
                u1 = random.uniform(-args.vel, args.vel)
                u2 = random.uniform(-args.ang, args.ang)

                kyno = dp.DDR_FIRST_PROPAGATION(
                    u1, u2, state[0], state[1], state[2])

            case "second":
                u1 = random.uniform(-args.accel, args.accel)
                u2 = random.uniform(-args.accel, args.accel)

                kyno = dp.DDR_SECOND_PROPAGATION(
                    u1, u2, state[3], state[4], state[0], state[1], state[2])
                
            case "carlike":
                u1 = random.uniform(-args.vel, args.vel)
                u2 = random.uniform(-args.ang, args.ang)

                kyno = dp.CARLIKE(u1, u2, state[0], state[1], state[2])

            # TODO ACROBOT
            
        # Propagation process
        new_state, sub_x, sub_y = kyno.propagation(delta_t)
        
        # YOUR FAVORITE COLLISION CHECKER HERE

        # Position constraint
        x, y = new_state[0], new_state[1]
        x, y = etools.bound_position(
            x, y, STATE_X_LOWER, STATE_X_UPPER, STATE_Y_LOWER, STATE_Y_UPPER)
        
        # New state -> Current state
        new_state[0], new_state[1] = x, y
        state = [new_state[_] for _ in range(len(new_state))]

        full_history_x += sub_x
        full_history_y += sub_y
        history_control.append([u1, u2, delta_t])
                  

    # Plot settings
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_xlim([STATE_X_LOWER - 5, STATE_X_UPPER + 5])
    ax.set_ylim([STATE_Y_LOWER - 5, STATE_Y_UPPER + 5])
    ax.set_title(args.system.upper())

    # Display info
    display = ax.text(STATE_X_LOWER - 3, STATE_Y_LOWER, '', size='small')

    # Background
    img = plt.imread("./imgs/background.png")
    ax.imshow(img, extent=[STATE_X_LOWER - 5, STATE_X_UPPER + 5,
                           STATE_Y_LOWER - 5, STATE_Y_UPPER + 5])

    # Walking sprite
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

        ab = AnnotationBbox(getImage(paths[iter % len(paths)]), (x, y), frameon=False)
        point = ax.add_artist(ab)

        #point, = ax.plot(history_x[0], history_y[0], c='r', marker="o")
        #point.set_data([x], [y])
        
        full_string = f'Iteration: {iter}\nControl 1: {history_control[iter][0]}\n'
        full_string += f'Control 2: {history_control[iter][1]}\nApplication Time: {history_control[iter][2]}\n'
        full_string += f'Position: ({x:.3f}, {y:.3f})'
        display.set_text(full_string)

        return [point] + [display,]
    

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