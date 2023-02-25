import numpy as np

# Collision checker for a disk robot (substates)
def collision_check_disk(obs, x_sub, y_sub, w = 5, h = 5, r = 1):

    # Obstacles width
    # Obstacles height
    # Disk radius

    for k in range(len(x_sub)):
        for i in range(len(obs[0])):
            # rx, ry - Lower left corner coordinates
            rx = obs[0][i]
            ry = obs[1][i]
            cx, cy = x_sub[k], y_sub[k]

            testX = cx
            testY = cy

            if cx < rx:
                testX = rx
            elif cx > rx+w:
                testX = rx+w
            if cy < ry:
                testY = ry
            elif cy > ry+h:
                testY = ry+h

            distX = cx-testX
            distY = cy-testY
            distance = np.sqrt((distX*distX) + (distY*distY))

            if distance < r:
                return True

    return False


# Collision checker for acrobot (substates)
def collision_check_acro(obs, th1_sub, th2_sub, w = 0.5, h = 0.5, steps = 30, link_length = 1):

    # Obstacles width
    # Obstacles height

    # Length
    l = link_length

    for k in range(len(th1_sub)):
        th1 = th1_sub[k]
        th2 = th2_sub[k]

        # FIRST LINK
        for i in range(len(obs[0])):
            rx = obs[0][i]
            ry = obs[1][i]

            # Assuming origin (0,0)
            x1 = 0
            x2 = l*np.cos(th1)
            y1 = 0
            y2 = l*np.sin(th1)

            dx = (x2 - x1)/steps
            dy = (y2 - y1)/steps

            for j in range(steps + 1):
                px = x1 + j*dx
                py = y1 + j*dy

                if px >= rx and px <= rx + w and py >= ry and py <= ry + h:
                    return True

            # Second LINK
            x1 = l*np.cos(th1)
            x2 = l*np.cos(th1) + l*np.cos(th1+th2)
            y1 = l*np.sin(th1)
            y2 = l*np.sin(th1) + l*np.sin(th1+th2)

            dx = (x2 - x1)/steps
            dy = (y2 - y1)/steps

            for j in range(steps + 1):
                px = x1 + j*dx
                py = y1 + j*dy

                if px >= rx and px <= rx + w and py >= ry and py <= ry + h:
                    return True

    return False


# Bound the robot's position into the frame
def bound_position(x, y, x_low, x_hi, y_low, y_hi):

    if x < x_low: x = x_low
    elif x > x_hi: x = x_hi
    if y < y_low: y = y_low
    elif y > y_hi: y = y_hi

    return x, y