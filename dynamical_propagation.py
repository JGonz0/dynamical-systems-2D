import numpy as np

'''
Definition for the following dynamical systems (2D envs)
- Holonomic & Omnidireccional Robot
- Differential Drive Robot 1st Order
- Differential Drive Robot 2st Order
- Acrobot
- Car-like robot (Kinda same as DDR 1st)
'''


# Class used for Omnidirectional robot (2D)
class HOLOCONTROLS:
    def __init__(self, u1, u2, x, y):
        self.u1 = u1
        self.u2 = u2
        self.x = x
        self.y = y

    def propagation(self, t, steps = 15):

        # Direct propagation
        x_new = self.x + self.u1*t
        y_new = self.y + self.u2*t

        # SUBSTATES - for collision checker
        x_sub = []
        y_sub = []
        x_delta = self.u1*t/steps
        y_delta = self.u2*t/steps

        for i in range(1, steps+1):
            x_sub.append(self.x + i*x_delta)
            y_sub.append(self.y + i*y_delta)

        new_states = [x_new, y_new]
        return new_states, x_sub, y_sub


# Class used for Differential Drive Robot's propagation within state space (1st - Wheel velocity)
class DDR_FIRST_PROPAGATION:
    def __init__(self, u1, u2, x, y, th, b = 0.71):
        self.u1 = u1
        self.u2 = u2
        self.x = x
        self.y = y

        # Bound to [-2pi, 2pi]
        if (th > 2*np.pi or th < 0):
            th %= 2*np.pi

        self.th = th
        self.b = b

    # Checks if the system's velocity is still bounded
    def theta_function(self, t):
        return self.th + self.u2*t

    # Indicator True -> X, Otherwise -> Y
    def func(self, indicator, t):
        if (indicator):
            return self.u1 * np.cos(self.theta_function(t))
        return self.u1 * np.sin(self.theta_function(t))

    def simpson_1_3(self, Li, Ls, P = 15, indicator = True):
        h = (Ls-Li)/P
        integral = 0
        substates = []

        for k in range(0, P-1, 2):
            integral += self.func(indicator, Li+(k*h)) + 4*self.func(indicator, Li+(k+1)*h) +\
                self.func(indicator, Li+(k+2)*h)

            if indicator:
                substates.append(self.x + (h/3)*integral)
            else:
                substates.append(self.y + (h/3)*integral)

        return (h/3.0)*integral, substates

    # Propagation process, retrieves new states
    def propagation(self, t):
        th_new = self.th + self.u2*t

        # New states + Substates
        x_new, x_sub = self.simpson_1_3(0, t, 15, True)
        y_new, y_sub = self.simpson_1_3(0, t, 15, False)
        x_new += self.x
        y_new += self.y

        new_states = [x_new, y_new, th_new]
        return new_states, x_sub, y_sub


# Class used for Differential Drive Robot's propagation within state space (2nd - Wheel acceleration)
class DDR_SECOND_PROPAGATION:
    def __init__(self, al, ar, vl, vr, x, y, th, b = 0.71, V_max = 3.0):
        self.al = al
        self.ar = ar
        self.vl = vl
        self.vr = vr
        self.x = x
        self.y = y
        self.th = th
        self.b = 0.71
        self.V_max = V_max
        self.bounded = False

    # Theta function used to integrate w/simpson method
    def theta(self, t):
        return (0.5/self.b)*(np.sign(self.vr+self.ar*t)*min(abs(self.vr+self.ar*t), self.V_max) -
                             np.sign(self.vl+self.al*t)*min(abs(self.vl+self.al*t), self.V_max))

    def simpson_1_3_theta(self, Li, Ls, P = 15):
        h = (Ls-Li)/P
        integral = 0

        for k in range(0, P-1, 2):
            integral += self.theta(Li+(k*h)) + 4 * \
                self.theta(Li+(k+1)*h) + self.theta(Li+(k+2)*h)

        return (h/3)*integral

    # Checks if the system's velocity is still bounded
    def theta_function(self, t):
        if self.bounded:
            return self.th + (0.5/self.b)*(0.5*(self.ar-self.al)*t*t+(self.vr-self.vl)*t)

        return self.th + self.simpson_1_3_theta(0, t, 10)

    # Similar to theta function, for (x,y) states
    # Indicator True -> X, Otherwise -> Y
    def func(self, indicator, t):
        Vl = np.sign(self.vl+self.al*t)*min(abs(self.vl+self.al*t), self.V_max)
        Vr = np.sign(self.vr+self.ar*t)*min(abs(self.vr+self.ar*t), self.V_max)

        if (indicator):
            return 0.5*(Vl+Vr)*np.cos(self.theta_function(t))
        return 0.5*(Vl+Vr)*np.sin(self.theta_function(t))

    def simpson_1_3(self, Li, Ls, P = 15, indicator = True):
        h = (Ls-Li)/P
        integral = 0
        substates = []

        for k in range(0, P-1, 2):
            integral += self.func(indicator, Li+(k*h)) + 4*self.func(indicator, Li+(k+1)*h) +\
                self.func(indicator, Li+(k+2)*h)

            if indicator:
                substates.append(self.x + (h/3)*integral)
            else:
                substates.append(self.y + (h/3)*integral)

        return (h/3)*integral, substates

    # Propagation process, retrieves new states
    def propagation(self, t):
        vl_new = np.sign(self.vl+self.al*t) * \
            min(abs(self.vl+self.al*t), self.V_max)
        vr_new = np.sign(self.vr+self.ar*t) * \
            min(abs(self.vr+self.ar*t), self.V_max)

        self.bounded = True
        if abs(vl_new) == self.V_max or abs(vr_new) == self.V_max:
            self.bounded = False

        th_new = self.theta_function(t)

        # Bound to [-2pi, 2pi]
        if (th_new > 2*np.pi or th_new < 0):
            th_new %= 2*np.pi

        # New states + Substates
        x_new, x_sub = self.simpson_1_3(0, t, 15, True)
        y_new, y_sub = self.simpson_1_3(0, t, 15, False)
        x_new += self.x
        y_new += self.y

        new_states = [x_new, y_new, th_new, vl_new, vr_new]
        return new_states, x_sub, y_sub


# Class used for a two-link acrobot system (Boone model)
class ACROBOT:
    def __init__(self, tau, th1, th2, th1d, th2d, th1dmax = 6, th2dmax = 6):
        self.tau = tau
        self.th1 = th1
        self.th2 = th2
        self.th1d = th1d
        self.th2d = th2d

        # Max angular velocity
        self.th1dmax = th1dmax
        self.th2dmax = th2dmax

    def propagation(self, t, steps=20):
        # Control (torque)
        tau = self.tau

        # Parameters (states)
        th1 = self.th1
        th2 = self.th2
        th1d = self.th1d
        th2d = self.th2d

        # Parameters (system)
        l1 = 1.0
        l2 = 1.0
        lc1 = 0.5
        lc2 = 0.5
        m = 1.0
        I1 = 1.0
        I2 = 1.0
        g = 9.81

        # Inertial acceleration
        d11 = m*lc1*lc1 + m*(l1*l1 + lc2*lc2 + 2*l1*l2*np.cos(th2)) + I1 + I2
        d22 = m*lc2*lc2 + I2
        d12 = m*(lc2*lc2 + l1*lc2*np.cos(th2)) + I2
        d21 = d12

        # Coriolis and centrifugal contribution
        c1 = -m*l1*lc2*th2d*th2d*np.sin(th2) - 2*m*l1*lc2*th1d*th2d*np.sin(th2)
        c2 = m*l1*lc2*th1d*th1d*np.sin(th2)

        # Gravitational loading
        phi1 = (m*lc1 + m*l1)*g*np.cos(th1) + m*lc2*g*np.cos(th1 + th2)
        phi2 = m*lc2*g*np.cos(th1 + th2)

        # Double derivative
        th2dd = (d11*(tau - c2 - phi2) + d12*(c1 + phi1))/(d11*d22 - d12*d21)
        th1dd = (d12*th2dd + c1 + phi1)/(-d11)

        # Direct integration
        th1d_new = self.th1d + th1dd*t
        th2d_new = self.th2d + th2dd*t

        # BOUNDS - First Link
        if abs(th1d_new) > self.th1dmax:
            if th1d_new > 0:
                th1d_new = self.th1dmax
            else:
                th1d_new = -self.th1dmax

        # BOUNDS - Second Link
        if abs(th2d_new) > self.th2dmax:
            if th2d_new > 0:
                th2d_new = self.th2dmax
            else:
                th2d_new = -self.th2dmax

        # Second integrator
        th1_new = self.th1 + th1d_new*t
        th2_new = self.th2 + th2d_new*t

        # Angles bounded to [-2pi, 2pi]
        if (th1_new > 2*np.pi or th1_new < 0):
            th1_new %= 2*np.pi

        if (th2_new > 2*np.pi or th2_new < 0):
            th2_new %= 2*np.pi

        # SUBSTATES - For collision check
        th1_sub = []
        th2_sub = []
        th1_delta = th1d_new*t/steps
        th2_delta = th2d_new*t/steps

        for i in range(1, steps+1):
            th1_sub.append(self.th1 + i*th1_delta)
            th2_sub.append(self.th2 + i*th2_delta)

        new_states = [th1_new, th2_new, th1d_new, th2d_new]
        return new_states, th1_sub, th2_sub


# Class used for car-like robot
class CARLIKE:
    def __init__(self, u1, u2, x, y, th):
        self.u1 = u1
        self.u2 = u2
        self.x = x
        self.y = y
        self.th = th

    def propagation(self, t, steps=20):
        # Angles
        integration_step = t/steps
        angles = self.th + np.arange(1, steps+1)*(integration_step*self.u2)

        # X and Y projections
        coss = np.cos(angles)*(integration_step*self.u1)
        sins = np.sin(angles)*(integration_step*self.u1)
        state = [0, 0, 0]

        # Integrate (euler)
        state[0] = self.x + np.sum(coss)
        state[1] = self.y + np.sum(sins)
        state[2] = angles[-1]

        x_sub, y_sub = [], []

        for i in range(steps):
            x_sub.append(np.sum(coss[:i]) + self.x)
            y_sub.append(np.sum(sins[:i]) + self.y)

        return state, x_sub, y_sub