# What functionality do we need the simulator to have?
# Need to be able to draw from it at one point in time
# Need to animate

import numpy as np


class pendulum:
    
    def __init__(self, theta, t, noise):

        self.theta = theta
        self.t = t
        self.noise = noise
        
        if not self.noise:
            # If it is not defined, then no noise
            self.noise = np.zeros(np.shape(theta))
        
    #def animate(self):
        
    
    def simulate_x(self):
        ts = np.repeat(self.t[:, np.newaxis], self.theta.shape[0], axis=1)


        if theta.ndim == 1:
            theta = theta[np.newaxis, :]

        # time to solve for position and velocity

        # nested for loop, there's probably a better way to do this
        # output needs to be (n,len(t))
        x = np.zeros((theta.shape[0],len(t)))
        
        for n in range(theta.shape[0]):

            # Draw parameter (theta) values from normal distributions
            # To produce noise in the thetas you are using to produce the position
            # and momentum of the pendulum at each moment in time
            # Another way to do this would be to just draw once and use that same noisy theta 
            # value for all moments in time, but this would be very similar to just drawing
            # from the prior, which we're already doing.

            gs = np.random.normal(loc=theta[n][0], scale=noise[0], size=np.shape(t))
            Ls = np.random.normal(loc=theta[n][1], scale=noise[1], size=np.shape(t))
            theta_os =  np.random.normal(loc=theta[n][2], scale=noise[2], size=np.shape(t))

            theta_t = np.array([theta_os[i] * math.cos(np.sqrt(gs[i] / Ls[i]) * t[i]) for i, _ in enumerate(t)])

            x[n,:] = np.array([Ls[i] * math.sin(theta_t[i]) for i, _ in enumerate(t)])
            
        return x
    
    def simulate_q_p(self):
        ts = np.repeat(self.t[:, np.newaxis], self.theta.shape[0], axis=1)


        if theta.ndim == 1:
            theta = theta[np.newaxis, :]

        # time to solve for position and velocity

        # nested for loop, there's probably a better way to do this
        # output needs to be (n,len(t))
        x = np.zeros((theta.shape[0],len(t)))
        y = np.zeros((theta.shape[0],len(t)))

        # TO DO: I'm not strictly solving for momentum, just velocities:
        dx_dt = np.zeros((theta.shape[0],len(t)))
        dy_dt = np.zeros((theta.shape[0],len(t)))
        for n in range(theta.shape[0]):

            # Draw parameter (theta) values from normal distributions
            # To produce noise in the thetas you are using to produce the position
            # and momentum of the pendulum at each moment in time
            # Another way to do this would be to just draw once and use that same noisy theta 
            # value for all moments in time, but this would be very similar to just drawing
            # from the prior, which we're already doing.

            gs = np.random.normal(loc=theta[n][0], scale=noise[0], size=np.shape(t))
            Ls = np.random.normal(loc=theta[n][1], scale=noise[1], size=np.shape(t))
            theta_os =  np.random.normal(loc=theta[n][2], scale=noise[2], size=np.shape(t))

            theta_t = np.array([theta_os[i] * math.cos(np.sqrt(gs[i] / Ls[i]) * t[i]) for i, _ in enumerate(t)])

            x[n,:] = np.array([Ls[i] * math.sin(theta_t[i]) for i, _ in enumerate(t)])
            y[n,:] = np.array([-Ls[i] * math.cos(theta_t[i]) for i, _ in enumerate(t)])

            # Okay and what about taking the time derivative?
            dx_dt[n,:] = np.array([-Ls[i] * theta_os[i] * np.sqrt(gs[i] / Ls[i]) * math.cos(theta_t[i]) * math.sin( np.sqrt(gs[i] / Ls[i]) * t[i]) for i, _ in enumerate(t)])
            dy_dt[n,:] = np.array([-Ls[i] * theta_os[i] * np.sqrt(gs[i] / Ls[i]) * math.sin(theta_t[i]) * math.sin( np.sqrt(gs[i] / Ls[i]) * t[i]) for i, _ in enumerate(t)])





        return x, y, dx_dt, dy_dt
        
    
    
