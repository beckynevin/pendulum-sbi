# running the 3 parameter modeling but with simple ABC
# also adding in noise

# for pretty plotting:
import corner

import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib

# sbi from Mackelab
import sbi.utils as utils
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi.analysis import pairplot

# setting the sigmas on the noise distribution
# you will eventually draw the parameter values from a normal
# distribution with the width given by these values
noiz = [0.1,0.0,0.0] # here, there's only noise on g


# This is the simulator
# Given thetas, it outputs the x and y position (cartesian)
# of the pendulum over a range of times
# p and q are position and momentum

# Option is to input the width of the noise normal distributions
# around each parameter.
# default is a bit of noise for each of the three parameters

def create_t_p_q_noise(theta, t = np.linspace(0, 10, 200), noise=[0.1, 0.1, 0.1]):
    """
    Return an t, x, y array for plotting based on params
    Also introduces noise to parameter draws    
    """

    

    # Decide on time, here I'm sampling a few oscillations, but this could be
    # something you import or change
    
    
    ts = np.repeat(t[:, np.newaxis], theta.shape[0], axis=1)


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
    
    

    return t, x, y, dx_dt, dy_dt


def forward_one_t(theta, t = 5, noise=[0.1, 0.1, 0.1]):
    """
    Return an t, x, y array for plotting based on params
    Also introduces noise to parameter draws    
    """


    #if theta.ndim == 1:
    #    theta = theta[np.newaxis, :]

    
    # Do a loop over n still
    g = np.random.normal(loc=theta[0], scale=noise[0])#, size = np.shape(theta)[0])#, size=np.shape(t))
    L = np.random.normal(loc=theta[1], scale=noise[1])#, size=np.shape(t))
    theta_o =  np.random.normal(loc=theta[2], scale=noise[2])#, size=np.shape(t))

    theta_t = theta_o * math.cos(np.sqrt(g / L) * t)
    
    x = L * math.sin(theta_t) 

    return x



thetas = np.random.normal(size=(100000,3))
thetas[:,0] *= 0.2
thetas[:,0] += 10

thetas[:,1] *= 0.5
thetas[:,1] += 10

thetas[:,2] *= np.pi/12
thetas[:,2] += np.pi/4

Xs1 = np.array([forward_one_t(tt) for tt in thetas])

plt.clf()
plt.scatter(thetas[:,0], Xs1)
plt.xlabel('g')
plt.ylabel('x pos')
plt.show()







'''
# 3. SBI Example 2: ABC in higher dimensions
Lets try a slightly more complicated problem and allow redshift to vary in our forward model. That way we can infer posteriors for both $\log M_*$ and redshift: 
$$p(g, L, theta_0 | X_{\rm obs}=19)$$

For our priors on  $\log M_*$ and $z$, lets use
$$p(\log M_*) = \mathcal{N}(10.5, 0.2)$$
$$p(z) = \mathcal{N}(0.2, 0.02)$$
'''
print(np.shape(thetas))
print(np.shape(Xs1))
print(np.shape(Xs1[:,]))
print(np.shape(Xs1[:,None]))

print(np.concatenate([thetas, Xs1[:,None]], axis=1))
print(np.shape(np.concatenate([thetas, Xs1[:,None]], axis=1)))

Xobs = 2
thresh = 0.1

fig = corner.corner(np.concatenate([thetas, Xs1[:,None]], axis=1), 
                 labels=[r'$g$', '$L$', '$\Theta_0$','x'],
                 label_kwargs={'fontsize': 25},
                 range=[(8,12), (8, 12), (0, np.pi/2), (-1,7)],
                 color='C0')

axes = np.array(fig.axes).reshape((4,4))
ax = axes[3,0]
ax.plot([8,12], [Xobs, Xobs], c='k', ls='--')
#ax.text(9.6, 18.9, r'$X_{\rm obs}$', fontsize=20)
ax.fill_between([8,12], [Xobs - thresh, Xobs - thresh], [Xobs + thresh, Xobs + thresh], 
    color='C1', alpha=0.5)

ax = axes[3,1]
ax.plot([8,12], [Xobs, Xobs], c='k', ls='--')
ax.fill_between([8,12], [Xobs - thresh, Xobs - thresh], [Xobs + thresh, Xobs + thresh], color='C1', alpha=0.5)

ax = axes[3,2]
ax.plot([0,2], [Xobs, Xobs], c='k', ls='--')
ax.fill_between([0,2], [Xobs - thresh, Xobs - thresh], [Xobs + thresh, Xobs + thresh], color='C1', alpha=0.5)


ax = axes[0, 2]
ax.fill_between([], [], [], color='C0', label=r"$(\theta', X')\sim p(\theta, X)$")
ax.fill_between([], [], [], color='C1', alpha=0.5, label=r'$|X-X_{\rm obs}| < 0.1$')
ax.legend(loc='lower left', fontsize=20)
plt.show()


abc_thresh = (np.abs(Xs1 - Xobs) < 0.1)
print('%i samples in ABC threshold' % np.sum(abc_thresh))
print('this is the ABC threshold result, everything within the thresh')

plt.clf()
fig = corner.corner(thetas[abc_thresh], 
                    labels=[r'$g$', '$L$', '$\Theta_0$'], 
                    label_kwargs={'fontsize': 25},
                    color='C0')
plt.show()

