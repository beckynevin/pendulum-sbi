# In Caldeira and Nord:
# Statistical and Aleatoric is noise in data, is the spread on their measurements of the period, T
# or the standard deviation
# Basically, drawing from a normal distribution for T 
# Aleatoric but systematic is spread in measurement of L, but not stdev
# They assess epistemic by looking at points far from the training set

# They are progating the uncertainty through to the measurement of g
# How does this change if we're trying to estimate g, L, and theta from the positions and momenta
# at each moment in time
# Can we propagate uncertainty through to each of these variables and then use that to compare to the
# posterior distributions we get out the back end?

# SO IN OUR EXPERIMENT:
# we are solving for the position and momentum as a function of time,
# so we can introduce noise into these measurements via
# the parameter draws
# Meaning that we are drawing from a distribution of values at each time?
# Try this with one variable

# Trying this with three params now
# also adding in noise

# but write so that it includes momenta as well

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

# If noise is on or off for each parameter
# g, L, theta_not
noiz = [0.1, 0, 0]

# num_dim is the number of parameters were controlling (g, L, )
num_dim = 3
prior = utils.BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))





# This is the simulator
# Option is to input the width of the noise normal distributions
# around each parameter.
def create_t_p_q_noise(theta, seed=None, noise=[0.1,0.1,0.1]):
    """
    Return an t, x, y array for plotting based on params
    Also introduces noise to parameter draws    
    """

    # Decide on time
    t = np.linspace(0, 10, 200)
    ts = np.repeat(t[:, np.newaxis], theta.shape[0], axis=1)


    if theta.ndim == 1:
        theta = theta[np.newaxis, :]
    if noise:
        # Draw parameter (theta) values from normal distributions
 
        gs = np.random.normal(loc=theta[0][0], scale=noise[0], size=len(t))
        Ls = np.random.normal(loc=theta[0][1], scale=noise[1], size=len(t))
        theta_os =  np.random.normal(loc=theta[0][2], scale=noise[2], size=len(t))

        '''
        plt.scatter(t, gs, label='gs')
        plt.scatter(t, Ls, label='Ls')
        plt.scatter(t, theta_os, label='thetas')
        plt.legend()
        plt.show()
        '''
        


    # Okay from this then write q and p
    if noise:
        x = np.array([np.sin(theta_os[i] * np.cos( np.sqrt(gs[i] / Ls[i]) * t[i])) * Ls[i] for i, _ in enumerate(t)])
        y = np.array([Ls[i] - np.cos(theta_os[i] * np.cos(np.sqrt(gs[i] / Ls[i]) * t[i])) * Ls[i] for i, _ in enumerate(t)])

        dx_dt = np.array([theta_os[i] * Ls[i] * (- np.sqrt( gs[i] / Ls[i])) * np.sin( t[i] * np.sqrt( gs[i] / Ls[i])) * np.cos(theta_os[i] * np.cos(t[i] * np.sqrt( gs[i] / Ls[i]))) for i, _ in enumerate(t)])
        dy_dt = np.array([theta_os[i] * Ls[i] * (- np.sqrt( gs[i] / Ls[i])) * np.sin( t[i] * np.sqrt( gs[i] / Ls[i])) * np.sin(theta_os[i] * np.cos(t[i] * np.sqrt( gs[i] / Ls[i]))) for i, _ in enumerate(t)])
    
    else:
        x = ( np.sin(theta[:, 2] * np.cos( np.sqrt(theta[:, 0] / theta[:, 1]) * ts)) * theta[:, 1] ) 
        y = ( theta[:, 1] - np.cos(theta[:, 2] * np.cos(np.sqrt(theta[:, 0] / theta[:, 1]) * ts)) * theta[:, 1] ) 

        dx_dt = theta[:, 2] * theta[:, 1] * (- np.sqrt(theta[:, 0]/theta[:, 1])) * np.sin( ts * np.sqrt(theta[:, 0]/theta[:, 1])) * np.cos(theta[:, 2]* np.cos(ts * np.sqrt(theta[:, 0]/theta[:,1])))
        #np.sin(theta[:, 2]) * theta[:, 1] * np.sqrt(theta[:, 0] / theta[:, 1]) * np.sin(np.sqrt(theta[:, 0] / theta[:, 1]) * ts)
        #* theta[:, 1] ( np.sin(theta[:, 2] * np.cos( np.sqrt(theta[:, 0] / theta[:, 1]) * ts)) * theta[:, 1] )
        dy_dt = theta[:, 2] * theta[:, 1] * (- np.sqrt(theta[:, 0] / theta[:, 1])) * np.sin(ts * np.sqrt(theta[:, 0] / theta[:, 1])) * np.sin(theta[:, 2] * np.cos(ts * np.sqrt(theta[:, 0] / theta[:, 1])))
    return t, x, y, dx_dt, dy_dt


def eval(theta, t, seed=None, noise = False):
    """Evaluate the pendulum at time `t`"""
    starting_theta = np.pi / 4

    if theta.ndim == 1:
        theta = theta[np.newaxis, :]
    if noise:
        # Draw parameter (theta) values from normal distributions
 
        gs = np.random.normal(loc=theta[0][0], scale=noise[0], size=len(t))
        Ls = np.random.normal(loc=theta[0][1], scale=noise[1], size=len(t))
        theta_os =  np.random.normal(loc=theta[0][2], scale=noise[2], size=len(t))

    

    # Okay from this then write q and p
    if noise:
        x = np.array([np.sin(theta_os[i] * np.cos( np.sqrt(gs[i] / Ls[i]) * t[i])) * Ls[i] for i, _ in enumerate(t)])
        y = np.array([Ls[i] - np.cos(theta_os[i] * np.cos(np.sqrt(gs[i] / Ls[i]) * t[i])) * Ls[i] for i, _ in enumerate(t)])

        dx_dt = np.array([theta_os[i] * Ls[i] * (- np.sqrt( gs[i] / Ls[i])) * np.sin( t[i] * np.sqrt( gs[i] / Ls[i])) * np.cos(theta_os[i] * np.cos(t[i] * np.sqrt( gs[i] / Ls[i]))) for i, _ in enumerate(t)])
        dy_dt = np.array([theta_os[i] * Ls[i] * (- np.sqrt( gs[i] / Ls[i])) * np.sin( t[i] * np.sqrt( gs[i] / Ls[i])) * np.sin(theta_os[i] * np.cos(t[i] * np.sqrt( gs[i] / Ls[i]))) for i, _ in enumerate(t)])
     
    else:
        x = ( np.sin(theta[:, 2] * np.cos( np.sqrt(theta[:, 0] / theta[:, 1]) * t)) * theta[:, 1] ) 
        y = ( theta[:, 1] - np.cos(theta[:, 2] * np.cos(np.sqrt(theta[:, 0] / theta[:, 1]) * t)) * theta[:, 1] ) 

        dx_dt = theta[:, 2] * theta[:, 1] * (- np.sqrt(theta[:, 0]/theta[:, 1])) * np.sin( t * np.sqrt(theta[:, 0]/theta[:, 1])) * np.cos(theta[:, 2]* np.cos(t * np.sqrt(theta[:, 0]/theta[:,1])))
        #np.sin(theta[:, 2]) * theta[:, 1] * np.sqrt(theta[:, 0] / theta[:, 1]) * np.sin(np.sqrt(theta[:, 0] / theta[:, 1]) * ts)
        #* theta[:, 1] ( np.sin(theta[:, 2] * np.cos( np.sqrt(theta[:, 0] / theta[:, 1]) * ts)) * theta[:, 1] )
        dy_dt = theta[:, 2] * theta[:, 1] * (- np.sqrt(theta[:, 0] / theta[:, 1])) * np.sin(t * np.sqrt(theta[:, 0] / theta[:, 1])) * np.sin(theta[:, 2] * np.cos(t * np.sqrt(theta[:, 0] / theta[:, 1])))
    return x, y, dx_dt, dy_dt


# Below are the two summary statistics I'm using to optimize the SBI, one of them gets MSE
# for all of the positions and momentum for every moment in time in the time series
# one of them grabs just four moments in time
# As you might expect, the MSE one does better

def get_MSE(theta, theta_o, seed=None):
    """
    Return the mean-squared error (MSE) i.e. Euclidean distance from the observation function
    """
    _, x, y, mom_x, mom_y = create_t_p_q_noise(theta_o, seed=seed)  # truth
    _, x_, y_, mom_x_, mom_y_ = create_t_p_q_noise(theta, seed=seed)  # simulations
    return np.mean(np.sqrt(np.square(y_ - y) + np.square(x_ - x) + np.square(mom_y_ - mom_y) + np.square(mom_x_ - mom_x)), axis=0, keepdims=True).T  # MSE

def get_4_values(theta, seed=None):
    """
    Return 4 'y' values corresponding to t=-0.5,0,0.75 as summary statistic vector
    """
    return np.array(
        [
            eval(theta, 0, seed=seed)[0],
            eval(theta, 0.25, seed=seed)[0],
            eval(theta, 0.5, seed=seed)[0],
            eval(theta, 0.75, seed=seed)[0],
            eval(theta, 0, seed=seed)[1],
            eval(theta, 0.25, seed=seed)[1],
            eval(theta, 0.5, seed=seed)[1],
            eval(theta, 0.75, seed=seed)[1],
            eval(theta, 0, seed=seed)[2],
            eval(theta, 0.25, seed=seed)[2],
            eval(theta, 0.5, seed=seed)[2],
            eval(theta, 0.75, seed=seed)[2],
            eval(theta, 0, seed=seed)[3],
            eval(theta, 0.25, seed=seed)[3],
            eval(theta, 0.5, seed=seed)[3],
            eval(theta, 0.75, seed=seed)[3],
        ]
    ).T


# Following this example: 
# https://github.com/mackelab/sbi/blob/main/tutorials/10_crafting_summary_statistics.ipynb
# Also could look at Miles' example:
# file:///Users/rnevin/Zotero/storage/L52NCDXF/simulation-based-inference.html
# Now redo with the pendulum simulator
# params are g and L and theta_not
prior_min = [5, 0, 0] # range on all of the params
prior_max = [15, 10, np.pi/2]

prior = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
)

theta_o = np.array([10, 5, np.pi/4])
t, x, y, mom_x, mom_y = create_t_p_q_noise(theta_o, noise = False)


fig = plt.figure(figsize = (10,4))
#fig.title('True values')
ax = fig.add_subplot(121)
for i in range(100):
    tn, xn, yn, mom_xn, mom_yn = create_t_p_q_noise(theta_o)
    ax.plot(t, xn, color='grey')

ax.plot(t, x, "red")
ax.set_xlabel('time')
ax.set_ylabel('x pos')

ax1 = fig.add_subplot(122)
for i in range(100):
    tn, xn, yn, mom_xn, mom_yn = create_t_p_q_noise(theta_o)
    ax1.plot(t, yn, color='grey')
ax1.plot(t, y, "red")
ax1.set_xlabel('time')
ax1.set_ylabel('y pos')
plt.show()



# Now plot momentum and position
fig = plt.figure(figsize = (10,4))
#fig.title('True values')
ax = fig.add_subplot(121)
ax.plot(x, mom_x, "k")
ax.set_xlabel('x pos')
ax.set_ylabel('momentum in x')

ax1 = fig.add_subplot(122)
ax1.plot(y, mom_y, "k")
ax1.set_xlabel('y pos')
ax1.set_ylabel('momentum in y')

plt.show()



plt.clf()
fig = plt.figure(figsize = (10,4))

ax = fig.add_subplot(121)

t, x_truth, y_truth, mom_x_truth, mom_y_truth = create_t_p_q_noise(theta_o, noise = noiz)

ax.plot(t, x_truth, "k", zorder=1, label="truth")
n_samples = 100
theta = prior.sample((n_samples,))
t, x, y, mom_x, mom_y = create_t_p_q_noise(theta.numpy(), noise = noiz)
print('shape t', np.shape(t))

# maybe rearrange by theta value
print(np.shape(t), np.shape(x), np.shape(theta.numpy()))
# okay so there are 10 samples of the posterior
# and 200 points in time to measure

if n_samples > 99:
    ax.plot(t, x, "grey", zorder=0)
else:

    for i in range(n_samples):
        im = ax.plot(t, x[:,i], label=str(theta.numpy()[i]))#c = matplotlib.cm.hot(theta.numpy()[i,0]))


# was: ax.plot(t, x, "grey", zorder=0)
#im = ax.plot(t, x, c=matplotlib.cm.hot(theta.numpy()[:,0]), zorder=0)
#plt.colorbar(im, label='g value')
plt.legend()
ax.set_xlabel('time')
ax.set_ylabel('x')

ax1 = fig.add_subplot(122)
ax1.plot(t, y_truth, "k", zorder=1, label="truth")

if n_samples > 99:
    ax1.plot(t, y, "grey", zorder=0)
else:
    for i in range(n_samples):
        im = ax1.plot(t, y[:,i], label=str(theta.numpy()[i]))#c = matplotlib.cm.hot(theta.numpy()[i,0]))


#ax1.plot(t, y, "grey", zorder=0)
plt.legend()
ax1.set_xlabel('time')
ax1.set_ylabel('y')
plt.show()

# Do this but for momentum
plt.clf()
fig = plt.figure(figsize = (10,4))

ax = fig.add_subplot(121)

ax.plot(x_truth, mom_x_truth, "k", zorder=1, label="truth")



if n_samples > 99:
    ax.plot(x, mom_x, "grey", zorder=0)
else:

    for i in range(n_samples):
        im = ax.plot(x[:,i], mom_x[:,i], label=str(theta.numpy()[i]))#c = matplotlib.cm.hot(theta.numpy()[i,0]))


# was: ax.plot(t, x, "grey", zorder=0)
#im = ax.plot(t, x, c=matplotlib.cm.hot(theta.numpy()[:,0]), zorder=0)
#plt.colorbar(im, label='g value')
plt.legend()
ax.set_xlabel('x')
ax.set_ylabel('mom x')

ax1 = fig.add_subplot(122)
ax1.plot(y_truth, mom_y_truth, "k", zorder=1, label="truth")

if n_samples > 99:
    ax1.plot(y, mom_y, "grey", zorder=0)
else:
    for i in range(n_samples):
        im = ax1.plot( y[:,i], mom_y[:,i], label=str(theta.numpy()[i]))#c = matplotlib.cm.hot(theta.numpy()[i,0]))


#ax1.plot(t, y, "grey", zorder=0)
plt.legend()
ax1.set_xlabel('y')
ax1.set_ylabel('mom y')
plt.show()




# Lets see how the four values do
x = get_4_values(theta.numpy())
x = torch.as_tensor(x, dtype=torch.float32)

# Macke lab
inference = SNPE(prior)

_ = inference.append_simulations(theta, x).train()
posterior = inference.build_posterior()

x_o = torch.as_tensor(get_4_values(theta_o), dtype=float)

theta_p = posterior.sample((10000,), x=x_o)

fig, axes = pairplot(
    theta_p,
    limits=list(zip(prior_min, prior_max)),
    ticks=list(zip(prior_min, prior_max)),
    figsize=(7, 7),
    labels=["g", "L", "theta_init"],
    points_offdiag={"markersize": 6},
    points_colors="r",
    points=theta_o,
)
plt.show()

x_o_t, x_o_x, x_o_y, x_o_mom_x, x_o_mom_y = create_t_p_q_noise(theta_o)
plt.plot(x_o_t, x_o_x, "k", zorder=1, label="truth")
theta_p = posterior.sample((100,), x=x_o)
ind_10_highest = np.argsort(np.array(posterior.log_prob(theta=theta_p, x=x_o)))[-10:]
theta_p_considered = theta_p[ind_10_highest, :]
x_t, x_x, x_y, x_mom_x, x_mom_y = create_t_p_q_noise(theta_p_considered.numpy())
plt.plot(x_t, x_x, "green", zorder=0)

ind_10_lowest = np.argsort(np.array(posterior.log_prob(theta=theta_p, x=x_o)))[0:10]
theta_p_considered = theta_p[ind_10_lowest, :]
x_t, x_x, x_y, x_mom_x, y_mom_y = create_t_p_q_noise(theta_p_considered.numpy())
plt.plot(x_t, x_x, "red", zorder=0)

plt.legend()
plt.show()




# Lets see how the MSE does
# So randomly sample from the prior and then compare the 
# MSE of the positions produced from that prior
# to the acutal positions at all of the times
# (MSE samples at all times)
theta = prior.sample((1000,))
# These are the losses
x = get_MSE(theta.numpy(), theta_o)

print('actual theta', theta_o)
print('thetas', theta)
print('MSE', x)

theta = torch.as_tensor(theta, dtype=torch.float32)
x = torch.as_tensor(x, dtype=torch.float32)

# Sets up the neural posterior estimator
inference = SNPE(prior)
# Links theta and data through the MSE (x), 
# it does not explicitly know the formula of a pendulum,
# but it implicitly does through the MSE
# trains sequential (or not?) neural posterior estimator
# to learn the link between the two
_ = inference.append_simulations(theta, x).train()
# Now builds the posterior using SNPE
posterior = inference.build_posterior()

x_o = torch.as_tensor(
    [
        [
            0.0,
        ]
    ]
)

# fills empty tensor with samples from the posterior
theta_p = posterior.sample((10000,), x=x_o)

fig, axes = pairplot(
    theta_p,
    limits=list(zip(prior_min, prior_max)),
    ticks=list(zip(prior_min, prior_max)),
    figsize=(7, 7),
    labels=["g", "L", "theta_0"],
    points_offdiag={"markersize": 6},
    points_colors="r",
    points=theta_o,
)
plt.show()


x_o_t, x_o_x, x_o_y, mom_o_x, mom_o_y = create_t_p_q_noise(theta_o)
plt.plot(x_o_t, x_o_x, "k", zorder=1, label="truth")

theta_p = posterior.sample((10,), x=x_o)
x_t, x_x, x_y, mom_x, mom_y = create_t_p_q_noise(theta_p.numpy())
plt.plot(x_t, x_x, "grey", zorder=0)
plt.ylabel('pos in x')
plt.legend()
plt.show()

plt.plot(x_o_t, mom_o_x, "k", zorder=1, label="truth")

plt.plot(x_t, mom_x, "grey", zorder=0)
plt.ylabel('mom in x')
plt.legend()
plt.show()

## MSE does better