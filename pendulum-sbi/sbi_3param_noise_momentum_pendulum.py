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

def create_t_p_q_noise(theta, noise=[0.1,0.1,0.1]):
    """
    Return an t, x, y array for plotting based on params
    Also introduces noise to parameter draws    
    """

    

    # Decide on time, here I'm sampling a few oscillations, but this could be
    # something you import or change
    
    t = np.linspace(0, 10, 200)
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
    

    '''
    # This was the old code for only running on a 1D theta input array:
    else:
        gs = np.random.normal(loc=theta[0][0], scale=noise[0], size=np.shape(t))
        Ls = np.random.normal(loc=theta[0][1], scale=noise[1], size=np.shape(t))
        theta_os =  np.random.normal(loc=theta[0][2], scale=noise[2], size=np.shape(t))

        theta_t = np.array([theta_os[i] * math.cos(np.sqrt(gs[i] / Ls[i]) * t[i]) for i, _ in enumerate(t)])
        x = np.array([Ls[i] * math.sin(theta_t[i]) for i, _ in enumerate(t)])
        y = np.array([-Ls[i] * math.cos(theta_t[i]) for i, _ in enumerate(t)])

        # Okay and what about taking the time derivative?
        dx_dt = np.array([-Ls[i] * theta_os[i] * np.sqrt(gs[i] / Ls[i]) * math.cos(theta_t[i]) * math.sin( np.sqrt(gs[i] / Ls[i]) * t[i]) for i, _ in enumerate(t)])
        dy_dt = np.array([-Ls[i] * theta_os[i] * np.sqrt(gs[i] / Ls[i]) * math.sin(theta_t[i]) * math.sin( np.sqrt(gs[i] / Ls[i]) * t[i]) for i, _ in enumerate(t)])
    

    '''
    

    return t, x, y, dx_dt, dy_dt

# This is a wrapper for the SBI that allows it use the position and momentum at all points in time
# as its optimization target
# i.e., its trying to optimize its hyperparameters so that the given a theta input the xs 
# (xs == position and momentum at all moments in time) match
# that of the simulator

# TO DO: probably could just use the above function and get rid of time

def eval_for_all_time(theta, noise):
    """Evaluate the pendulum at all times"""

    # nested for loop
    # output needs to be (n,len(t))
    x = np.zeros((theta.shape[0],len(t)))
    y = np.zeros((theta.shape[0],len(t)))
    dx_dt = np.zeros((theta.shape[0],len(t)))
    dy_dt = np.zeros((theta.shape[0],len(t)))
    for n in range(theta.shape[0]):
    
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

# Similar to the above but draws at one specific moment in time
def eval(theta, t, noise):
    """
    Evaluate the pendulum at time `t`
    But you still need to account for multiple draws of theta
    """

    if theta.ndim == 1:
        theta = theta[np.newaxis, :]

    
    x = np.zeros((theta.shape[0]))
    y = np.zeros((theta.shape[0]))
    dx_dt = np.zeros((theta.shape[0]))
    dy_dt = np.zeros((theta.shape[0]))
    # Do a loop over n still
    for n in range(theta.shape[0]):
        
        # Draw the individual values
        g = np.random.normal(loc=theta[n][0], scale=noise[0])#, size = np.shape(theta)[0])#, size=np.shape(t))
        L = np.random.normal(loc=theta[n][1], scale=noise[1])#, size=np.shape(t))
        theta_o =  np.random.normal(loc=theta[n][2], scale=noise[2])#, size=np.shape(t))

        theta_t = theta_o * math.cos(np.sqrt(g / L) * t)
        x[n] = L * math.sin(theta_t) 
        y[n] = -L * math.cos(theta_t) 

        # Okay and what about taking the time derivative?
        dx_dt[n] = -L * theta_o * np.sqrt(g / L) * math.cos(theta_t) * math.sin( np.sqrt(g / L) * t) 
        dy_dt[n] = -L * theta_o * np.sqrt(g / L) * math.sin(theta_t) * math.sin( np.sqrt(g / L) * t) 
    


    return x, y, dx_dt, dy_dt

# MSE = mean squared error
# This is another tool for the SBI to use to optimize
# Given a noisy pendulum SBI run, it compares the true position and velocities
# to those predicted by the SBI
def get_MSE(theta, theta_o, noise = [0.1,0.1,0.1]):
    """
    Return the mean-squared error (MSE) i.e. Euclidean distance from the observation function
    """
    _, x, y, mom_x, mom_y = create_t_p_q_noise(theta_o, noise = noise)  # truth
    _, x_, y_, mom_x_, mom_y_ = create_t_p_q_noise(theta, noise = noise)  # simulations
    return np.mean(np.sqrt(np.square(y_ - y) + np.square(x_ - x) + np.square(mom_y_ - mom_y) + np.square(mom_x_ - mom_x)), axis=0, keepdims=True).T  # MSE

# Another SBI optimization tool, this uses the position and velocities at multiple specific moments in time
# TO DO: It is no longer drawing only four values
def get_4_values(theta, noise = [0.1,0.1,0.1]):
    """
    Return 4 'y' values corresponding to t=-0.5,0,0.75 as summary statistic vector
    """

    return np.array(
        [
            eval(theta, 0, noise = noise)[0],
            eval(theta, 0.25, noise = noise)[0],
            eval(theta, 0.5, noise = noise)[0],
            eval(theta, 0.75, noise = noise)[0],
            eval(theta, 0, noise = noise)[1],
            eval(theta, 0.25, noise = noise)[1],
            eval(theta, 0.5, noise = noise)[1],
            eval(theta, 0.75, noise = noise)[1],
            eval(theta, 0, noise = noise)[2],
            eval(theta, 0.25, noise = noise)[2],
            eval(theta, 0.5, noise = noise)[2],
            eval(theta, 0.75, noise = noise)[2],
            eval(theta, 0, noise = noise)[3],
            eval(theta, 0.25, noise = noise)[3],
            eval(theta, 0.5, noise = noise)[3],
            eval(theta, 0.75, noise = noise)[3],
        ]
    ).T

# The next step is to implement SBI using the Macklab example from:
# https://github.com/mackelab/sbi/blob/main/tutorials/10_crafting_summary_statistics.ipynb

# Also could look at Miles' example for reference:
# file:///Users/rnevin/Zotero/storage/L52NCDXF/simulation-based-inference.html
# Now redo with the pendulum simulator

# params are g and L and theta_not, which is the initial theta position
# Here, I'm setting up the min and max on the parameters for an Uniform distribution
# The "true" values for this experiment sit right in the middle.
prior_min = [5, 0, 0] # range on all of the params
prior_max = [15, 10, np.pi/2]

prior = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
)

# TO DO: terrible notation flying around, theta_o is referring to
# parameters and to the starting theta of the pendulum

# True parameter values
theta_o = np.array([10, 5, np.pi/4])


'''
t, x, y, mom_x, mom_y = create_t_p_q_noise(theta_o, noise = noiz)
fig = plt.figure(figsize = (10,4))
#fig.title('True values')
ax = fig.add_subplot(121)
ax.plot(t, x[0,:], "k")
ax.set_xlabel('time')
ax.set_ylabel('x pos')

ax1 = fig.add_subplot(122)
ax1.plot(t, y[0,:], "k")
ax1.set_xlabel('time')
ax1.set_ylabel('y pos')
plt.show()

# Now plot momentum and position
fig = plt.figure(figsize = (10,4))
#fig.title('True values')
ax = fig.add_subplot(121)
ax.plot(x[0,:], mom_x[0,:], "k")
ax.set_xlabel('x pos')
ax.set_ylabel('momentum in x')

ax1 = fig.add_subplot(122)
ax1.plot(y[0,:], mom_y[0,:], "k")
ax1.set_xlabel('y pos')
ax1.set_ylabel('momentum in y')

plt.show()
'''



plt.clf()
fig = plt.figure(figsize = (10,4))

ax = fig.add_subplot(121)

# First, visualize by running these true thetas through the simulator with some
# noise and plotting the x, y, v_x, v_y
t, x_truth, y_truth, mom_x_truth, mom_y_truth = create_t_p_q_noise(theta_o, noise = noiz)

ax.plot(t, x_truth[0,:], "k", zorder=1, label="truth")

# Also make 1000 draws from the priors and get the position and velocities from these
n_samples = 100
theta = prior.sample((n_samples,))

t, x, y, mom_x, mom_y = create_t_p_q_noise(theta.numpy(), noise = noiz)


# If there are a bunch of samples, only plot a few of them
if n_samples > 99:
    for i in range(10):
        if i == 0:
            im = ax.plot(t, x[i,:], label=str(theta.numpy()[i]), color = 'grey')#c = matplotlib.cm.hot(theta.numpy()[i,0]))
        else:
            im = ax.plot(t, x[i,:], color = 'grey')# don't label most
else:
    for i in range(n_samples):
        if i == 0:
            im = ax.plot(t, x[i,:], label=str(theta.numpy()[i]), color = 'grey')#c = matplotlib.cm.hot(theta.numpy()[i,0]))
        else:
            im = ax.plot(t, x[i,:], color = 'grey')# don't label most


plt.legend()
ax.set_xlabel('time')
ax.set_ylabel('x')

ax1 = fig.add_subplot(122)
ax1.plot(t, y_truth[0,:], "k", zorder=1, label="truth")

if n_samples > 99:
    for i in range(10):
        if i == 0 :
            im = ax1.plot(t, y[i,:], label=str(theta.numpy()[i]), color = 'grey')#c = matplotlib.cm.hot(theta.numpy()[i,0]))
        else:
            im = ax1.plot(t, y[i,:], color = 'grey')# don't label most
else:
    for i in range(n_samples):
        if i == 0 :
            im = ax1.plot(t, y[i,:], label=str(theta.numpy()[i]), color = 'grey')#c = matplotlib.cm.hot(theta.numpy()[i,0]))
        else:
            im = ax1.plot(t, y[i,:], color = 'grey')# don't label most

plt.legend()
ax1.set_xlabel('time')
ax1.set_ylabel('y')
plt.show()

# Same thing for velocity, which again, we're calling momentum
plt.clf()
fig = plt.figure(figsize = (10,4))

ax = fig.add_subplot(121)

ax.plot(x_truth[0,:], mom_x_truth[0,:], "k", zorder=1, label="truth")



if n_samples > 99:
    ax.plot(x, mom_x, "grey", zorder=0)
else:

    for i in range(n_samples):
        im = ax.plot(x[i,:], mom_x[i,:])
    
plt.legend()
ax.set_xlabel('x')
ax.set_ylabel('mom x')

ax1 = fig.add_subplot(122)
ax1.plot(y_truth[0,:], mom_y_truth[0,:], "k", zorder=1, label="truth")

if n_samples > 99:
    ax1.plot(y, mom_y, "grey", zorder=0)
else:
    for i in range(n_samples):
        im = ax1.plot( y[i,:], mom_y[i,:])
     
plt.legend()
ax1.set_xlabel('y')
ax1.set_ylabel('mom y')
plt.show()


# Time to get into SBI

# First, experiment with the summary statistic of drawing multiple (but not all)
# moments in time

# x will be the 'connection point' that the SBI is trying to connect with the 
# thetas
x = get_4_values(theta.numpy(), noise = noiz)
x = torch.as_tensor(x, dtype=torch.float32)

# call sequential neural posterior estimation
inference = SNPE(prior)

# Now give it what you want, which is a tool that will connect the summary
# statistic with thetas
# and train
_ = inference.append_simulations(theta, x).train()

# from that build a posterior
posterior = inference.build_posterior()

# get the true summary statistical from the true thetas (with noise)
x_o = torch.as_tensor(get_4_values(theta_o, noise = noiz), dtype=float)

# the nice thing about SBI (or one of the nice things)
# is that you can sample from the posterior a bunch of times
# without re-running the training
# so to get the probability distribution of thetas, sample a bunch at the 
# true x values
theta_p = posterior.sample((n_samples,), x=x_o)

# this will make a corner plot that shows the true values of the parameters
# and the SBI solution
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

# Continuing on with this same solution, let's plot the time evolution
# for the 10 highest probability draws
# and the 10 worst
x_o_t, x_o_x, x_o_y, x_o_mom_x, x_o_mom_y = create_t_p_q_noise(theta_o)
plt.plot(x_o_t, x_o_x[0,:], "k", zorder=1, label="truth")
theta_p = posterior.sample((n_samples,), x=x_o)
ind_10_highest = np.argsort(np.array(posterior.log_prob(theta=theta_p, x=x_o)))[-10:]
theta_p_considered = theta_p[ind_10_highest, :]
x_t, x_x, x_y, x_mom_x, x_mom_y = create_t_p_q_noise(theta_p_considered.numpy())
for n in range(10):
    if n == 0:
        plt.plot(x_t, x_x[n,:], "green", zorder=0, label = 'Best draws')
    else:
        plt.plot(x_t, x_x[n,:], "green", zorder=0)

ind_10_lowest = np.argsort(np.array(posterior.log_prob(theta=theta_p, x=x_o)))[0:10]
theta_p_considered = theta_p[ind_10_lowest, :]
x_t, x_x, x_y, x_mom_x, y_mom_y = create_t_p_q_noise(theta_p_considered.numpy())
for n in range(10):
    if n == 0:
        plt.plot(x_t, x_x[n,:], "red", zorder=0, label = 'Worst draws')
    else:
        plt.plot(x_t, x_x[n,:], "red", zorder=0)

plt.legend()
plt.show()




# Lets see how the MSE does
# So randomly sample from the prior and then compare the 
# MSE of the positions produced from that prior
# to the acutal positions at all of the times
# (MSE samples at all times)
theta = prior.sample((n_samples,))
# Another way to think about x is that its the loss
x = get_MSE(theta.numpy(), theta_o, noise = noiz)


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
theta_p = posterior.sample((n_samples,), x=x_o)

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

## MSE does better, which makes sense because its using more info