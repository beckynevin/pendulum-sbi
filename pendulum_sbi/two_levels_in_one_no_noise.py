# Following the formulation from our overleaf doc
# p(\theta_2,\theta_3,\alpha|t,x) ~  (x - \theta_2 * sin(\theta_3 * cos(\sqrt{GM / r^2 \theta_2} * t)
# p(t) p(\theta_2) p(\theta_3) p(\alpha) 

# First term is akin to a likelihood
# second term is a bunch of priors

import torch
import torch.nn as nn

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

# stuff for MCMC:
import emcee
import corner

# modules that I wrote
import sys
sys.path.insert(0,"..") # this is so it can find pendulum_sbi
from pendulum_sbi.pendulum import pendulum


# setting the sigmas on the noise distribution
# you will eventually draw the parameter values from a normal
# distribution with the width given by these values
noise = [0.1,0.0,0.0] # here, there's only noise on g

def model(theta, t, M, r):
    # given thetas and ts, solves for x
    theta_2, theta_3, G = theta
    # x - \theta_2 * sin(\theta_3 * cos(\sqrt{GM / r^2 \theta_2} * t
    return theta_2 * np.sin(theta_3 * np.cos(np.sqrt((G * M) / (r**2 * theta_2))*t))


# Now set up the likelihood, which will rely on the simulator
def log_likelihood(theta, t, x, yerr, M, r):
    x_model = model(theta, t, M, r)
    sigma2 = yerr**2 
    return -0.5 * np.sum((x - x_model) ** 2 / 2 * sigma2) #+ np.log(sigma2))

def log_prior_uniform(theta):
    g, L, theta_o = theta

    if 5.0 < g < 15.0 and 0.0 < L < 10.0 and 0 < theta_o < np.pi/2:
        return 0.0    
    return -np.inf

def log_prior_normal(theta):
    L, theta_0, G = theta
    
    # right off the bat, disallow negative numbers:
    if L < 0.0 or theta_0 < 0.0 or theta_0 > np.pi/2 or G < 0.0:
        prior = -np.inf
        
        return prior
    
    mu_L, sigma_L = 5, 2
    mu_theta_0, sigma_theta_0 = 0.75, 0.2
    mu_G, sigma_G = 10, 2
    prior = np.log(1.0/(np.sqrt(2*np.pi)*sigma_G))-0.5*(G-mu_G)**2/sigma_G**2 + np.log(1.0/(np.sqrt(2*np.pi)*sigma_L))-0.5*(L - mu_L)**2/sigma_L**2 + np.log(1.0/(np.sqrt(2*np.pi)*sigma_theta_0))-0.5*(theta_0 - mu_theta_0)**2/sigma_theta_0**2
    
    return prior


def log_probability(theta, t, x, yerr, M, r):
    
    lp = log_prior_normal(theta)
    if not np.isfinite(lp):
        return -np.inf

    p_density =  lp + log_likelihood(theta, t, x, yerr, M, r)
    '''
    if p_density < -15:
        print('input values', theta)
        print('posterior density', p_density)
        print('prior', lp)
        print('log likelihood', log_likelihood(theta, t, y, yerr))
    '''
    return p_density

#def log_likelihood(theta, t, x, yerr, M, r):
time =  np.linspace(0, 10, 100)
theta_true = np.array([5, np.pi/4, 10])
M = 10
r = 10
x_true = model(theta_true, time, M, r)

theta_o = np.array([10, np.pi/4, 10])
x_o = model(theta_true, time, M, r)

print(log_likelihood(theta_o, time, x_true, 1, M, r))

plt.clf()
plt.plot(time, x_o.flatten(), color = 'black')

color_list = ['#F0A202','#F18805','#D95D39','#90B494','#7B9E89']
for i, offset_Gs in enumerate(np.linspace(5,15,5)):
    print(i, offset_Gs)
    theta = np.array([5, np.pi/4, offset_Gs])
    offset_position = model(theta, time, M, r)
    plt.plot(time, offset_position.flatten(), color = color_list[i])
    plt.annotate(f'log L = {round(log_likelihood(theta, time, x_o, 1, M, r),2)}, G = {round(offset_Gs, 2)}', 
        xy = (0.02, 0.3 - 0.05*i), xycoords = 'axes fraction', color = color_list[i])
plt.xlabel('time')
plt.ylabel('x position')
plt.show()


# Okay run the two-levels-in-one MCMC
# why not start around the true position?
start = np.array([5, np.pi/4, 5])#theta_true
#theta_true = np.array([5, np.pi/4, 10])
#np.array([7, 5, np.pi/4])

pos = start + np.array([2e0,1e-1,1e0]) * np.random.randn(32, 3)

print('starting positiong', pos)

print(np.shape(pos))
# Make a histogram of the starting positions
plt.clf()
fig = plt.figure(figsize = (5,7))
ax0 = fig.add_subplot(311)
ax0.hist(pos[:,0], color = '#CCC9DC')
ax0.axvline(x = theta_true[0])
ax0.set_xlabel('L')

ax1 = fig.add_subplot(312)
ax1.hist(pos[:,1], color = '#59C9A5')
ax1.axvline(x = theta_true[1])
ax1.set_xlabel('theta_0')

ax2 = fig.add_subplot(313)
ax2.hist(pos[:,2], color = '#F3C969')
ax2.axvline(x = theta_true[2])
ax2.set_xlabel('G')

plt.show()

nwalkers, ndim = pos.shape


# Generate some synthetic data that you know the solution to 

t =  np.linspace(0, 20, 100)
run = True

filename = "MCMC_chains/two_in_one.h5"
if run:
# Optionally save chain
# Set up the backend
# Don't forget to clear it in case the file already exists
    true = theta_true
    y = model(theta_true, t, M, r)
    yerr = 0.2 * np.ones(np.shape(y))

    plt.clf()
    plt.errorbar(t, y, yerr = yerr)
    plt.show()


    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    
    # So the simple setup is that you have one true data array to input
    # that is 100 t steps long and you're comparing to it
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(t, y, yerr, M, r), backend = backend
    )
    # run it
    sampler.run_mcmc(pos, 1000, progress=True);
else:
    sampler = emcee.backends.HDFBackend(filename)

# get the chains
plt.clf()
fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["L", "theta_0", "G"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    if run:
        ax.axhline(y = theta_true[i])
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
plt.show()

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)
# Now build the hierarchical part by sampling K samples from these chains
# Currently they can just have the above shape

print(flat_samples)
# So for each of these we're just grabbing the g values?
print(flat_samples[:,0])
G = flat_samples[:,2]

plt.clf()
plt.hist(G, bins = 20, alpha = 0.5, color = 'green')
plt.xlabel('G sampling')
plt.show()   


plt.clf()
fig = corner.corner(
    flat_samples, truths=theta_true,
    labels=[
        r"$L$",
        r"$Theta_0$",
        r"$G$",
    ],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},
)
plt.show()
