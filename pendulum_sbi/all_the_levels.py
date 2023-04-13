# This is from the formulation from Sam

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

# So this model needs to have multiple levels of hierarchy
def model(eta, t):
    # eta can be a weird shape
    L, eta_0, G, phi = eta

    # x - \eta_2 * sin(\eta_3 * cos(\sqrt{GM / r^2 \eta_2} * t
    '''
    try:
        model = [L[x] * np.sin(eta_0[x] * np.cos(np.sqrt((G[x] * phi[x])/ L[x])*t)) for i, x in enumerate(np.shape(L)-1)]
        print('shape of model', np.shape(model))
    except TypeError:
        model = L * np.sin(eta_0 * np.cos(np.sqrt((G * phi) / L)*t))
    '''

    try:
        #print(len(L))
        model = np.zeros((len(t),len(L)))
        #print('shape of model', np.shape(model))
        for i in range(len(L)):
            model[:,i] = L[i] * np.sin(eta_0[i] * np.cos(np.sqrt((G[i] * phi[i]) / L[i])*t))
    
    except TypeError or IndexError:
        model = L * np.sin(eta_0 * np.cos(np.sqrt((G * phi) / L)*t))

    return model



# Now set up the likelihood, which will rely on the simulator
def log_likelihood(eta, t, x, yerr):
    x_model = model(eta, t)
    sigma2 = yerr**2 
    '''
    print('shape of x before', np.shape(x))
    x_rep = np.repeat(x[:, np.newaxis], np.shape(x_model)[1], axis = 1)
    print('shape of x after repeat', np.shape(x_rep))
    print(x_model)
    print(x_rep)
    print('0', -0.5 * np.sum((x_rep[:,0] - x_model[:,0]) ** 2 / 2 * sigma2))
    print('1', -0.5 * np.sum((x_rep[:,1] - x_model[:,1]) ** 2 / 2 * sigma2))
    '''
    return -0.5 * np.sum((x - x_model) ** 2 / 2 * sigma2) #+ np.log(sigma2))

def log_prior_uniform(eta):
    g, L, eta_o = eta

    if 5.0 < g < 15.0 and 0.0 < L < 10.0 and 0 < eta_o < np.pi/2:
        return 0.0    
    return -np.inf

def log_prior_normal(eta):
    L, eta_0, G, phi = eta
    # right off the bat, disallow negative numbers:
    if L.any() < 0.0 or eta_0.any() < 0.0 or eta_0.any() > np.pi/2 or G.any() < 0.0 or phi.any() < 0.0:
        prior = -np.inf
        
        return prior
    
    mu_L, sigma_L = 5, 2
    mu_eta_0, sigma_eta_0 = 0.75, 0.2
    mu_G, sigma_G = 10, 2
    mu_phi, sigma_phi = 1, 0.1
    prior = np.log(1.0/(np.sqrt(2*np.pi)*sigma_G))-0.5*(G-mu_G)**2/sigma_G**2 + np.log(1.0/(np.sqrt(2*np.pi)*sigma_L))-0.5*(L - mu_L)**2/sigma_L**2 + np.log(1.0/(np.sqrt(2*np.pi)*sigma_eta_0))-0.5*(eta_0 - mu_eta_0)**2/sigma_eta_0**2 + np.log(1.0/(np.sqrt(2*np.pi)*sigma_phi))-0.5*(phi - mu_phi)**2/sigma_phi**2
    
    return prior


def log_probability(eta, t, x, yerr):
    
    lp = log_prior_normal(eta)
    
    if not np.isfinite(lp.any()):
        return -np.inf

    p_density =  lp + log_likelihood(eta, t, x, yerr)
    '''
    if p_density < -15:
        print('input values', eta)
        print('posterior density', p_density)
        print('prior', lp)
        print('log likelihood', log_likelihood(eta, t, y, yerr))
    '''
    return p_density

#def log_likelihood(eta, t, x, yerr, M, r):
time =  np.linspace(0, 1, num = 100)
eta_true = np.array([[2, np.pi/4, 10, 1],[1, np.pi/4, 10, 1]])
eta_true_reshape = eta_true.reshape((4,2))
x_true = model(eta_true_reshape, time)

# okay now the eta array will have the top level always have the same shape:
# L, eta_0, G, phi = eta
# lets just make sure t is the same for all
# Here, there's two different pendulii
eta_o = np.array([[3, np.pi/4, 10, 1],[4, np.pi/4, 10, 1]])
eta_o_reshape = eta_o.reshape((4,2))
print(np.shape(eta_o_reshape))

#eta_o = np.array([10, np.pi/4, 10])
offset_position = model(eta_o_reshape, time)

'''
plt.clf()
for i in range(2):
    plt.plot(time, x_true[:,i].flatten(), color = 'black')
    plt.plot(time, offset_position[:,i].flatten(), color = 'green')
color_list = ['#F0A202','#F18805','#D95D39','#90B494','#7B9E89']
plt.xlabel('time')
plt.ylabel('x position')
plt.show()
'''


start = np.array([[1.5, np.pi/4, 10, 1],[1.5, np.pi/4, 10, 1]])
start_reshape = start.reshape((4,2))



pos = start_reshape + np.array([[2e-1,1e-1,1e0,1e0],[2e-1,1e-1,1e0,1e0]]).reshape((4,2)) * np.random.randn(10, 4, 2)
'''
# Make a histogram of the starting positions
plt.clf()
fig = plt.figure(figsize = (5,7))
ax0 = fig.add_subplot(311)
ax0.hist(pos[:,0,0], color = '#CCC9DC')
ax0.axvline(x = eta_true[0,0])
ax0.set_xlabel('L')

ax1 = fig.add_subplot(312)
ax1.hist(pos[:,0,1], color = '#59C9A5')
ax1.axvline(x = eta_true[0,1])
ax1.set_xlabel('eta_0')

ax2 = fig.add_subplot(313)
ax2.hist(pos[:,0,2], color = '#F3C969')
ax2.axvline(x = eta_true[0,2])
ax2.set_xlabel('G')

plt.show()
'''

nwalkers, ndim, npendulii = pos.shape


# Generate some synthetic data that you know the solution to 

run = True

filename = "MCMC_chains/all_the_levels.h5"
if run:
# Optionally save chain
# Set up the backend
# Don't forget to clear it in case the file already exists
    true = eta_true_reshape
    y = model(eta_true_reshape, time)
    yerr = 0.2 * np.ones(np.shape(y))

    print(np.shape(y))
    
    plt.clf()
    plt.errorbar(time, y[:,0], yerr = yerr[:,0])
    plt.errorbar(time, y[:,1], yerr = yerr[:,1])
    plt.show()


    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    
    # So the simple setup is that you have one true data array to input
    # that is 100 t steps long and you're comparing to it
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(time, y, yerr), backend = backend
    )
    # run it
    sampler.run_mcmc(pos, 1000, progress=True);
else:
    sampler = emcee.backends.HDFBackend(filename)

# get the chains
plt.clf()
fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["L", "eta_0", "G"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    if run:
        ax.axhline(y = eta_true[i])
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
    flat_samples, truths=eta_true,
    labels=[
        r"$L$",
        r"$eta_0$",
        r"$G$",
    ],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},
)
plt.show()
