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

def model_collapsed(eta, t):
    # eta can be a weird shape
    L_1, theta_0_1, L_2, theta_0_2, G, phi = eta

    # x - \eta_2 * sin(\eta_3 * cos(\sqrt{GM / r^2 \eta_2} * t
    

    model_1 = L_1 * np.sin(theta_0_1 * np.cos(np.sqrt((G * phi) / L_1)*t))
    model_2 = L_2 * np.sin(theta_0_2 * np.cos(np.sqrt((G * phi) / L_2)*t))

    model = np.append(model_1, model_2)
    return model


# Now set up the likelihood, which will rely on the simulator
def log_likelihood(eta, t, x, yerr):
    x_model = model_collapsed(eta, t)
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
    L_1, theta_0_1, L_2, theta_0_2, G, phi = eta
    # right off the bat, disallow negative numbers:
    if L_1 < 0.0 or L_2 < 0.0 or theta_0_1.any() < 0.0 or theta_0_1 > np.pi/2 or theta_0_1.any() < 0.0 or theta_0_1 > np.pi/2 or G < 0.0 or phi < 0.0:
        prior = -np.inf
        
        return prior
    
    mu_L_1, sigma_L_1 = 5, 2
    mu_theta_0_1, sigma_theta_0_1 = 0.75, 0.2
    
    mu_L_2, sigma_L_2 = 5, 2
    mu_theta_0_2, sigma_theta_0_2 = 0.75, 0.2
    
    mu_G, sigma_G = 10, 2
    mu_phi, sigma_phi = 1, 0.1

    prior = np.log(1.0/(np.sqrt(2*np.pi)*sigma_G))-0.5*(G-mu_G)**2/sigma_G**2 + np.log(1.0/(np.sqrt(2*np.pi)*sigma_L_1))-0.5*(L_1 - mu_L_1)**2/sigma_L_1**2 + np.log(1.0/(np.sqrt(2*np.pi)*sigma_theta_0_1))-0.5*(theta_0_1 - mu_theta_0_1)**2/sigma_theta_0_1**2 + np.log(1.0/(np.sqrt(2*np.pi)*sigma_phi))-0.5*(phi - mu_phi)**2/sigma_phi**2 + np.log(1.0/(np.sqrt(2*np.pi)*sigma_L_2))-0.5*(L_2 - mu_L_2)**2/sigma_L_2**2 + np.log(1.0/(np.sqrt(2*np.pi)*sigma_theta_0_2))-0.5*(theta_0_2 - mu_theta_0_2)**2/sigma_theta_0_2**2
    
    return prior


def log_probability(eta, t, x, yerr):
    
    lp = log_prior_normal(eta)
    
    if not np.isfinite(lp):
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
eta_true = np.array([2, np.pi/4, 1, np.pi/4, 10, 1])
x_true = model_collapsed(eta_true, time)

# okay now the eta array will have the top level always have the same shape:
# L, eta_0, G, phi = eta
# lets just make sure t is the same for all
# Here, there's two different pendulii
eta_o = np.array([3, np.pi/4, 4, np.pi/4, 10, 1])
offset_position = model_collapsed(eta_o, time)

double_time = np.repeat(time, 2)
plt.clf()
for i in range(2):
    plt.plot(double_time, x_true.flatten(), color = 'black')
    plt.plot(double_time, offset_position.flatten(), color = 'green')
color_list = ['#F0A202','#F18805','#D95D39','#90B494','#7B9E89']
plt.xlabel('time')
plt.ylabel('x position')
plt.show()



start = np.array([1.5, np.pi/4, 1.5, np.pi/4, 10, 1])

pos = start + np.array([2e-1,1e-1,2e-1,1e-1,1e0,1e0]) * np.random.randn(100, 6)


nwalkers, ndim = pos.shape


# Generate some synthetic data that you know the solution to 

run = False

filename = "MCMC_chains/all_the_levels_flattened.h5"
if run:
# Optionally save chain
# Set up the backend
# Don't forget to clear it in case the file already exists
    true = eta_true
    y = model_collapsed(eta_true, time)
    yerr = 0.2 * np.ones(np.shape(y))

    print(np.shape(y))
    
    plt.clf()
    plt.errorbar(double_time, y, yerr = yerr)
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
fig, axes = plt.subplots(6, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["L_1", "theta_0_1", 
            "L_2", "theta_0_2", "G", "phi"]
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


STOP
# So for each of these we're just grabbing the g values?
print(flat_samples[:,0])
G = flat_samples[:,4]

plt.clf()
plt.hist(G, bins = 20, alpha = 0.5, color = 'green')
plt.axvline(x = eta_true[4])
plt.xlabel('G sampling')
plt.show()   


plt.clf()
fig = corner.corner(
    flat_samples, truths=eta_true,
    labels=[r"$L_1$",
        r"$theta_0_1$",
        r"$L_2$",
        r"$theta_0_2$",
        r"$G$",
        r"$phi$"],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12})
plt.show()
