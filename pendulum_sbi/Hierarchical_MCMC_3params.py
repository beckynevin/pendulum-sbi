# Adding a hierarchical sampling to the MCMC chains
# I'm thinking that there will probably eventually be an option
# to not re-run the initial simple first level inference
# chains and instead to simply sample from the saved chains
# I'm using Hogg 2016 as a reference for how to do this

# To do:
# 1. Create N different pendulums
# 2. Sample from these pendulums using the basic MCMC
# 3. Re-sample using eqn 9 from Hogg+2016

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


def simulator(theta, t = np.linspace(0, 10, 100), noise=[0.5,0.0,0.0]):
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
    

   
    

    return x

# Now set up the likelihood, which will rely on the simulator
def log_likelihood(theta, t, y, yerr):
    model = simulator(theta, t = t)
    sigma2 = yerr**2 
    return -0.5 * np.sum((y - model) ** 2 / 2 * sigma2) #+ np.log(sigma2))

def log_prior_uniform(theta):
    g, L, theta_o = theta

    if 5.0 < g < 15.0 and 0.0 < L < 10.0 and 0 < theta_o < np.pi/2:
        return 0.0    
    return -np.inf

def log_prior_normal(theta):
    g, L, theta_0 = theta
    
    # right off the bat, disallow negative numbers:
    if g < 0.0 or L < 0.0 or theta_0 < 0.0 or theta_0 > np.pi/2:
        prior = -np.inf
        
        return prior
    mu_g, sigma_g = 10, 2
    mu_L, sigma_L = 5, 2
    mu_theta_0, sigma_theta_0 = 0.75, 0.2
    prior = np.log(1.0/(np.sqrt(2*np.pi)*sigma_g))-0.5*(g-mu_g)**2/sigma_g**2 + np.log(1.0/(np.sqrt(2*np.pi)*sigma_L))-0.5*(L - mu_L)**2/sigma_L**2 + np.log(1.0/(np.sqrt(2*np.pi)*sigma_theta_0))-0.5*(theta_0 - mu_theta_0)**2/sigma_theta_0**2
    
    return prior

def log_prior_normal_g(g):
    if g < 0.0:
        prior = -np.inf
        return prior
    mu_g, sigma_g = 10, 2
    #prior = np.log(1.0/(np.sqrt(2*np.pi)*sigma_g))-0.5*(g-mu_g)**2/sigma_g**2 + np.log(1.0/(np.sqrt(2*np.pi)*sigma_L))-0.5*(L - mu_L)**2/sigma_L**2 + np.log(1.0/(np.sqrt(2*np.pi)*sigma_theta_0))-0.5*(theta_0 - mu_theta_0)**2/sigma_theta_0**2
    
    return np.log(1.0/(np.sqrt(2*np.pi)*sigma_g))-0.5*(g-mu_g)**2/sigma_g**2




def log_probability(theta, t, y, yerr):
    
    lp = log_prior_normal(theta)
    if not np.isfinite(lp):
        return -np.inf

    p_density =  lp + log_likelihood(theta, t, y, yerr)
    '''
    if p_density < -15:
        print('input values', theta)
        print('posterior density', p_density)
        print('prior', lp)
        print('log likelihood', log_likelihood(theta, t, y, yerr))
    '''
    return p_density

# The next step is to implement SBI using the Macklab example from:
# https://github.com/mackelab/sbi/blob/main/tutorials/10_crafting_summary_statistics.ipynb

# Also could look at Miles' example for reference:
# file:///Users/rnevin/Zotero/storage/L52NCDXF/simulation-based-inference.html
# Now redo with the pendulum simulator



# TO DO: terrible notation flying around, theta_o is referring to
# parameters and to the starting theta of the pendulum

# True parameter values
theta_o = np.array([10, 5, np.pi/4])
time =  np.linspace(0, 10, 100)

true_position = simulator(theta_o, t = time)

theta_o = np.array([9, 5, np.pi/4])
offset_position = simulator(theta_o, t = time)



plt.clf()
plt.plot(time, true_position.flatten(), color = 'black')

color_list = ['#F0A202','#F18805','#D95D39','#90B494','#7B9E89']
for i, offset_gs in enumerate(np.linspace(5,15,5)):
    print(i, offset_gs)
    theta = np.array([offset_gs, 5, np.pi/4])
    offset_position = simulator(theta, t = time)
    plt.plot(time, offset_position.flatten(), color = color_list[i])
    plt.annotate(f'log L = {round(log_likelihood(theta, time, true_position, yerr = 1),2)}, g = {round(offset_gs, 2)}', 
        xy = (0.02, 0.3 - 0.05*i), xycoords = 'axes fraction', color = color_list[i])
plt.xlabel('time')
plt.ylabel('x position')
plt.show()



start = np.array([7, 5, np.pi/4])

pos = start + np.array([1e0,2e0,1e-1]) * np.random.randn(32, 3)

print('starting positiong', pos)



nwalkers, ndim = pos.shape


# Generate some synthetic data that you know the solution to 

t =  np.linspace(0, 10, 100)

# sample all of these true draws (maybe just do 10) from the prior?


mu_g, sigma_g = 10, 2
mu_L, sigma_L = 5, 2
mu_theta_0, sigma_theta_0 = 0.75, 0.2
    


g_sample = np.random.normal(loc = mu_g, scale = sigma_g, size = 3)
L_sample = np.random.normal(loc = mu_L, scale = sigma_L, size = 3)
theta_0_sample = np.random.normal(loc = mu_theta_0, scale = sigma_theta_0, size = 3)

plt.clf()
for g, L, theta_0 in zip(g_sample, L_sample, theta_0_sample):
    theta = np.array([g, L, theta_0])
    offset_position = simulator(theta, t = time)
    plt.plot(time, offset_position.flatten())
plt.show()

# Now run an MCMC for all of these:
run = False
counter = 0

for g, L, theta_0 in zip(g_sample, L_sample, theta_0_sample):
    filename = "MCMC_chains/chain_pendulum_number_"+str(counter)+".h5"
    if run:
    # Optionally save chain
    # Set up the backend
    # Don't forget to clear it in case the file already exists
        true = np.array([g, L, theta_0])
        y = simulator(true, t = t)
        yerr = 0.05 * np.ones(np.shape(y))
    
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)


        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(t, y, yerr), backend = backend
        )
        # run it
        sampler.run_mcmc(pos, 1000, progress=True);
    else:
        sampler = emcee.backends.HDFBackend(filename)

    # get the chains
    plt.clf()
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["g", "L", "theta_0"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        if run:
            ax.axhline(y = true[i])
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

    # Now how do we solve for the best fit alpha values?

    STOP

    def f(g, alpha):
        a, b = alpha
        return g**(a-1)*(1-g)**(b-1)

    def likelihood_alpha():
        return np.sum(f(g,alpha)/log_prior_normal_g(g))
    
    def log_probability(alpha, t, y, yerr):
    
        lp = log_prior_normal(theta)
        if not np.isfinite(lp):
            return -np.inf

        p_density =  lp + log_likelihood(theta, t, y, yerr)
        
        return p_density


    

    plt.clf()
    fig = corner.corner(
        flat_samples, truths=[9.8, 5, np.pi/4],
        labels=[
            r"$g$",
            r"$L$",
            r"$\theta_0$",
        ],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
    )
    plt.show()

    counter += 1

print('you successfully ran all chains')


STOP


# Get the details of the sampling:
tau = sampler.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))
print("flat log prior shape: {0}".format(log_prior_samples.shape))

STOP




tau = sampler.get_autocorr_time()
print('autocorr time', tau)

