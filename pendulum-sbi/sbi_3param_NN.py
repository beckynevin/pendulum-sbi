# Trying this with three params now
# also adding in noise
# but write so that it includes momenta as well

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

def simulator_old(theta, t = np.linspace(0, 10, 200), noise=[0.1,0.1,0.1]):
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

def simulator(theta, t = np.linspace(0, 10, 100), noise=[0.1,0.0,0.0]):
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





# Time to get into SBI
# Taking some code from Sree to run the NN version of this:

'''
# The Mackelab example is with a CNN
class SummaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 2D convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # Maxpool layer that reduces 32x32 image to 4x4
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layer taking as input the 6 flattened output arrays from the maxpooling layer
        self.fc = nn.Linear(in_features=6 * 1 * 100, out_features=2)

    def forward(self, x):
        x = x.view(-1, 1, len(time), 2)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 6 * 1 * 100)
        x = F.relu(self.fc(x))
        return x


embedding_net = SummaryNet()
'''

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim_1)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="relu")
        self.layer_3 = nn.Linear(hidden_dim_2, output_dim)
       
    def forward(self, x):
        print(np.shape(x))
        x = x.view(1, -1)
        print(np.shape(x))
        #x = self.fc1(x)
        x = torch.nn.functional.relu(self.layer_1(x))
        #x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.layer_2(x))
        x = torch.nn.functional.sigmoid(self.layer_3(x))

        return x

x = simulator(theta_o, noise = noiz)
print(x)
print(np.shape(x))
print(np.shape(x)[1])



input_dim = np.shape(x)[1] # dimension of the parameters
hidden_dim_1 = 600
hidden_dim_2 = 100



output_dim = 2#len(x) # simulator output dimension, lets start with just x


model = nn.Sequential(nn.Linear(input_dim, hidden_dim_1),
                      nn.ReLU(),
                      nn.Linear(hidden_dim_1, output_dim),
                      nn.Sigmoid())
print(model)
embedding_net = model


#embedding_net = NeuralNetwork(input_dim, hidden_dim_1, hidden_dim_2, output_dim)
print(embedding_net)





#x = simulator(theta.numpy(), noise = noiz)

# make a SBI-wrapper on the simulator object for compatibility
simulator_wrapper, prior = prepare_for_sbi(simulator, prior)

# instantiate the neural density estimator
neural_posterior = utils.posterior_nn(
    model="maf", embedding_net=embedding_net, hidden_features = 10, num_transforms=2
)

# setup the inference procedure with the SNPE-C procedure
inference = SNPE(prior=prior, density_estimator=neural_posterior)


# run the inference procedure on one round and 10000 simulated data points
theta, x = simulate_for_sbi(simulator_wrapper, prior, num_simulations=20000)# was 20k

density_estimator = inference.append_simulations(theta, x).train()
posterior = inference.build_posterior(density_estimator)

# generate posterior samples
#theta_o = np.array([10, 5, np.pi/4])

true_parameter = torch.tensor([10.0, 4.0, np.pi/4])
x_observed = simulator(true_parameter)
samples = posterior.set_default_x(x_observed).sample((50000,))

# create the figure
fig, ax = analysis.pairplot(
    samples,
    points=true_parameter,
    labels=["g", "L", "theta_init"],
    #limits=[[0.0, 5.0], [2.0,5.0], [9.0,12.0]],
    points_colors="r",
    points_offdiag={"markersize": 6},
    figsize=(5, 5),
)
plt.show()
STOP




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
print('theta shape', np.shape(theta), 'x shape', np.shape(x))
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
x_o_t, x_o_x, x_o_y, x_o_mom_x, x_o_mom_y = simulator(theta_o)
plt.plot(x_o_t, x_o_x[0,:], "k", zorder=1, label="truth")
theta_p = posterior.sample((n_samples,), x=x_o)
ind_10_highest = np.argsort(np.array(posterior.log_prob(theta=theta_p, x=x_o)))[-10:]
theta_p_considered = theta_p[ind_10_highest, :]
x_t, x_x, x_y, x_mom_x, x_mom_y = simulator(theta_p_considered.numpy())
for n in range(10):
    if n == 0:
        plt.plot(x_t, x_x[n,:], "green", zorder=0, label = 'Best draws')
    else:
        plt.plot(x_t, x_x[n,:], "green", zorder=0)

ind_10_lowest = np.argsort(np.array(posterior.log_prob(theta=theta_p, x=x_o)))[0:10]
theta_p_considered = theta_p[ind_10_lowest, :]
x_t, x_x, x_y, x_mom_x, y_mom_y = simulator(theta_p_considered.numpy())
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

print('theta shape', np.shape(theta), 'x shape', np.shape(x))

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


x_o_t, x_o_x, x_o_y, mom_o_x, mom_o_y = simulator(theta_o)
plt.plot(x_o_t, x_o_x, "k", zorder=1, label="truth")

theta_p = posterior.sample((n_samples,), x=x_o)
x_t, x_x, x_y, mom_x, mom_y = simulator(theta_p.numpy())
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