"""
This implements a model of mesolimbic dopamine cell activity during monkey
conditioning as found in `Montague, Dayan, and Sejnowski (1996) in PsyNeuLink

Reference:
Montague, P. R., Dayan, P., & Sejnowski, T. J. (1996).
A framework for mesencephalic dopamine systems based on predictive Hebbian learning.
The Journal of Neuroscience, 16(5), 1936–1947.
http://www.jneurosci.org/content/jneuro/16/5/1936.full.pdf
"""
import numpy as np
import psyneulink as pnl
import matplotlib.pyplot as plt
import seaborn as sns

from model import get_model
from mpl_toolkits import mplot3d

np.random.seed(0)
sns.set(style='white', context='talk', palette="colorblind")


def make_stimuli(
        n_trials, no_reward_trials,
        stimulus_onset_time=41,
        reward_delivery_time=54,
):
    """generate stimuli for Montague et al. 1996

    Parameters
    ----------
    n_trials : int
        number of trials
    no_reward_trials : list, 1d array, set
        trials without reward
    stimulus_onset_time : int
        stimulus onset time
    reward_delivery_time : int
        the time point when the reward is delivered

    Returns
    -------
    2d array, 2d array
        input, reward signal

    """
    # make samples
    samples = np.zeros((n_trials, n_time_steps))
    samples[:, stimulus_onset_time:] = 1
    # make targets
    targets = np.zeros((n_trials, n_time_steps))
    targets[:, reward_delivery_time] = 1
    # reward withheld trials
    for no_reward_trial in no_reward_trials:
        targets[no_reward_trial, :] = 0
    return samples, targets


"""
This creates the plot for figure 5A in the Montague paper. Figure 5A is
a 'plot of ∂(t) over time for three trials during training (1, 30, and 50).'
"""
# Create Stimulus Dictionary
n_time_steps = 60
n_trials = 120
no_reward_trials = {14, 29, 44, 59, 74, 89}
samples, targets = make_stimuli(n_trials, no_reward_trials)

# Create the model
comp, nodes = get_model(n_time_steps=n_time_steps)
[sample_mechanism, prediction_error_mechanism, target_mechanism] = nodes
# Run Composition
inputs = {sample_mechanism: samples, target_mechanism: targets}
comp.run(inputs=inputs)

# Get Delta Values from Log
delta_vals = prediction_error_mechanism.log.nparray_dictionary()[
    comp.name][pnl.VALUE]

# Plot Delta Values form trials 1, 30, and 50
t_start = 35
fig = plt.figure(figsize=(8, 5))
plt.plot(delta_vals[0][0], "-o", label="Trial 1")
plt.plot(delta_vals[29][0], "-s", label="Trial 30")
plt.plot(delta_vals[49][0], "-o", label="Trial 50")
plt.title("Montague et. al. (1996) -- Figure 5A")
plt.xlabel("Timestep")
plt.ylabel(r"$\delta$")
plt.legend(frameon=False)
plt.xlim([t_start, n_time_steps])
plt.xticks()
fig.tight_layout()
sns.despine()
fig.savefig('figs/fig5a.png', dpi=100)

"""
This creates the plot for figure 5B in the Montague paper. Figure 5B shows
the 'entire time course of model responses (trials 1-150).' The setup is
the same as in Figure 5A, except that training begins at trial 10.
"""
# Create Stimulus Dictionary
n_time_steps = 60
n_trials = 120
no_reward_trials = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 29, 44, 59, 74, 89, 104, 119
}
samples, targets = make_stimuli(n_trials, no_reward_trials)

# Create the model
comp, nodes = get_model(n_time_steps=n_time_steps)
[sample_mechanism, prediction_error_mechanism, target_mechanism] = nodes

# Run Composition
inputs = {sample_mechanism: samples, target_mechanism: targets}
comp.run(inputs=inputs)

# Get Delta Values from Log
delta_vals = np.squeeze(
    prediction_error_mechanism.log.nparray_dictionary()[comp.name][pnl.VALUE]
)

t_start = 40
x_vals, y_vals = np.meshgrid(
    np.arange(n_trials), np.arange(t_start, n_time_steps))
d_vals = delta_vals[:, t_start:n_time_steps].T

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(25, 80)
ax.plot_surface(x_vals, y_vals, d_vals, linewidth=.5)
ax.invert_xaxis()
ax.set_xlabel("\nTrial")
ax.set_ylabel("\nTimestep")
ax.set_zlabel('\n'+r"$\delta$")
ax.set_title("Montague et. al. (1996) -- Figure 5B")
fig.savefig('figs/fig5b.png', dpi=100)


"""
This creates the plot for Figure 5C in the Montague paper. Figure 5C shows
'extinction of response to the sensory cue.' The setup is the same as
Figure 5A, except that reward delivery stops at trial 70
"""
# Create Stimulus Dictionary
n_trials = 150
reward_removal_onset = 70
no_reward_trials = np.arange(reward_removal_onset, n_trials)
samples, targets = make_stimuli(n_trials, no_reward_trials)

# Create the model
comp, nodes = get_model(n_time_steps=n_time_steps)
[sample_mechanism, prediction_error_mechanism, target_mechanism] = nodes

# Run Composition
inputs = {sample_mechanism: samples, target_mechanism: targets}
comp.run(inputs=inputs)

# Get Delta Values from Log
delta_vals = np.squeeze(
    prediction_error_mechanism.log.nparray_dictionary()[comp.name][pnl.VALUE]
)

t_start = 40
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
x_vals, y_vals = np.meshgrid(
    np.arange(n_trials), np.arange(t_start, n_time_steps))
d_vals = delta_vals[:, t_start:n_time_steps].T
ax.plot_surface(x_vals, y_vals, d_vals, linewidth=.5)
ax.view_init(25, 275)
ax.invert_yaxis()
ax.set_xlabel("\nTrial")
ax.set_ylabel("\nTimestep")
ax.set_zlabel('\n'+r"$\delta$")
ax.set_title("Montague et. al. (1996) -- Figure 5C")
fig.savefig('figs/fig5c.png', dpi=100)
