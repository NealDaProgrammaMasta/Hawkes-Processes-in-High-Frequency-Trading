# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scienceplots

# scienceplots
plt.style.use('science')
plt.rcParams.update({
    'axes.prop_cycle': plt.cycler(color=['#1F77B4', '#FF7F0E']),    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#cccccc',
    'lines.linewidth': 2,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'text.usetex': False  
})

# data from csv
df = pd.read_csv('amd_tsla_trades.csv')  

# interpret columns
# convert 'timestamp' to datetime
df['time'] = pd.to_datetime(df['timestamp'])

prices = df['price'].values  
volumes = df['volume'].values  

# extract 'event' to separate buy and sell events
buy_times = df[df['event'] == 'Buy']['time'].values  
sell_times = df[df['event'] == 'Sell']['time'].values  

# preprocess data: 
start_time = df['time'].min()  # first event timestamp
df['milliseconds'] = (df['time'] - start_time).dt.total_seconds() * 1000  # convert to milliseconds

# separate buy and sell events in milliseconds
buy_times = df[df['event'] == 'Buy']['milliseconds'].values
sell_times = df[df['event'] == 'Sell']['milliseconds'].values

# define intensity functions and kernels (power-law kernel)
def power_law_kernel(t, alpha, beta, gamma):
    return alpha / (t + gamma) ** beta

def intensity_power_law(t, events, mu, alpha_self, alpha_cross, beta_self, beta_cross, gamma, other_events):
    intensity_value = mu
    for event_time in events[events <= t]:
        intensity_value += power_law_kernel(t - event_time, alpha_self, beta_self, gamma)
    for event_time in other_events[other_events <= t]:
        intensity_value += power_law_kernel(t - event_time, alpha_cross, beta_cross, gamma)
    return intensity_value

# fit the model (power-law kernel)
def negative_log_likelihood(params, buy_times, sell_times):
    mu, alpha_self_buy, alpha_self_sell, alpha_cross_buy_to_sell, alpha_cross_sell_to_buy, beta_self_buy, beta_self_sell, beta_cross_buy_to_sell, beta_cross_sell_to_buy, gamma = params
    total_loss = 0

    # calculate intensity for buy events
    for t in buy_times:
        intensity = intensity_power_law(t, buy_times, mu, alpha_self_buy, alpha_cross_buy_to_sell, beta_self_buy, beta_cross_buy_to_sell, gamma, sell_times)
        total_loss -= np.log(intensity)

    # calculate intensity for sell events
    for t in sell_times:
        intensity = intensity_power_law(t, sell_times, mu, alpha_self_sell, alpha_cross_sell_to_buy, beta_self_sell, beta_cross_sell_to_buy, gamma, buy_times)
        total_loss -= np.log(intensity)

    # add integral term
    end_time = max(max(buy_times), max(sell_times))
    times = np.linspace(0, end_time, 1000)
    integral = np.sum([intensity_power_law(t, buy_times, mu, alpha_self_buy, alpha_cross_buy_to_sell, beta_self_buy, beta_cross_buy_to_sell, gamma, sell_times) +
                       intensity_power_law(t, sell_times, mu, alpha_self_sell, alpha_cross_sell_to_buy, beta_self_sell, beta_cross_sell_to_buy, gamma, buy_times)
                       for t in times]) * (end_time / 1000)
    total_loss += integral

    return total_loss

# initial guess for parameters
initial_params = [0.6, 1.9, 0.2, 0.3, 0.4, 1.8, 1.6, 1.6, 1.2, 1.99] 

# minimize the negative log-likelihood with bounds and a better algorithm
result = minimize(negative_log_likelihood, initial_params, args=(buy_times, sell_times),
                  method='L-BFGS-B', bounds=[(0, 1), (0, 10), (0, 10), (0, 10), (0, 10), (0.1, 10), (0.1, 10), (0.1, 10), (0.1, 10), (0.1, 10)])

# check for convergence
if not result.success:
    raise ValueError("Optimization failed: " + result.message)

# get estimated parameters
mu, alpha_self_buy, alpha_self_sell, alpha_cross_buy_to_sell, alpha_cross_sell_to_buy, beta_self_buy, beta_self_sell, beta_cross_buy_to_sell, beta_cross_sell_to_buy, gamma = result.x

print("Base intensity (μ):", mu)
print("Self-excitation for Buy (α_self_buy):", alpha_self_buy)
print("Self-excitation for Sell (α_self_sell):", alpha_self_sell)
print("Cross-excitation Buy to Sell (α_cross_buy_to_sell):", alpha_cross_buy_to_sell)
print("Cross-excitation Sell to Buy (α_cross_sell_to_buy):", alpha_cross_sell_to_buy)
print("Self-decay rate for Buy (β_self_buy):", beta_self_buy)
print("Self-decay rate for Sell (β_self_sell):", beta_self_sell)
print("Cross-decay rate Buy to Sell (β_cross_buy_to_sell):", beta_cross_buy_to_sell)
print("Cross-decay rate Sell to Buy (β_cross_sell_to_buy):", beta_cross_sell_to_buy)
print("Power-law constant (γ):", gamma)

# calculate intensities using fitted model
times = np.linspace(0, max(max(buy_times), max(sell_times)), 1000)  # Time points in milliseconds

# calculate intensities for buy and sell orders (power-law kernel)
lambda1_pl = [intensity_power_law(t, buy_times, mu, alpha_self_buy, alpha_cross_buy_to_sell, beta_self_buy, beta_cross_buy_to_sell, gamma, sell_times) for t in times]
lambda2_pl = [intensity_power_law(t, sell_times, mu, alpha_self_sell, alpha_cross_sell_to_buy, beta_self_sell, beta_cross_sell_to_buy, gamma, buy_times) for t in times]

# Step 6: visualize results (power-law kernel)
plt.figure(figsize=(12, 6))
plt.plot(times, lambda1_pl, label='Buy Intensity (Power-Law Kernel)', color='blue')
plt.plot(times, lambda2_pl, label='Sell Intensity (Power-Law Kernel)', color='orange')
plt.xlabel('Time (milliseconds)')
plt.ylabel('Intensity')
plt.title('Figure X: Intensity Functions Against Time for AMD and TSLA Fitted Using Power-Law Kernel')
plt.legend()
plt.grid()
plt.show()