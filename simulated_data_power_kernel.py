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
    'text.usetex': False  # disable LaTeX rendering
})

# big dataset
np.random.seed(42) # Hitchhiker's Guide to the Galaxy!
num_events = 1000  # large number of events
times = np.cumsum(np.random.exponential(scale=10, size=num_events))  # event times in seconds

# Simulate prices with trends and noise
prices = 100 + np.cumsum(np.random.normal(scale=0.01, size=num_events))  # simulated prices (smaller scale for HFT)

# brownian  model for stock prices !

# simulate buy/sell events
events = []
for i in range(num_events):
    if i == 0:
        events.append(np.random.choice(['Buy', 'Sell']))  # First event is random
    else:
        if events[-1] == 'Buy':
            events.append('Buy' if np.random.rand() < 0.7 else 'Sell')  
        else:
            events.append('Sell' if np.random.rand() < 0.7 else 'Buy')  

# create DataFrame
df = pd.DataFrame({
    'time': times,
    'price': prices,
    'event': events
})
df['time'] = pd.to_datetime(df['time'], unit='ms', origin='2025-03-14 19:55:00')

# preprocess the data
start_time = df['time'].min()
df['milliseconds'] = (df['time'] - start_time).dt.total_seconds() * 1000  # Convert to milliseconds

# separate buy and sell events
buy_times = df[df['event'] == 'Buy']['milliseconds'].values
sell_times = df[df['event'] == 'Sell']['milliseconds'].values

# define intensity functions and kernels
def power_law_kernel(t, alpha, beta, gamma):
    return alpha / (t + gamma) ** beta

def intensity_power_law(t, events, mu, alpha_self, alpha_cross, beta_self, beta_cross, gamma, other_events):
    intensity_value = mu
    for event_time in events[events <= t]:
        intensity_value += power_law_kernel(t - event_time, alpha_self, beta_self, gamma)
    for event_time in other_events[other_events <= t]:
        intensity_value += power_law_kernel(t - event_time, alpha_cross, beta_cross, gamma)
    return intensity_value

# fit the model
def negative_log_likelihood(params, buy_times, sell_times):
    mu, alpha_self_buy, alpha_self_sell, alpha_cross_buy_to_sell, alpha_cross_sell_to_buy, beta_self_buy, beta_self_sell, beta_cross_buy_to_sell, beta_cross_sell_to_buy, gamma = params
    total_loss = 0

    # calculate intensity for buy 
    for t in buy_times:
        intensity = intensity_power_law(t, buy_times, mu, alpha_self_buy, alpha_cross_buy_to_sell, beta_self_buy, beta_cross_buy_to_sell, gamma, sell_times)
        total_loss -= np.log(intensity)

    # calculate intensity for sell 
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

# guess for parameters
initial_params = [0.1, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]  # Adjusted initial guesses

# minimize negative log-likelihood w bounds using L-BFGS-B
result = minimize(negative_log_likelihood, initial_params, args=(buy_times, sell_times),
                  method='L-BFGS-B', bounds=[(0, 1), (0, 10), (0, 10), (0, 10), (0, 10), (0.1, 10), (0.1, 10), (0.1, 10), (0.1, 10), (0.1, 10)])

# convergence
if not result.success:
    raise ValueError("Optimization failed: " + result.message)

# estimated parameters
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

# calculate intensities using the fitted model
times = np.linspace(0, max(max(buy_times), max(sell_times)), 1000)  # Time points in milliseconds

# calculate intensities for buy and sell orders
lambda1_pl = [intensity_power_law(t, buy_times, mu, alpha_self_buy, alpha_cross_buy_to_sell, beta_self_buy, beta_cross_buy_to_sell, gamma, sell_times) for t in times]
lambda2_pl = [intensity_power_law(t, sell_times, mu, alpha_self_sell, alpha_cross_sell_to_buy, beta_self_sell, beta_cross_sell_to_buy, gamma, buy_times) for t in times]

# visualize results
plt.figure(figsize=(12, 6))
plt.plot(times, lambda1_pl, label='Buy Intensity (Power-Law Kernel)', color='blue')
plt.plot(times, lambda2_pl, label='Sell Intensity (Power-Law Kernel)', color='orange')
plt.xlabel('Time (milliseconds)')
plt.ylabel('Intensity')
plt.title('Intensity Functions for Buy and Sell Orders (Power-Law Kernel)')
plt.legend()
plt.grid()
plt.show()