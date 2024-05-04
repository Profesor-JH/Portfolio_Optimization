import numpy as np
import matplotlib.pyplot as plt

# Sample data
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']
allocation = np.array([0.25, 0.2, 0.15, 0.3, 0.1])  # Example allocation weights

# Plotting
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Compute evenly spaced angles for radar chart
theta = np.linspace(0, 2*np.pi, len(tickers), endpoint=False).tolist()
theta += theta[:1]  # To connect the plot

# Plot the allocation values
allocation = np.concatenate((allocation, [allocation[0]]))  # To connect the plot
ax.fill(theta, allocation, 'b', alpha=0.1)  # Fill the area

# Plot the allocation values as points
ax.plot(theta, allocation, linestyle='-', marker='o')

# Add labels
ax.set_ylim(0, 0.35)  # Adjust the limit for better visualization
ax.set_xticks(theta[:-1])
ax.set_xticklabels(tickers)

# Show the radar plot
plt.title('Optimized Portfolio Allocation (Radar Plot)')
plt.show()
