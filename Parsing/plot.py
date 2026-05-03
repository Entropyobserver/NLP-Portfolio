import matplotlib.pyplot as plt
import numpy as np

# Dataset values
swedish = [0.04, 0.18, 0.40, 0.75, 1.58, 1.32, 0.44, 2.19, 0.97, 1.93, 7.51, 5.05, 7.81, 12.07, 12.99, 13.74, 16.24, 15.45, 18.79, 18.66, 17.03, 19.27, 21.25, 20.85, 22.34, 21.86, 25.46, 25.86, 24.89, 25.20]

swedish_chinese = [0.57, 1.19, 4.08, 7.29, 11.41, 8.60, 13.17, 16.94, 17.91, 19.23, 19.45, 21.51, 24.06, 25.68, 27.44, 27.30, 27.83, 29.37, 30.29, 31.39, 29.10, 30.68, 32.13, 34.02, 32.09, 33.49, 33.32, 35.69, 33.98, 35.56]

swedish_norwegian = [0.61, 0.92, 5.71, 14.84, 18.44, 26.51, 25.55, 28.09, 30.95, 35.43, 36.35, 40.96, 39.29, 41.53, 41.48, 44.25, 45.48, 46.09, 46.09, 47.67, 48.42, 47.85, 46.66, 47.54, 50.53, 49.39, 47.32, 49.87, 49.74, 49.56]

# Create epochs (x-axis values)
epochs = range(1, 31)

# Create figure and axis
plt.figure(figsize=(12, 7))

# Plot the lines
plt.plot(epochs, swedish, marker='o', linestyle='-', color='#8884d8', linewidth=2, label='Swedish')
plt.plot(epochs, swedish_norwegian, marker='s', linestyle='-', color='#82ca9d', linewidth=2, label='Swedish + Norwegian')
plt.plot(epochs, swedish_chinese, marker='^', linestyle='-', color='#ff7300', linewidth=2, label='Swedish + Chinese')

# Add title and labels
plt.title('Epoch Progress Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Value', fontsize=12)

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend(fontsize=12)

# Set y-axis limits to start from 0
plt.ylim(bottom=0)

# Add annotations for final values
plt.annotate(f'{swedish[-1]:.2f}', xy=(epochs[-1], swedish[-1]), xytext=(epochs[-1]+0.5, swedish[-1]), 
             fontsize=10, fontweight='bold')
plt.annotate(f'{swedish_norwegian[-1]:.2f}', xy=(epochs[-1], swedish_norwegian[-1]), xytext=(epochs[-1]+0.5, swedish_norwegian[-1]), 
             fontsize=10, fontweight='bold')
plt.annotate(f'{swedish_chinese[-1]:.2f}', xy=(epochs[-1], swedish_chinese[-1]), xytext=(epochs[-1]+0.5, swedish_chinese[-1]), 
             fontsize=10, fontweight='bold')

# Improve layout
plt.tight_layout()

# Save the plot (optional)
plt.savefig('epoch_comparison.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()