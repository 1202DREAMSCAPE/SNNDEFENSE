import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

runs = ['Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5']
far = [0.0, 0.98, 87.94, 0.05, 81.08]
frr = [100.0, 98.58, 2.87, 99.92, 15.56]

x = np.arange(len(runs))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, far, width, label='FAR (%)', color='skyblue')
bars2 = ax.bar(x + width/2, frr, width, label='FRR (%)', color='salmon')

# Add percentage labels above bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Percentage')
ax.set_title('FAR and FRR Across 5 Runs Using Youden’s J')
ax.set_xticks(x)
ax.set_xticklabels(runs)
ax.legend()
ax.set_ylim(0, 110)

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()