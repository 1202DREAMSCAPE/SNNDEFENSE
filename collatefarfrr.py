import matplotlib.pyplot as plt
import numpy as np

# Define your FAR and FRR data for each dataset (5 runs each)
metrics = {
    "CEDAR": {
        "FAR": [46.81, 47.94, 49.93, 45.11, 46.24],
        "FRR": [4.26, 4.26, 4.82, 5.39, 5.39]
    },
    "BHSig260_Bengali": {
        "FAR": [36.10, 43.77, 39.94, 42.01, 43.52],
        "FRR": [13.02, 9.94, 14.03, 10.31, 9.69]
    },
    "BHSig260_Hindi": {
        "FAR": [32.79, 35.38, 38.61, 37.41, 35.63],
        "FRR": [12.66, 11.46, 7.88, 10.06, 11.54]
    }
}

runs = ['Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5']
width = 0.35
x = np.arange(len(runs))

# Create one plot per dataset
for dataset, values in metrics.items():
    far = values["FAR"]
    frr = values["FRR"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, far, width, label='FAR (%)', color='skyblue')
    bars2 = ax.bar(x + width/2, frr, width, label='FRR (%)', color='salmon')

    # Annotate bars
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
    ax.set_title(f'FAR and FRR Across 5 Runs ({dataset})')
    ax.set_xticks(x)
    ax.set_xticklabels(runs)
    ax.set_ylim(0, 110)
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{dataset}_FAR_FRR_BarPlot.png")
    plt.close()
