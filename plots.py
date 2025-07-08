import pandas as pd
from matplotlib import pyplot as plt 
import seaborn as sns
import numpy as np

df = pd.read_csv('./output/metrics_log.csv')

cols = ['Accuracy', 'F1', 'Precision', 'Recall', 'Loss']

for col in cols:
    g = sns.relplot(data=df, kind="line", x="Attention", y=col, hue="Model", height=5, aspect=2, dashes=False,markers=True, marker='o', palette="deep")
    g.set(ylim=(0, 1)) 
    g.set(yticks=np.arange(0.0, 1.01, 0.1))
    g.fig.suptitle(f"Comparison of {col} Across Different Models and Attention Mechanisms", fontsize=16)
    g.fig.subplots_adjust(top=0.9)
    g.set_xlabels("Types of attention")
    g.set_ylabels(f"{col}")
    plt.savefig(f"plots/{col}.png", dpi=300)
    plt.show()