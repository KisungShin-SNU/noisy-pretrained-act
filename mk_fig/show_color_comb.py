import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =========================
# 1. Revised color definitions
# =========================
colors = {
    "A": "#1F77B4",  # blue
    "B": "#D62728",  # red
    "C": "#2CA02C",  # green
    "D": "#E377C2",  # pink
    "E": "#9467BD",  # purple
    "F": "#8C564B",  # brown
}

colors = {
    "A": "#8FBBD9",  # blue
    "B": "#F7D4D4",  # red
    "B2": "#DE5253",  # red 2
    "C": "#C5E0B4",  # green
    "D": "#E377C2",  # pink
    "E": "#9467BD",  # purple
    "F": "#8C564B",  # brown
}

alpha_abc = 1.0
alpha_def = 0.45

# =========================
# 2. Show palette
# =========================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.set_title("Revised 6-color palette", fontsize=14)

labels = list(colors.keys())
for i, key in enumerate(labels):
    ax.add_patch(Rectangle((i, 0), 1, 1, color=colors[key]))
    ax.text(i + 0.5, -0.18, f"{key}\n{colors[key]}", ha="center", va="top", fontsize=10)

ax.set_xlim(0, 6)
ax.set_ylim(-0.35, 1)
ax.axis("off")

# =========================
# 3. A/B/C: clearly separated categories
# =========================
ax = axes[1]
ax.set_title("A, B, C: distinct categories", fontsize=14)

x = np.arange(3)
heights = [0.85, 1.10, 0.95]

ax.bar(x[0], heights[0], color=colors["A"], alpha=alpha_abc, label="A")
ax.bar(x[1], heights[1], color=colors["B"], alpha=alpha_abc, label="B")
ax.bar(x[2], heights[2], color=colors["C"], alpha=alpha_abc, label="C")

ax.set_xticks(x)
ax.set_xticklabels(["A", "B", "C"])
ax.set_ylabel("Value")
ax.legend(frameon=False)

# =========================
# 4. D/E/F: overlapping distributions
# =========================
ax = axes[2]
ax.set_title(f"D, E, F: overlapping distributions (alpha={alpha_def})", fontsize=14)

np.random.seed(42)
d = np.random.normal(loc=-0.6, scale=1.0, size=1200)
e = np.random.normal(loc= 0.1, scale=1.0, size=1200)
f = np.random.normal(loc= 0.8, scale=1.0, size=1200)

bins = np.linspace(-4, 4, 40)

ax.hist(d, bins=bins, density=True, color=colors["D"], alpha=alpha_def, label="D")
ax.hist(e, bins=bins, density=True, color=colors["E"], alpha=alpha_def, label="E")
ax.hist(f, bins=bins, density=True, color=colors["F"], alpha=alpha_def, label="F")

ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend(frameon=False)

plt.tight_layout()
#plt.show()
plt.savefig('show_color_comb.png')