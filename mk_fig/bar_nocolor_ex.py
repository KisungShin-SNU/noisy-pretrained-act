"""
Not used
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['hatch.linewidth'] = 1.8

labels = ['A', 'B', 'C', 'D', 'E', 'F']
values = [12, 9, 14, 7, 11, 10]
patterns = ['//', '\\\\', 'xx', '++', 'oo', '..']

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    labels,
    values,
    color='white',
    edgecolor='black',
    linewidth=1.2
)

for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)

ax.set_title('실전용 hatch bar 예시')
ax.set_ylabel('Score')

plt.tight_layout()
#plt.show()
plt.savefig('bar_nocolor_ex.png')