import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 글꼴 설정: Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15

plt.rcParams.update({
    'font.size': 12,        # 기본 글자 크기
    'axes.titlesize': 16,   # 제목
    'axes.labelsize': 14,   # x, y 라벨
    'xtick.labelsize': 12,  # x축 눈금
    'ytick.labelsize': 12,  # y축 눈금
    'legend.fontsize': 12   # 범례
})

x = np.arange(6)

labels = [
    'Baseline', 
    'Image',
    'Action (small)',
    'Action',
    'Image & Action (small)',
    'Image & Action',
]

values = [26, 32, 42, 40, 48, 40]
# baseline 26
# noisy img 32
# small noisy act 42
# noisy act 40
# small noisy act img 48
# noisy act img 40
plt.figure(figsize=(10, 5))

# 앞 4개 bar: 단색
solid_colors = ["#C5E0B4", "#8FBBD9", "#F7D4D4", "#DE5253"]

for i in range(4):
    plt.barh(i, values[i], color=solid_colors[i], height=0.8)

# 뒤 2개 bar: 그라데이션
grad = np.linspace(0, 1, 256).reshape(256, 1)

# 5번째 bar용 colormap
cmap1 = LinearSegmentedColormap.from_list(
    "grad1",
    ["#F7D4D4", "#8FBBD9"]
)

# 6번째 bar용 colormap
cmap2 = LinearSegmentedColormap.from_list(
    "grad2",
    ["#DE5253", "#8FBBD9"]
)

# "#DBD1CE", 

gradient_settings = [
    (4, values[4], cmap1),
    (5, values[5], cmap2),
]

for y_center, value, cmap in gradient_settings:
    height = 0.8
    y0 = y_center - height / 2
    y1 = y_center + height / 2

    plt.imshow(
        grad,
        extent=[0, value, y0, y1],
        cmap=cmap,
        aspect='auto'
    )

# 값 텍스트 표시
for i, value in enumerate(values):
    plt.text(value + 0.5, i, str(value), va='center')

plt.yticks(x, labels)
plt.xlabel('Task success rate (%)')
plt.ylim(-0.6, 5.6)
plt.xlim(0, 52)
plt.tight_layout()
plt.gca().invert_yaxis()

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig('result_1.png', dpi=300, bbox_inches='tight')
#plt.show()