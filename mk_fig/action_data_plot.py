import h5py
import matplotlib.pyplot as plt

# 글꼴 설정: Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

plt.rcParams.update({
    'font.size': 12,        # 기본 글자 크기
    'axes.titlesize': 16,   # 제목
    'axes.labelsize': 14,   # x, y 라벨
    'xtick.labelsize': 12,  # x축 눈금
    'ytick.labelsize': 12,  # y축 눈금
    'legend.fontsize': 12   # 범례
})

file_a = "/media/bi_admin/4tb_hdd/data/noisy-pretrained-act/bef_260330/sim_transfer_cube_human/episode_0.hdf5"
file_b = "/media/bi_admin/4tb_hdd/data/noisy-pretrained-act/sim_transfer_cube_small_noisy_scripted/episode_0.hdf5"
file_c = "/media/bi_admin/4tb_hdd/data/noisy-pretrained-act/bef_260330/sim_transfer_cube_noisy_scripted/episode_0.hdf5"

with h5py.File(file_a, "r") as f:
    A = f["action"][:, 7]

with h5py.File(file_b, "r") as f:
    B = f["action"][:, 7]

with h5py.File(file_c, "r") as f:
    C = f["action"][:, 7]

n = min(len(A), len(B))

plt.figure(figsize=(10, 4))
plt.plot(A[:n], label="Original", color='#2CA02C')# linestyle='-') #baseline
plt.plot(B[:n], label="Random noise action (small)", color='#F7D4D4')# linestyle='--') #small action
plt.plot(C[:n], label="Random noise action", color='#DE5253')# linestyle='-.') #action
plt.xlabel("Frame (0.02 sec)")
plt.ylabel("Joint position (rad)")
plt.title("Joint position changes in waist of right arm action data")
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig("action_first_element_comparison.png", dpi=200, bbox_inches="tight")