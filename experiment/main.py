import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class HVDR:
    def __init__(self, l):
        self.l = l
    def __mul__(self, other):
        ret = list()
        for x in self.l:
            tmp = list()
            for y in other.l:
                tmp.append(x * y)
            ret.append(tmp)
        return ret

sigma_0 = HVDR([1, 1, 0, 0])
sigma_1 = HVDR([-1, -1, 2, 0])
sigma_2 = HVDR([-1, -1, 0, 2])
sigma_3 = HVDR([1, -1, 0, 0])
sigma = [sigma_0, sigma_1, sigma_2, sigma_3]
sigma_pow = [[None for _ in range(4)] for _ in range(4)]
for i in range(4):
    for j in range(4):
        sigma_pow[i][j] = sigma[i] * sigma[j]

polar_dict = {'H': 0, 'V': 1, 'D': 2, 'R': 3}
raw_data = np.loadtxt("data.csv", dtype = "str", delimiter = ",")
data = [[None for _ in range(4)] for _ in range(4)]
for i in range(1, len(raw_data)):
    data[polar_dict[raw_data[i][0]]][polar_dict[raw_data[i][1]]] = int(raw_data[i][2])

u = [[0 for _ in range(4)] for _ in range(4)]
for i1 in range(4):
    for j1 in range(4):
        for i2 in range(4):
            for j2 in range(4):
                u[i1][j1] += sigma_pow[i1][j1][i2][j2] * data[i2][j2]

sigma_num_0 = np.array([[1, 0], [0, 1]])
sigma_num_1 = np.array([[0, 1], [1, 0]])
sigma_num_2 = np.array([[0, -1j], [1j, 0]])
sigma_num_3 = np.array([[1, 0], [0, -1]])
sigma_num = [sigma_num_0, sigma_num_1, sigma_num_2, sigma_num_3]
sigma_num_pow = [[None for _ in range(4)] for _ in range(4)]
for i in range(4):
    for j in range(4):
        sigma_num_pow[i][j] = np.kron(sigma_num[i], sigma_num[j])

rho = [[0 for _ in range(4)] for _ in range(4)]
rho_trace = 0
for i1 in range(4):
    for j1 in range(4):
        for i2 in range(4):
            for j2 in range(4):
                rho[i2][j2] += u[i1][j1] * sigma_num_pow[i1][j1][i2][j2]
for i in range(4): rho_trace += rho[i][i]
rho_norm = [[rho[i][j] / rho_trace for j in range(4)] for i in range(4)]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection = "3d")
# c_points = ax.scatter(height, mass, fat)