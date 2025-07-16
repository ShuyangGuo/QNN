import numpy as np

h_prev = np.array([1, 2, 3])  # 形状: (3,)
x_t = np.array([4, 5])        # 形状: (2,)

combined = np.concatenate((h_prev, x_t))
print(combined.shape)  # 输出: (5,)
print(combined)        # 输出: [1 2 3 4 5]