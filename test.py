import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# MNIST 데이터 불러오기
data = load_digits()
img_color = plt.imread('./test.jpg')
# 2차원으로 차원 축소
n_components = 2

x = np.loadtxt("./sequence_int.txt", dtype='float', delimiter=',', skiprows = 1)
#print(x[0][0])
print("y.shape:{}".format(x))

# t-sne 모델 생성
model = TSNE(n_components=n_components)
print(model.fit_transform(x))
#
y = model.fit_transform(x)
## 학습한 결과 2차원 공간 값 출력
#print(model.fit_transform(data.data))
# [
#     [67.38322, -1.9517338],
#     [-11.936052, -8.906425],
#     ...
#     [-10.278599, 8.832907],
#     [25.714725, 11.745557],
# ]
plt.scatter(y[:,0], y[:,1], alpha=0.9, c=y[:,0], s=3, cmap='viridis')
plt.show()