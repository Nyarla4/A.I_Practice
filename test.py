import matplotlib.pyplot as plt#시각화에 필요
import numpy as np#이건 모르겠다
from sklearn.manifold import TSNE#T-SNE 사용
from sklearn.datasets import load_digits#dataSet중 하나
import pandas as pd#데이터 처리용
from sklearn.cluster import KMeans#클러스터링 관련 모시깽이
from sklearn import preprocessing#데이터 전처리용

# MNIST 데이터 불러오기
data = load_digits()
img_color = plt.imread('./test.jpg')

# 2차원으로 차원 축소
n_components = 2

x = np.loadtxt("./sequence_int.txt", dtype='float', delimiter=',', skiprows = 1)
#print(x[0][0]). x는 2차원 배열
#print("y.shape:{}".format(x))

# t-sne 모델 생성
model = TSNE(n_components=n_components)#2차원으로 차원축소하는 model
#print(model.fit_transform(x))

#y = model.fit_transform(x)
#print(y)
## 학습한 결과 2차원 공간 값 출력
#print(model.fit_transform(data.data))
# [
#     [67.38322, -1.9517338],
#     [-11.936052, -8.906425],
#     ...
#     [-10.278599, 8.832907],
#     [25.714725, 11.745557],
# ]
#plt.scatter(y[:,0], y[:,1], alpha=0.9, c=y[:,0], s=3, cmap='viridis')
#plt.show()

student_data=pd.read_excel('./ex2.xls')
students = student_data.groupby('Name').mean()

# normalizer 생성
#min_max_scaler = preprocessing.MinMaxScaler()
# 표준화하기
#students[['Mark', 'Attended']] = min_max_scaler.fit_transform(students[['Mark', 'Attended']])

# k=3 클러스터 생성
estimator = KMeans(n_clusters=3)
cluster_ids = estimator.fit_predict(students)
# 플롯
plt.scatter(students['Attended'], students['Mark'], c=cluster_ids)
plt.xlabel("Attended classes")
plt.ylabel("Mark")
# 범례 달기
for name, mark, attended in students.itertuples():
    plt.annotate(name, (attended, mark))

plt.show()

# load library
#from matplotlib import font_manager
#from sklearn.datasets import load_digits
#import matplotlib
#import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE

#font_path = "./Suseong Dotum.ttf"
#font_name = font_manager.FontProperties(fname=font_path).get_name()
# matplotlib 설정
#matplotlib.rc('font', family=font_name) # 한글 출력
#plt.rcParams['axes.unicode_minus'] = False # 축 - 설정

# data load
#digits = load_digits()

# subplot 객체 생성
#fig, axes = plt.subplots(2, 5, #  subplot객체(2x5)를 axes에 할당
#                         subplot_kw={'xticks':(), 'yticks':()}) # subplot 축 눈금 해제
#for ax, img in zip(axes.ravel(), digits.images): # axes.ravel()과 digits.images를 하나씩 할당
#    ax.imshow(img)
#plt.gray() # 그래프 흑백
#plt.show() # 그래프 출력

# PCA 모델을 생성
#pca = PCA(n_components=2) # 주성분 갯수
#pca.fit(digits.data) # PCA 적용

# 처음 두 개의 주성분으로 숫자 데이터를 변환
#digits_pca = pca.transform(digits.data) # PCA를 데이터에 적용
#colors = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525',
#               '#A83683', '#4E655E', '#853541', '#3A3120', '#535D8E']
#for i in range(len(digits.data)): # digits.data의 길이까지 정수 갯수
    # 숫자 텍스트를 이용해 산점도 그리기
#    plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]), # x, y, 그룹; str은 문자로 변환
#             color=colors[digits.target[i]], # 산점도 색상
#             fontdict={'weight':'bold', 'size':9}) # font 설정
#plt.xlim(digits_pca[:, 0].min(), digits_pca[:,1].max()) # 최소, 최대
#plt.ylim(digits_pca[:, 1].min(), digits_pca[:,1].max()) # 최소, 최대
#plt.xlabel('first principle component') # x 축 이름
#plt.ylabel('second principle componet') # y 축 이름
#plt.show()

# t-SNE 모델 생성 및 학습
#tsne = TSNE(random_state=0)
#digits_tsne = tsne.fit_transform(digits.data)

# 시각화

#for i in range(len(digits.data)): # 0부터  digits.data까지 정수
#    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]), # x, y , 그룹
#             color=colors[digits.target[i]], # 색상
#             fontdict={'weight': 'bold', 'size':9}) # font
#plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max()) # 최소, 최대
#plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max()) # 최소, 최대
#plt.xlabel('t-SNE 특성0') # x축 이름
#plt.ylabel('t-SNE 특성1') # y축 이름
#plt.show() # 그래프 출력

