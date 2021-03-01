import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.e**(-x))

data = [
  [2,0],
  [4,0],
  [6,0],
  [8,1],
  [10,1],
  [12,1],
  [14,1]
]

x = [2,4,6,8,10,12,14]
y = [0,0,0,1,1,1,1]

a = 0
b = 0

learn_rate = 0.05

for i in range(20000):
  for x, y in data:
    # a에 대한 편미분 결과 적용
    a_diff = x *(sigmoid(a * x +b) - y)

    # b에 대한 편미분 결과 적용
    b_diff = sigmoid(a * x +b) - y      

    # a와 b 학습률에 따라 업데이트
    # 만약 오차율이 0이 나왔다면 a_diff b_diff는 0이므로 이전 값 그대로 유지
    a = a - learn_rate * a_diff
    b = b - learn_rate * b_diff

    if i%500 == 0:
      print('epoch: %f, 기울기: %f, 절편: %f'%( i, a, b))

plt.scatter([i[0] for i in data], [i[1] for i in data], color="#2ca02c")
plt.xlim(0, 15)
plt.ylim(-.1, 1.1)
x_range = (np.arange(0, 15, 0.1))
plt.plot(np.arange(0, 15, 0.1), np.array([sigmoid(a * x + b) for  x in x_range]))

plt.show()