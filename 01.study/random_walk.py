import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

np.random.seed(50120057)

steps = np.random.standard_normal(10000)
steps[0] = 0 ### 첫번째 값을 0ㅇ으로 초기화

random_walk = np.cumsum(steps)

fig, ax = plt.subplots()
ax.plot(random_walk)
ax.set_xlabel('timestamps')
ax.set_ylabel('value')
#### 데이터의 흐름을 보면 데이터가 감소했다 증가했다 다시 감소하고 증가하고를 반복한다. 이러한 데이터는 확률 보행이라 볼 수 있다

adf_result = adfuller(random_walk)
print(f"== ADF 통계량 : {adf_result[0]}")
print(f"== p-value : {adf_result[1]}")
#### ADF 검정에 대한 p-value 값이 0.05보다 크기 때문에 시계열이 비정상적이라는 귀무가설을 기각할 수 없다.

plot_acf(random_walk, lags = 30)
#### 자기상관관계가 confidence interval에서 한참 벗어나있고 계속 1의 값에서 존재하므로 이 데이터는 자기상관관계를 갖는 비정상적인 확률보행이다 

diff_random_walk = np.diff(random_walk, n=1)
#### 정상 시계열로 변경하기 위해서 1차차분을 진행 

fig, ax = plt.subplots()
ax.plot(diff_random_walk)
ax.set_xlabel('timestamps')
ax.set_ylabel('value')
#### 차분을 통해서 생성한 확률보행 데이터는 추세가 제거된 것으로 보이며, 분산도 안정된 것으로 보임

adf_result2 = adfuller(diff_random_walk)
print(f"== ADF 통계량 : {adf_result2[0]}")
print(f"== p-value : {adf_result2[1]}")
#### p-value 값이 0.05보다 작으므로 시계열이 비정상적이라는 귀무가설을 기각할 수 있다.

plot_acf(diff_random_walk, lags = 30)
#### lag가 1부터 바로 유의미한 자기상관계수가 없다. 1차 차분한 데이터가 완전히 무작위인 정상 프로세스를 따른다고 볼 수 있다.