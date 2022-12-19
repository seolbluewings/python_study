'''
필요 라이브러리 불러옴
'''

from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np


'''
아무것도 선택하지 않은 초기 상태를 가정한 plot 
'''

fig, ax = plt.subplots(1, 1)
x = np.linspace(0, 1, 100)
ax.plot(x, beta.pdf(x, 1, 1), 'r-', lw=3, alpha=0.6, label='banner1') # 첫번째 배너
ax.plot(x, beta.pdf(x, 1, 1), 'g-', lw=3, alpha=0.6, label='banner2') # 두번째 배너
ax.plot(x, beta.pdf(x, 1, 1), 'b-', lw=3, alpha=0.6, label='banner3') # 세번째 배너



'''
5번 정도의 배너 노출이 진행된 후,
1. 1번 배너는 3번 클릭
2. 2번 배너는 1번 클릭
3. 3번 배너는 단 1번도 클릭되지 않은 것으로 가정 
'''


fig, ax = plt.subplots(1, 1)
x = np.linspace(0, 1, 100)

banner1_rvs = beta.rvs(4, 3, size=1)
banner2_rvs = beta.rvs(2, 5, size=1)
banner3_rvs = beta.rvs(1, 6, size=1)

ax.plot(x, beta.pdf(x, 4, 3), 'r-', lw=3, alpha=0.6, label='banner1') # 첫번째 배너
ax.plot(x, beta.pdf(x, 2, 5), 'g-', lw=3, alpha=0.6, label='banner2') # 두번째 배너
ax.plot(x, beta.pdf(x, 1, 6), 'b-', lw=3, alpha=0.6, label='banner3') # 세번째 배너

ax.plot(banner1_rvs, 0, 'x', color='red')
ax.plot(banner2_rvs, 0, 'x', color='green')
ax.plot(banner3_rvs, 0, 'x', color='blue')
ax.legend(loc='best', frameon=False)
plt.show()

'''
1번 배너가 가장 높은 확률로 출력되므로 1번 배너를 우선 노출할 배너로 채택한다 
'''

'''
충분한 데이터가 쌓인 후의 비교
'''

fig, ax = plt.subplots(1, 1)
x = np.linspace(0, 1, 100)

banner1_rvs = beta.rvs(33, 100, size=1)
banner2_rvs = beta.rvs(100, 223, size=1)
banner3_rvs = beta.rvs(435, 611, size=1)

ax.plot(x, beta.pdf(x, 33, 100), 'r-', lw=3, alpha=0.6, label='banner1')
ax.plot(x, beta.pdf(x, 100, 223), 'g-', lw=3, alpha=0.6, label='banner2')
ax.plot(x, beta.pdf(x, 435, 611), 'b-', lw=3, alpha=0.6, label='banner3')

ax.plot(banner1_rvs, 0, 'x', color='red')
ax.plot(banner2_rvs, 0, 'x', color='green')
ax.plot(banner3_rvs, 0, 'x', color='blue')
ax.legend(loc='best', frameon=False)
plt.show()

