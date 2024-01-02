import numpy as np
import matplotlib.pyplot as plt

# 초기 상태에 대한 확률 분포를 설정함
# 맑음, 흐림, 비, 눈 4가지 상태에 대해 동일한 확률을 설정함
prior_probs = np.array([0.25, 0.25, 0.25, 0.25]) 

# 날씨의 각 상태 변화 확률 (가정하는 값임)
# (1,1) 값은 P(t+1시점 맑음|t시점 맑음)의 값이고 각 상태별로 대응시킬 수 있음
transition_probs = np.array([[0.7, 0.2, 0.1, 0.0],
                             [0.3, 0.4, 0.2, 0.1],
                             [0.0, 0.2, 0.6, 0.2],
                             [0.0, 0.1, 0.2, 0.7]])

# 관측 모델
# 각 상태의 날씨에서 비가 올 확률을 의미함
# 상태의 추정값이 주어졌을 때, 정말로 비가 오는가?에 대한 확률
# p(관측 = 비 | 추정값 = 맑음) = 0.1 이라는 말임
observation_probs = np.array([[0.1, 0.3, 0.8, 0.5]])

def bayesian_filter(prior_probs, transition_probs, observation_probs, n_steps):
    #결과를 저장할 배열을 초기화하고 prior로 설정할 확률을 첫번째 값으로 저장
    filtered_probs = np.zeros((n_steps+1, len(prior_probs)))
    filtered_probs[0] = prior_probs

    for t in range(1, n_steps+1):
        # 예측 단계 (상태 전이)
        # 다음 단계에서의 상태값을 예측함
        predicted_probs = np.dot(filtered_probs[t-1], transition_probs)

        # 업데이트 단계 (측정 모델을 통한 업데이트)
        # 상태값에 대한 확률 분포값이 주어졌을 때, 관측값에 대한 분포를 예측함
        # 
        observation = np.dot(observation_probs, predicted_probs)
        updated_probs = predicted_probs * observation

        # 정규화, 확률의 합이 1이 되도록 조치함
        updated_probs /= np.sum(updated_probs)
        
        # 업데이트 된 확률을 배열에 저장함
        filtered_probs[t] = updated_probs

    return filtered_probs

# 5번의 시간 단계 동안의 예측 및 업데이트
n_steps = 5
filtered_probs = bayesian_filter(prior_probs, transition_probs, observation_probs, n_steps)

# 결과 시각화
plt.figure(figsize=(10, 6))
weather_states = ['Clear', 'Cloudy', 'Rainy', 'Snowy']
for i in range(len(prior_probs)):
    plt.plot(filtered_probs[:, i], label=f'State: {weather_states[i]}')

plt.title('Bayesian Filter for Weather Prediction')
plt.xlabel('Time Step')
plt.ylabel('Probability')
plt.legend()
plt.show()