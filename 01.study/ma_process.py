import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX

### 데이터 불러오기
df = pd.read_csv(r"C:\Users\woori\jupyter\data\widget_sales.csv")
df.head(3)

### 시계열 데이터 Plot 그리기 : 정상 시계열이 아님을 눈으로 확인할 수 있음
fig, ax= plt.subplots()
ax.plot(df['widget_sales'])
ax.set_xlabel("Time")
ax.set_ylabel("Widget Sales (k$)")
fig.autofmt_xdate()
plt.tight_layout()

### 데이터의 정상성 테스트
### p-value가 0.05보다 크기 때문에 시계열은 정상성을 갖는다고 볼 수 없다.
adf_result = adfuller(df['widget_sales'])
print(f"== ADF statistic : {adf_result[0]}")
print(f"== p-value : {adf_result[1]}")

### 데이터에 대한 1차 차분을 진행하여 이 데이터에 대한 MA(q) 프로세스 적합을 착수
df_dif1 = np.diff(df['widget_sales'],n=1)
fig, ax= plt.subplots()
ax.plot(df_dif1)
ax.set_xlabel("Time")
ax.set_ylabel("Widget Sales (k$)")
fig.autofmt_xdate()
plt.tight_layout()

### 차분 데이터의 정상성 테스트
### p-value가 0.05보다 직기 때문에 시계열은 정상성을 갖는다.
adf_result = adfuller(df_dif1)
print(f"== ADF statistic : {adf_result[0]}")
print(f"== p-value : {adf_result[1]}")

### p-value를 고려한 결과, 1차 차분한 데이터는 정상성을 가지므로 자기상관성을 체크함
### lag = 2까지 유의하므로 MA(2) process를 적용하는 것이 옳다
plot_acf(df_dif1, lags = 30)
plt.tight_laytout()

### 데이터를 차분한 결과를 9:1 비율로 train_test 데이터로 나눔
df_dif = pd.DataFrame({'widget_sales_diff':df_dif1})

tr = df_dif[:int(0.9*len(df_dif))]
ts = df_dif[int(0.9*len(df_dif)):]
print(len(tr),len(ts))

### 차분하기 전 데이터와 차분 후 데이터에 대한 시계열 데이터 Plot을 그림
fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex=True)
ax1.plot(df['widget_sales'])
ax1.set_xlabel("Time")
ax1.set_ylabel("Widget Sales (k$)")
ax1.axvspan(450, 500, color = '#808080', alpha = 0.2)

ax2.plot(df_dif['widget_sales_diff'])
ax2.set_xlabel("Time")
ax2.set_ylabel("Widget Sales (k$), 1diff")
ax2.axvspan(449, 498, color = '#808080', alpha = 0.2)

fig.autofmt_xdate()
plt.tight_layout()

### SARIMAX 라이브러리를 이용해서 MA(2) Process에 대한 model fitting
def rolling_forecast(df:pd.DataFrame, tr_length : int, horizon : int, window : int, method : str):
    tot_length = tr_length + horizon

    if method == 'MA':
        MA_pred_list = []
        
        for i in range(tr_length, tot_length, window):
            model = SARIMAX(df.iloc[:i], order = (0,0,2))
            res   = model.fit(disp = False)
            predictions = res.get_prediction(0, i+window-1)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            MA_pred_list.extend(oos_pred)
        return MA_pred_list
    
### 예측 및 결과 출력
pred_MA = rolling_forecast(df_dif['widget_sales_diff'], tr_length = len(tr), horizon = len(ts), window = 1, method = 'MA')
ts['pred_ma'] = pred_MA

### 결과 시각화
mse_MA = mean_squared_error(ts['widget_sales_diff'],ts['pred_ma'])
print(mse_MA)

### 누적 합계를 구해야 데이터를 원복시킨 형태로 값을 구할 수 있다.
df['pred_widget_sales'] = pd.Series()
df['pred_widget_sales'] = df['widget_sales'].iloc[450]+ts['pred_ma'].cumsum() 

fig, ax = plt.subplots()
ax.plot(df['widget_sales'], 'b-', label = 'actual')
ax.plot(df['pred_widget_sales'], 'k--', label = 'MA(2)')
ax.legend(loc = 2)

ax.set_xlabel('Time')
ax.set_ylabel('Widget Sales (k$)')
ax.axvspan(450,500, color = '#808080', alpha = 0.2)
ax.set_xlim(400,500)

fig.autofmt_xdate()
plt.tight_layout()