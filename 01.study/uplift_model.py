### 오퍼 정책(할인/1+1)에 따른 conversion rate을 확인하여 각 정책에 따른 고객의 CV 가능성을 예측해보고자 함

'''
변수 설명
    recency : 마지막 구매로부터 지난 개월수
    history : 과거 구매 금액
    used_discount : 과거 할인 오퍼에 반응한 이력이 있는지 여부
    used_bogo : 과거 1+1 오퍼에 반응한 이력이 있는지 여부
    zip_code : 거주 지역에 대한 정보
    is_referral : 고객이 추천 채널에서 구매했는지 여부
    channel : 고객이 이용한 채널(모바일/웹/양쪽 모두)
    offer : 고객에게 보낸 오퍼의 종류 (할인/1+1/오퍼없음)
    conversion : 고객의 CV데이터(buy or not)
'''

'''
데이터셋 링크 : https://www.kaggle.com/code/davinwijaya/uplift-modeling-qini-curve-with-python/input
'''
import numpy as np 
import pandas as pd 
import os
import joblib
import numpy as np
import pandas as pd
import warnings

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

from sklearn.model_selection import train_test_split
import xgboost as xgb

# setting up options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

df_data = pd.read_csv(r'C:\Users\seolbluewings\Desktop\sample\data.csv')
df_model = df_data.copy()

df_model.head()

''' 데이터에 대해서는 null 값이 존재하지 않고 총 3개의 컬럼은 범주형 데이터로 각 3개의 유효값을 가짐'''
df_model.describe(include=np.object)

''' 각 범주형 데이터에 대해서 개수 count하고 이에 대해 visualization '''

cat_features = [col for col in df_model.columns if df_model[col].dtype == 'object']
for c in cat_features:
    print('----')
    print(df_model[c].value_counts())

plt.rcParams['figure.dpi'] = 600
fig = plt.figure(figsize=(6, 6), facecolor='#f6f5f5')
gs = fig.add_gridspec(3, 3)
gs.update(wspace=0.3, hspace=0.3)

background_color = "#f6f5f5"
sns.set_palette(['#99d6e6']*3)

run_no = 0
for row in range(0, 1):
    for col in range(0, 3):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

ax0.text(-0.5, 55, 'Categorical Variable Distribution', fontsize=6, fontweight='bold')
# ax0.text(-0.5, 60, 'features_25 - features_49', fontsize=6, fontweight='light')        

run_no = 0
for col in cat_features:
    temp_df = pd.DataFrame(df_model[col].value_counts())
    temp_df = temp_df.reset_index(drop=False)
    temp_df.columns = ['Number', 'Count']
    sns.barplot(ax=locals()["ax"+str(run_no)],x=temp_df['Number'], y=temp_df['Count']/len(df_model)*100, zorder=2, linewidth=0, alpha=1, saturation=1)
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
    locals()["ax"+str(run_no)].grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
    locals()["ax"+str(run_no)].set_ylabel('')
    locals()["ax"+str(run_no)].set_xlabel(col, fontsize=4, fontweight='bold')
    locals()["ax"+str(run_no)].tick_params(labelsize=4, width=0.5, length=1.5)
    locals()["ax"+str(run_no)].yaxis.set_major_formatter(ticker.PercentFormatter())
    # data label
    for p in locals()["ax"+str(run_no)].patches:
        percentage = f'{p.get_height():.1f}%\n'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        locals()["ax"+str(run_no)].text(x, y, percentage, ha='center', va='center', fontsize=4)

    run_no += 1

plt.show()

''' 컬럼명 변경, conversion 값을 target 변수로 설정함 '''
df_model = df_model.rename(columns={'conversion': 'target'})

''' 컬럼명 수정하고 각 유효값을 -1,0,1 값으로 matching 처리함 '''
df_model = df_model.rename(columns={'offer': 'treatment'})
df_model.treatment = df_model.treatment.map({'No Offer': 0, 'Buy One Get One': -1, 'Discount': 1})

''' zip_code 값과 channel의 경우는 dummy 변수로 처리 '''
df_model = pd.get_dummies(df_model)
df_model.head()

''' 1+1 오퍼의 효과를 검증하기 위해서 오퍼 없음 고객과 1+1 오퍼 고객군을 모아놓은 데이터프레임 생성 '''
df_model_bogo = df_model[df_model.treatment <= 0] 

temp_df = df_model_bogo.mean()
temp_df = temp_df.reset_index(drop=False)
temp_df.columns = ['Features', 'Mean']
temp_df = temp_df.iloc[2:,:]
#temp_df = temp_df.sort_values('Mean', ascending=False)

plt.rcParams['figure.dpi'] = 600
fig = plt.figure(figsize=(5, 1.5), facecolor='#f6f5f5')
gs = fig.add_gridspec(1, 1)
gs.update(wspace=0.4, hspace=0.1)

background_color = "#f6f5f5"
sns.set_palette(['#ffa600']*11)

ax = fig.add_subplot(gs[0, 0])
for s in ["right", "top"]:
    ax.spines[s].set_visible(False)
ax.set_facecolor(background_color)
ax_sns = sns.barplot(ax=ax, x=temp_df['Features'], 
                      y=temp_df['Mean'], 
                      zorder=2, linewidth=0, alpha=1, saturation=1)
ax_sns.set_xlabel("Features",fontsize=4, weight='bold')
ax_sns.set_ylabel("Mean",fontsize=4, weight='bold')
ax_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
ax_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
ax_sns.tick_params(labelsize=2.5, width=0.5, length=1.5)
ax.text(-0.5, 0.9, 'Mean Values - BOGO Treatment', fontsize=6, ha='left', va='top', weight='bold')
ax.text(-0.5, 0.8, 'Each feature get value of 0 (No) and 1 (yes) => Mean is the % of Yes', fontsize=4, ha='left', va='top')
# data label
for p in ax.patches:
    percentage = f'{p.get_height():.1f}' ##{:. 0%}
    x = p.get_x() + p.get_width() / 2
    y = p.get_height() + 0.05
    ax.text(x, y, percentage, ha='center', va='center', fontsize=3,
           bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.3))

plt.show()

''' 할인 오퍼의 효과를 검증하기 위해서 오퍼 없음 고객과 할인 오퍼 고객군을 모아놓은 데이터프레임 생성'''
df_model_discount = df_model[df_model.treatment >= 0] 

temp_df = df_model_discount.mean()
temp_df = temp_df.reset_index(drop=False)
temp_df.columns = ['Features', 'Mean']
temp_df = temp_df.iloc[2:,:]
#temp_df = temp_df.sort_values('Mean', ascending=False)

plt.rcParams['figure.dpi'] = 600
fig = plt.figure(figsize=(5, 1.5), facecolor='#f6f5f5')
gs = fig.add_gridspec(1, 1)
gs.update(wspace=0.4, hspace=0.1)

background_color = "#f6f5f5"
sns.set_palette(['#ff6361']*11)

ax = fig.add_subplot(gs[0, 0])
for s in ["right", "top"]:
    ax.spines[s].set_visible(False)
ax.set_facecolor(background_color)
ax_sns = sns.barplot(ax=ax, x=temp_df['Features'], 
                      y=temp_df['Mean'], 
                      zorder=2, linewidth=0, alpha=1, saturation=1)
ax_sns.set_xlabel("Features",fontsize=4, weight='bold')
ax_sns.set_ylabel("Mean",fontsize=4, weight='bold')
ax_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
ax_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
ax_sns.tick_params(labelsize=2.5, width=0.5, length=1.5)
ax.text(-0.5, 0.8, 'Mean Values - Discount Treatment', fontsize=6, ha='left', va='top', weight='bold')
ax.text(-0.5, 0.7, 'Each feature get value of 0 (No) and 1 (yes) => Mean is the % of Yes', fontsize=4, ha='left', va='top')
# data label
for p in ax.patches:
    percentage = f'{p.get_height():.1f}' ##{:. 0%}
    x = p.get_x() + p.get_width() / 2
    y = p.get_height() + 0.05
    ax.text(x, y, percentage, ha='center', va='center', fontsize=3,
           bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.3))

plt.show()

''' 
타겟 Class에 대한 정의 
Control Non-Responders(CN) : 오퍼를 제공하지 않은 대조군이며 구매로 이어지지 않은 고객 (value == 0) \n
Control Responders(CR) : 오퍼를 제공하지 않은 대조군임에도 불구하고 구매한 고객 (value == 1) \n 
Treatment Non-Responders(TN) : 오퍼를 제공받았음에도 구매로 전환되지 않은 고객 (value == 2) \n
Treatment Responders(TR) : 오퍼를 제공받았으며 구매로 전환된 고객 (value == 3) \n
'''

def declare_tc(df:pd.DataFrame):
    df['target_class'] = 0 # CN
    df.loc[(df.treatment == 0) & (df.target != 0),'target_class'] = 1 # CR
    df.loc[(df.treatment != 0) & (df.target == 0),'target_class'] = 2 # TN
    df.loc[(df.treatment != 0) & (df.target != 0),'target_class'] = 3 # TR
    return df

df_model_bogo = declare_tc(df_model_bogo)
df_model_discount = declare_tc(df_model_discount)

''' 1+1 오퍼와 할인 오퍼에 대해서 실험군/대조군 별로 CV 여부에 대한 데이터 집계 '''
df_model_bogo['target_class'].value_counts()
df_model_discount['target_class'].value_counts()

'''
    - Uplift Score는 다음과 같이 계산한다
    - Uplift Score = (TR)/(T) + (CN)/(C) - (TN)/(T) - (CR)/(C)
    - T는 TR+TN 의 인원수이며 C는 CR+CN의 인원수값을 의미한다
'''

def uplift_split(df_model:pd.DataFrame):
    ## 1 - 데이터를 train/test 데이터로 분리
    
    x = df_model.drop(['target','target_class'],axis=1)
    y = df_model.target_class
    x_tr, x_ts, y_tr, y_ts = train_test_split(x, y,test_size=0.3, random_state=50120057,stratify=df_model['treatment'])
    return x_tr, x_ts, y_tr, y_ts


def uplift_model(x_tr:pd.DataFrame, x_ts:pd.DataFrame, y_tr:pd.DataFrame, y_ts:pd.DataFrame):
    ## 2 - uplift score 생성을 위해 xgboost 모델 사용
    ## result DataFrame 생성
    
    result = pd.DataFrame(x_ts).copy()    
    # Fit the model
    uplift_model = xgb.XGBClassifier().fit(x_tr.drop('treatment', axis=1), y_tr)
    
    #  test 데이터를 활용하여 예측
    
    uplift_proba = uplift_model.predict_proba(x_ts.drop('treatment', axis=1))
    result['proba_CN'] = uplift_proba[:,0] 
    result['proba_CR'] = uplift_proba[:,1] 
    result['proba_TN'] = uplift_proba[:,2] 
    result['proba_TR'] = uplift_proba[:,3]
    result['uplift_score'] = result.eval('\
    proba_CN/(proba_CN+proba_CR) \
    + proba_TR/(proba_TN+proba_TR) \
    - proba_TN/(proba_TN+proba_TR) \
    - proba_CR/(proba_CN+proba_CR)')  
    
    # test 데이터에 대한 결과값을 강제 입력
    result['target_class'] = y_ts
    return result


def uplift(df_model:pd.DataFrame):
    # Combine the split and Modeling function
    x_tr, x_ts, y_tr, y_ts = uplift_split(df_model)
    result = uplift_model(x_tr, x_ts, y_tr, y_ts)
    return result

''' 각 오퍼 데이터에 대해서 uplift 모델 적용 '''
bogo_uplift = uplift(df_model_bogo) 
discount_uplift = uplift(df_model_discount)

''' Qini-Curve 계산'''

# Functions to build the Uplift model and visualize the QINI Curve
def qini_rank(uplift:pd.DataFrame):
    """Rank the data by the uplift score
    """
    # Creat new dataframe
    ranked = pd.DataFrame({'n':[], 'target_class':[]})
    ranked['target_class'] = uplift['target_class']
    ranked['uplift_score'] = uplift['uplift_score']
    
    
    # Add proportion
    ranked['n'] = ranked.uplift_score.rank(pct=True, ascending=False)
    # Data Ranking   
    ranked = ranked.sort_values(by='n').reset_index(drop=True)
    return ranked


def qini_eval(ranked:pd.DataFrame):
    """Evaluate the uplift value with the QINI criterion
    """
    uplift_model, random_model = ranked.copy(), ranked.copy()
    # Using Treatment and Control Group to calculate the uplift (Incremental gain)
    C, T = sum(ranked['target_class'] <= 1), sum(ranked['target_class'] >= 2)
    ranked['cr'] = 0
    ranked['tr'] = 0
    ranked.loc[ranked.target_class == 1,'cr'] = 1
    ranked.loc[ranked.target_class == 3,'tr'] = 1
    ranked['cr/c'] = ranked.cr.cumsum() / C
    ranked['tr/t'] = ranked.tr.cumsum() / T
    # Calculate and put the uplift and random value into dataframe
    uplift_model['uplift'] = round(ranked['tr/t'] - ranked['cr/c'],5)
    random_model['uplift'] = round(ranked['n'] * uplift_model['uplift'].iloc[-1],5)
    
    
    # Add q0
    q0 = pd.DataFrame({'n':0, 'uplift':0, 'target_class': None}, index =[0])
    uplift_model = pd.concat([q0, uplift_model]).reset_index(drop = True)
    random_model = pd.concat([q0, random_model]).reset_index(drop = True)  
    # Add model name & concat
    uplift_model['model'] = 'Uplift model'
    random_model['model'] = 'Random model'
    merged = pd.concat([uplift_model, random_model]).sort_values(by='n').reset_index(drop = True)
    return merged


def qini_plot(merged:pd.DataFrame):
    """Plot the QINI"""
    # plot the data
    ax = sns.lineplot(x='n', y='uplift', hue='model', data=merged,
                      style='model', palette=['red','grey'])
    
    # Plot settings
    sns.set_style('whitegrid')
    handles, labels = ax.get_legend_handles_labels()
    plt.xlabel('Proportion targeted',fontsize=15)
    plt.ylabel('Uplift',fontsize=15)
    plt.subplots_adjust(right=1)
    plt.subplots_adjust(top=1)
    plt.legend(fontsize=12)
    ax.tick_params(labelsize=15)
    ax.legend(handles=handles[1:], labels=labels[1:], loc='upper right')
    return ax


def qini(uplift:pd.DataFrame):
    """Combine all functions
    """
    ranked = qini_rank(uplift)
    merged = qini_eval(ranked)
    ax = qini_plot(merged)
    return ax



qini(bogo_uplift)
plt.title('Qini Curve - Buy One Get One',fontsize=10)

qini(discount_uplift)
plt.title('Qini Curve - Discount',fontsize=10)

