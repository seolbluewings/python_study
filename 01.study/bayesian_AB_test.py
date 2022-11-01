from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
from scipy import stats as st
import numpy as np

camp_A = 57314
camp_B = 38342

conversion_A = 3709
conversion_B = 2632

alpha = 1; beta = 1

post_sample_A = st.beta(alpha+conversion_A, beta+camp_A-conversion_A).rvs(10000)
post_sample_B = st.beta(alpha+conversion_B, beta+camp_B-conversion_B).rvs(10000)

print((post_sample_A > post_sample_B).mean())

plt.hist(post_sample_A, label = "Campaign A", bins = 30, histtype = 'stepfilled', alpha = 0.8)
plt.hist(post_sample_B, label = "Campaign B", bins = 30, histtype = 'stepfilled', alpha = 0.8)
plt.xlabel("Probability"); plt.ylabel("Density")
plt.legend(loc = "best")