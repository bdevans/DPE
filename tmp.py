#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:51:53 2017

@author: ben
"""


prop_T1 = np.nan_to_num(hc3*hc1/(hc1+hc2)/max(hc3))
prop_T2 = np.nan_to_num(hc3*hc2/(hc1+hc2)/max(hc3))


model_T1 = lmfit.models.GaussianModel(prefix='T1')
res_T1 = model_T1.fit(prop_T1, x=x)
print(res_T1.fit_report())

plt.figure()
fig, (ax1, ax2, axM) = plt.subplots(3, 1, sharex=True, sharey=True)
ax1.plot(x, prop_T1, 'o')
ax1.plot(x, res_T1.best_fit, 'k-')
dely = res_T1.eval_uncertainty(sigma=3)
ax1.fill_between(x, res_T1.best_fit-dely, res_T1.best_fit+dely, color="#ABABAB")


model_T2 = lmfit.models.GaussianModel(prefix='T2')
res_T2 = model_T2.fit(prop_T2, x=x)
print(res_T2.fit_report())

#plt.figure()
ax2.plot(x, prop_T2, 'o')
ax2.plot(x, res_T2.best_fit, 'k-')
dely = res_T2.eval_uncertainty(sigma=3)
ax2.fill_between(x, res_T2.best_fit-dely, res_T2.best_fit+dely, color="#ABABAB")
#plt.show()


#params_mix = {**res_T1.best_values, **res_T2.best_values}
params_mix = res_T1.params.copy()
params_mix.update(res_T2.params)
params_mix['T1center'].vary = False
params_mix['T1sigma'].vary = False
params_mix['T2center'].vary = False
params_mix['T2sigma'].vary = False

#peak1 = lmfit.models.GaussianModel(prefix='p1')
#peak2 = lmfit.models.GaussianModel(prefix='p2')
model = model_T1 + model_T2
# TODO set params based on previous fitting
res_mix = model.fit(freqs_Mix, x=x, params=params_mix)
print(res_mix.fit_report())

#plt.figure()
axM.plot(x, freqs_Mix, 'o')
#axM.bar(x, freqs_Mix)
axM.plot(x, res_mix.best_fit, 'k-')

dely = res_mix.eval_uncertainty(sigma=3)
axM.fill_between(x, res_mix.best_fit-dely, res_mix.best_fit+dely, color="#ABABAB")
#plt.show()