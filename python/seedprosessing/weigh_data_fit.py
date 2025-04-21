#This code is used for the weight distribution of you and the seed at different germination stages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import beta
from scipy.optimize import minimize
from scipy.special import beta as beta_func
from scipy.stats import rayleigh
from scipy.stats import weibull_min
from scipy.stats import gamma
from scipy.special import gammaln
from scipy.stats import norm
import scipy.stats as stats

#maximum likelihood estimation
def maximum_likelihood_estimation(para,data):
    alpha_para,beta_para = para
    if alpha_para <= 0 or beta_para <= 0 :
        return np.inf
    #maximum function
    ln_alpha = (alpha_para - 1)*sum(np.log(data))
    ln_beta = (beta_para-1)*sum(np.log(1-data))
    ln_nbeta = len(data)*np.log(beta_func(alpha_para,beta_para))
    return  -(ln_alpha+ln_beta-ln_nbeta)
#weibull maximum likelihood estimation
def weibull_likelihood_estimation(para,data):
    k,lamda = para
    if k <= 0 or lamda <= 0:
        return np.inf
    n_lnk = len(data)*np.log(k)
    nk_lnlamda = len(data)*k*np.log(lamda)
    k1 = (k-1)*np.sum(np.log(data))
    sum_x_lamda = sum((data/lamda)**k)
    return -(n_lnk - nk_lnlamda + k1 -sum_x_lamda)
#gamma maximum likelihood estimation
def gamma_lilelihood_estimation(para,data):
    k,theta = para
    if k<= 0 or theta <= 0:
        return np.inf
    k1 = (k-1)*sum(np.log(data))
    theta1 = sum(data)/theta
    nk = len(data)*k*np.log(theta)
    nln = len(data)*gammaln(k)
    return -(k1-theta1-nk-nln)

file_path = 'datasheet.xlsx'
all_sheets = pd.read_excel(file_path,sheet_name=None)

ori_data = np.concatenate((all_sheets['2'],all_sheets['3'],all_sheets['4'],\
                       all_sheets['5'],all_sheets['6'],all_sheets['7'],all_sheets['8']\
                    ,all_sheets['9'],all_sheets['10'],all_sheets['11']),axis = 0)
data = np.concatenate((ori_data[:, 1:8], ori_data[:, 11].reshape(-1, 1)), axis=1)
data = np.array(data, dtype=np.float64)
#normalized
# data0_min,data0_max = data[:,2].min(),data[:,2].max()
# data0_norm = (data[:,2] - data0_min) / (data0_max - data0_min)
# data0_norm = data0_norm * (1 - 2 * 0.001) + 0.001
# print(data0_norm)
# alpha_fit, beta_fit, loc_fit, scale_fit = beta.fit(data0_norm, floc=0, fscale=1)
# print(f"beta: α={alpha_fit:.4f}, β={beta_fit:.4f}")
# plt.figure(figsize=(10, 6))
# data[:,3] = data[:,3].astype(float)
# sns.histplot(data[:,0],kde = True,bins = 64)
# plt.title('data distribution')
# plt.xlabel('weight')
# plt.ylabel('number')
# plt.show()
data0_norm = data[:,2]
print(data0_norm)
#moment estimate the initial value
data0_mean = np.mean(data0_norm)
data0_var = np.var(data0_norm,ddof = 0)
data0_norm = np.clip(data0_norm, data0_mean-4*data0_var**0.5, data0_mean+4*data0_var**0.5)
data0_min,data0_max = data0_norm.min(),data0_norm.max()
data0_norm = (data0_norm - data0_min) / (data0_max - data0_min)
data0_norm = data0_norm * (1 - 2 * 0.001) + 0.001

a_init = data0_mean*((data0_mean*(1-data0_mean)/data0_var)-1)
b_init = (1-data0_mean)*((data0_mean*(1-data0_mean)/data0_var)-1)
initial_value = [a_init,b_init]
result = minimize(maximum_likelihood_estimation,initial_value,args = (data0_norm,),method='L-BFGS-B',bounds = [(1e-6,None),(1e-6,None)])
alpha_est,beta_est = result.x
print(f'result:alpha={alpha_est:.4f},beta={beta_est:.4f}')

#likelihood estimation for calculation of nomal distribution parameters
nor0_mu_fit,nor0_sigma_fit = norm.fit(data0_norm)
print(f"normal:mu={nor0_mu_fit:.4f},sigma={nor0_sigma_fit:.4f}")
#likelihood estimation for calculation of Weibull distribution parameters
weibull0_k_fit,weibull0_loc_fit,weibull0_lamda_fit = weibull_min.fit(data0_norm,floc = 0)
print(f"weibull_fit:k={weibull0_k_fit:.4f},lamada={weibull0_lamda_fit:.4f}")
init_weibull0_guess = [1.0,np.mean(data0_norm)]
result = minimize(weibull_likelihood_estimation,init_weibull0_guess,args = (data0_norm,),method='L-BFGS-B',bounds = [(1e-6,None),(1e-6,None)])
weibull0_k_est,weibull0_lamda_est = result.x
print(f"weibull_est:k={weibull0_k_est:.4f},lamada={weibull0_lamda_est:.4f}")

#likelihood estimation for calculation of gamma distribution parameter
gamma0_k_fit,loc,gamma0_theta_fit = gamma.fit(data0_norm,floc = 0)
print(f"gamma_fit:gamma_k={gamma0_k_fit:.2f},gamma_theta={gamma0_theta_fit:.2f}")
init_gamma_value = [(data0_mean**2)/data0_var,data0_var/data0_mean]
result = minimize(gamma_lilelihood_estimation,init_gamma_value,args = (data0_norm,),method='L-BFGS-B',bounds = [(1e-6,None),(1e-6,None)])
gamma0_k_est,gamma0_theta_est = result.x
print(f"gamma_est:gamma_k={gamma0_k_est:.2f},gamma_theta={gamma0_theta_est:.2f}")


#plot picture
plt.figure(figsize=(20,12))
plt.hist(data0_norm,bins = 64,density=True,label="data_histogram",edgecolor='black')
x = np.linspace(np.min(data0_norm),np.max(data0_norm),100)
# pdf_fit = beta.pdf(x,alpha_fit,beta_fit,loc = 0,scale = 1)
# plt.plot(x,pdf_fit,'r-',lw = 2,label=f"Beta_fit(alpha={alpha_fit:.2f},beta={beta_fit:.2f})")
pdf0_norm = norm.pdf(x,loc = nor0_mu_fit,scale = nor0_sigma_fit)
plt.plot(x,pdf0_norm,'g-',lw = 2,label = f"norm(mu={nor0_mu_fit:.2f},sigma={nor0_sigma_fit:.2f})")
plt.text(0.7,0.9,u"\u25A0 "+f"normal:mu={nor0_mu_fit:.2f},sigma={nor0_sigma_fit:.2f}",transform=plt.gca().transAxes,color = 'green',fontsize=16)
pdf0_beta = beta.pdf(x,alpha_est,beta_est,loc = 0,scale = 1)
plt.plot(x,pdf0_beta,'r-',lw = 2,label=f"Beta_est(alpha={alpha_est:.2f},beta={beta_est:.2f})")
plt.text(0.7,0.95,u"\u25A0 "+f"beta:alpha={alpha_est:.2f},beta={beta_est:.2f}",transform=plt.gca().transAxes,color = 'red',fontsize=16)
pdf0_weibull = weibull_min.pdf(x,weibull0_k_est,loc = 0,scale = weibull0_lamda_est)
plt.plot(x,pdf0_weibull,'b-',lw = 2,label=f"weibull_est:k={weibull0_k_est:.2f},lamada={weibull0_lamda_est:.2f}")
plt.text(0.7,0.85,u"\u25A0 "+f"weibull:k={weibull0_k_est:.2f},lamada={weibull0_lamda_est:.2f}",transform=plt.gca().transAxes,color = 'blue',fontsize=16)
pdf0_gamma = gamma.pdf(x,gamma0_k_est,loc = 0,scale = gamma0_theta_est)
plt.plot(x,pdf0_gamma,'c-',lw = 2,label=f"gamma_est:k={gamma0_k_est:.2f},gamma_theta={gamma0_theta_est:.2f}")
plt.text(0.7,0.80,u"\u25A0 "+f"gamma:k={gamma0_k_est:.2f},gamma_theta={gamma0_theta_est:.2f}",transform=plt.gca().transAxes,color = 'cyan',fontsize=16)
plt.title('original weight distribution(normalized)',fontsize=16)
plt.xlabel('weight',fontsize=16)
plt.ylabel('number',fontsize=16)
plt.savefig("original weight distribution.png", dpi=600, bbox_inches='tight')
plt.show()
