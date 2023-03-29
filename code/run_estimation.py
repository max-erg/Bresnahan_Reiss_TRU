import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from matplotlib import pyplot as plt
import geopandas as gpd
# import seaborn as sns
import scipy.stats
import scipy.optimize
import mapclassify
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample
import seaborn as sns
import pickle 
import csv 
import os

filename = '../data/overall_data.csv'
df = pd.read_csv(filename)
df['total_income'] = df['households'] * df['mean_income']
relevant_vars = ['diesel','land_value','total_income','44-45_RCPTOT','72_RCPTOT', 'num_firms']
subset = df.dropna(subset=relevant_vars)
subset[['land_value','total_income','44-45_RCPTOT','72_RCPTOT', 'diesel']] = \
        StandardScaler().fit_transform(subset[['land_value','total_income','44-45_RCPTOT','72_RCPTOT', 'diesel']])

# parameters defining the model fitting
max_N = 5
weighted=True 
# Parameters defining the counterfactual simulation 
typical_cost = 65 #65,000 maximum amortized annual costs per typical business 
typical_gross_margin = 30000 # from BizMiner report on typical costs from large long-distance trucking firm 
# Gross margin: around 850,000 for small business, 30,000,000 for large business

out_folder = '../outputs/model_N{}{}/'.format(max_N, '_weighted' if weighted else '')
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

"""
Defining common functions
"""

def V(df, Ni, alpha=np.ones(max_N ), beta=1):
    """Variable profit
    
    As a function of the market size. 
    
    df: data
    Ni: number of incumbents in market i
    alpha: alpha parameters
    beta: beta parameters
    """
    n = np.min([max_N, Ni])
    
    profit = df['diesel'] * beta - np.sum(alpha[1:n ]) + alpha[0]
    return profit

def F(df, Ni, gamma=np.ones(max_N), W_L = 1):
    n = np.min([max_N, Ni])
    
    fixed_cost = np.sum(gamma[1:n]) + df['land_value'] * W_L + gamma[0]
    return fixed_cost


def S_func(df, lam=np.ones(3)):
    return 1 + df[['total_income','44-45_RCPTOT', '72_RCPTOT']] @ lam
#     S = dTPOP + lam[0] * df.OPOP + lam[1] * df.NGRW + 
#          lam[2] * df.PGRW + lam[3] * df.OCTY)
#     return S

def neg_log_lik(theta=np.ones(5 + 2 * max_N), df=None, weighted=weighted):
    lam = theta[0:3]
    alpha = theta[3: 3 + max_N]
    gamma = theta[3 + max_N: 3 + 2 * max_N]
    beta = theta[3 + 2* max_N: 4 + 2* max_N]
    W_L = theta[4 + 2* max_N: 5 + 2* max_N]
    
    Phi = scipy.stats.norm.cdf
    
    S = S_func(df, lam)

    with np.errstate(divide='ignore'): # Dividing by zero is anticipated. 
        P = [0] * (max_N + 1)
        Pi_bar = lambda N: S * V(df, N, alpha=alpha, beta=beta) - F(df, N, gamma=gamma, W_L=W_L)
        P[0] = np.log( 1 - Phi(Pi_bar(1)) )
        P[max_N] = np.log( Phi(Pi_bar(max_N)) )
        for i in range(1,max_N):
            P[i] = np.log( Phi(Pi_bar(i)) - Phi(Pi_bar(i+1)) )

        for i in range(max_N + 1):
            P[i][P[i] == -np.inf] = -100000000
            P[i][P[i] == np.inf] = 100000000
        log_lik = 0 
        weighting_factor = [1 / np.sum(df['num_firms'] == i) for i in range(max_N)] + \
                        [1 / np.sum(df['num_firms'] >= max_N)] \
                        if weighted else np.ones(max_N + 1)
        for i in range(max_N):
            log_lik = log_lik + np.sum(P[i] * (df['num_firms'] == i)) * weighting_factor[i]
        log_lik = log_lik + np.sum(P[max_N] * (df['num_firms'] >= max_N)) * weighting_factor[max_N]
    return -log_lik

def theta_to_param_dict(theta):
    lam = theta[0:3]
    alpha = theta[3: 3 + max_N]
    gamma = theta[3 + max_N: 3 + 2 * max_N]
    beta = theta[3 + 2* max_N: 4 + 2* max_N]
    W_L = theta[4 + 2* max_N: 5 + 2* max_N]
#     lam = theta[0:3]
#     alpha = theta[3: 2 + max_N]
#     gamma = theta[2 + max_N: 2 + 2 * max_N]
#     beta = theta[2 + 2* max_N: 3 + 2* max_N]
#     W_L = theta[3 + 2* max_N: 4 + 2* max_N]
    
    d = {'lam': lam, 'alpha': alpha, 'gamma': gamma, 'beta': beta, 'W_L':W_L, 'theta':theta}
    return d
lower_bounds = [0] * 3 + [0] * (2 * max_N ) + [-np.inf] * 2
theta_size = len(lower_bounds)
upper_bounds = [np.inf] * theta_size
bounds = list(zip(lower_bounds, upper_bounds))
nll = lambda theta: neg_log_lik(theta=theta, df=subset, weighted=weighted)

"""
Model fitting 
"""

initial_params = []
neg_likelihood = []
best_model = None
for param in np.logspace(-4, 1, 21):
    initial_guess = param * np.ones(theta_size)
    out = scipy.optimize.minimize(nll, initial_guess, 
                                  options={'disp':False, 'maxiter':30000}, bounds=bounds)
    initial_params.append(param)
    neg_likelihood.append(out.fun)
    if out.fun == np.min(neg_likelihood):
        best_model = out

min_param = initial_params[neg_likelihood.index(np.min(neg_likelihood))]
print("Loss minimized at", min_param)

"""
Model evaluation, just predicted vs real distribution 
"""

fig, ax = plt.subplots()
plt.plot(initial_params, neg_likelihood)
ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_ylim(bottom=None, top=15)
ax.set_ylabel('Loss (negative log likelihood)')
ax.set_xlabel('Scale of Initial Parameter Vector')
ax.axvline(min_param, linestyle='--', color='red')
plt.savefig(out_folder + 'grid_search.png', dpi=200, bbox_inches='tight')
plt.close()

param_dict = theta_to_param_dict(best_model.x )
with open(out_folder + 'model_params.pkl', mode='wb') as pfile:
    pickle.dump(param_dict, pfile)
print(param_dict)

def count_probabilities(theta=np.ones(5 + 2 * max_N), df=None):
    lam = theta[0:3]
    alpha = theta[3: 3 + max_N]
    gamma = theta[3 + max_N: 3 + 2 * max_N]
    beta = theta[3 + 2* max_N: 4 + 2* max_N]
    W_L = theta[4 + 2* max_N: 5 + 2* max_N]
    
    Phi = scipy.stats.norm.cdf
    
    S = S_func(df, lam)

    P = [0] * (max_N + 1)
    Pi_bar = lambda N: S * V(df, N, alpha=alpha, beta=beta) - F(df, N, gamma=gamma, W_L=W_L)
    P[0] = 1 - Phi(Pi_bar(1)) 
    P[max_N] = Phi(Pi_bar(max_N)) 
    for i in range(1,max_N):
        P[i] = Phi(Pi_bar(i)) - Phi(Pi_bar(i+1)) 
    return np.vstack(P).transpose()
P = count_probabilities(theta = param_dict['theta'], df = subset)
subset['predicted'] = np.argmax(P, axis=1)


true_labels = np.minimum(subset.num_firms, max_N)
predicted = np.argmax(P, axis=1)
confusion_matrix = confusion_matrix(true_labels, predicted, normalize='true')
ConfusionMatrixDisplay(confusion_matrix = confusion_matrix).plot()
plt.ylabel('True Markets, by Firms per Market')
plt.xlabel('Predicted Markets, by Firms per Market')
plt.savefig(out_folder + 'confusion_matrix.png', dpi=200, bbox_inches='tight')
plt.close()

state_df = gpd.read_file('../shapes/s_22mr22')
state_df = state_df[state_df.STATE.isin(['CA', 'OR', 'NV', 'AZ', 'UT', 'ID'])]
state_df = state_df.to_crs(epsg=2163)
counties = gpd.read_file('../shapes/cb_2018_us_county_500k')
counties = counties[counties.STATEFP == '06']
counties['GEOID'] = counties['GEOID'].astype(int)
counties['coords'] = counties['geometry'].apply(lambda x: x.representative_point().coords[:])
counties['coords'] = [coords[0] for coords in counties['coords']]
counties = counties.to_crs(epsg=2163)
places = gpd.read_file('../shapes/tl_2019_06_place')
places = places[places.STATEFP == '06']
places = places.to_crs(epsg=2163)
places['clean_city'] = [x.lower() for x in places.NAME]

def my_map_plot(merged, column, cmap='viridis_r'):
    fig, ax = plt.subplots(1, figsize=(6, 11))
    state_df[state_df.STATE == 'CA'].plot(ax=ax, color='white', edgecolor='black', linewidth=0.8)
    x_low,x_high = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax = merged.plot(column=column, cmap=cmap, linewidth=0.1, ax= ax, edgecolor='0.8', 
               legend=True)#classification_kwds={'bins':bounds[1:]})

    state_df[state_df.STATE != 'CA'].plot(ax=ax, color="#E2E2E2", edgecolor='#727475')
    counties.plot(ax=ax, color='none',edgecolor='black', linewidth=0.8)

    ax.set_xlim((x_low, x_high))
    ax.set_ylim((y_low, y_high))
    ax.axis('off')
    return ax, plt

merged = places.merge(subset, on='NAMELSAD', how='right')
for var in ['predicted','num_firms']:
    merged[var] = [str(int(x)) if x < 5 else ">= 5" for x in merged[var]]
    ax= my_map_plot(merged, column=var, cmap='viridis_r')
    plt.savefig(out_folder + f'map_{var}.png', dpi=200, bbox_inches='tight')
    plt.close()

subset['Number of Firms'] = np.minimum(subset.num_firms, max_N)
new_panel = subset[['Number of Firms']]
new_panel[''] = 'Actual'
predicted = subset[['predicted']]
predicted['Number of Firms'] = predicted['predicted']
predicted[''] = "Predicted"
ax = sns.countplot(data = pd.concat([new_panel, predicted]), x = 'Number of Firms', hue='')
ax.set_xlabel('Number of Firms per Market')
ax.set_ylabel('Count of Markets in Category')
plt.savefig(out_folder + 'count_comparison.png', dpi=200, bbox_inches='tight')
plt.close()

"""
Simulated regulation
"""
print("Running simulations:")
np.random.seed(1234)
rand_list = np.random.randint(10000, size=100)
write_header=True
for random in rand_list:
    np.random.seed(random)
    resampled = resample(subset)
    nll = lambda theta: neg_log_lik(theta=theta, df=resampled, weighted=weighted)
    initial_guess = best_model.x
    out = scipy.optimize.minimize(nll, initial_guess, 
                    options={'disp':False, 'maxiter':30000}, bounds=bounds)
    params = out.x 
    trials_dict = {}
    trials_dict['original'] = 1
    trials_dict['typical_business'] = (typical_gross_margin - typical_cost) / typical_gross_margin
    trials_dict['one_percent'] = 0.99
    trials_dict['five_percent'] = 0.95
    for trial_label, adj_factor in trials_dict.items():
        adjusted_params = params.copy()
        adjusted_params[3] = adjusted_params[3] * adj_factor
        P_adjusted = count_probabilities(theta = adjusted_params, df = resampled)
        with open(out_folder + f'monte_carlo.csv', 'w' if write_header else 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['random', 'trial_label','num_firms', 'count'])
            if write_header: writer.writeheader()
            counts_dict = dict(zip(*np.unique(np.argmax(P_adjusted, axis=1), return_counts=True)))
            for num_firms in range(max_N + 1):
                writer.writerow({'random': random,'trial_label':trial_label, 
                                 'num_firms': num_firms, 'count':counts_dict[num_firms] if num_firms in counts_dict else 0})
            write_header=False


