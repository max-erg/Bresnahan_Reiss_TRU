import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats

max_N = 5
weighted=False 
out_folder = '../outputs/model_N{}{}/'.format(max_N, '_weighted' if weighted else '')
mc_results = pd.read_csv(out_folder + 'monte_carlo.csv')

original = mc_results[(mc_results.trial_label == 'typical_business') & (mc_results.scenario == 'original')]
original = original[['random','num_firms','count']]
mc_diff = pd.merge(mc_results[mc_results.scenario == 'adjusted'], original, on=['random','num_firms'], suffixes=('_adj','_orig'))
mc_diff['diff'] = mc_diff['count_adj'] - mc_diff['count_orig']

overall_sum = mc_diff.copy()
for field in ['count_adj','count_orig','diff']:
    overall_sum[field] = overall_sum[field] * overall_sum['num_firms']
overall_sum = overall_sum.groupby(['random','trial_label']).sum().reset_index()
overall_sum['num_firms'] = 'total'
sum_adj = overall_sum[['random','trial_label','count_adj']]
sum_adj['count'] = sum_adj['count_adj']
sum_adj['scenario'] = sum_adj['trial_label']
sum_orig = overall_sum[['random','trial_label','count_orig']]
sum_orig['count'] = sum_orig['count_orig']
sum_orig['scenario'] = 'original'
to_plot = pd.concat([sum_orig, sum_adj])
fig, ax = plt.subplots()
sns.barplot(data=to_plot, x='scenario', y='count', ax=ax, capsize=0.1, errwidth=1.5)
def get_ttest(x,y):
    return stats.ttest_ind(x, y, equal_var=False).pvalue

pvals = overall_sum.groupby('trial_label').apply(lambda dfx: get_ttest(
    dfx['count_adj'],
    dfx['count_orig']))
plt.xticks(np.arange(4), ['Original \n ', '5% of Profit \n p: {:.3g}'.format(pvals['five_percent']),
								 '1% of Profit\n p: {:.3g}'.format(pvals['one_percent']), 
								 'SRIA\n p: {:.3g}'.format(pvals['typical_business'])],
                  fontsize=12)
ax.set_xlabel('')
ax.set_ylabel('Predicted Number of Firms', fontsize=11)
plt.savefig(out_folder + '/total_number_firms.png', dpi=200)
plt.close()

fig, ax = plt.subplots()
mc_diff['Counterfactual'] = [['SRIA','1% of Profit', '5% of Profit']\
                             [['typical_business','one_percent','five_percent'].index(x)]\
                             for x in mc_diff.trial_label]
sns.barplot(data=mc_diff, x='num_firms', y='diff', hue='Counterfactual', capsize=0.05, errwidth=1.5, ax=ax)
plt.legend()
ax.set_ylabel('Change in Number of Markets')
plt.xticks(np.arange(max_N + 1), [str(x) if x < max_N else r'$\geq$ ' + str(x) for x in range(max_N + 1)],
                  fontsize=12);
ax.set_xlabel('Number of Firms per Market')
plt.savefig(out_folder + '/diff_by_firms_per_market.png', dpi=200)
plt.close()