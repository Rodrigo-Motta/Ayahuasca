import pandas as pd
import utils as ut
import numpy as np
from model import GCN
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

df = pd.read_csv('/Users/rodrigo/Side-Projects/Ayahuasca/Data/corr_matrices.csv').drop(columns='Unnamed: 0')
#df = pd.read_csv('/Users/rodrigo/Side-Projects/Ayahuasca/Data/corr_matrices_cc200.csv').drop(columns='Unnamed: 0')
HRS = pd.read_csv('/Users/rodrigo/Side-Projects/Ayahuasca/Data/Ayahuasca_HRS.csv')
N = 333


condition = (
    ((df['Group'] == 'J') & (df['Subject'].isin([7, 9, 18]))) |     # Examinate subject 18
    ((df['Group'] == 'O') & (df['Subject'].isin([1, 7])))
)

# Inverting the condition to keep rows that do NOT match the condition
df = df[~condition]

metrics = pd.DataFrame()

for g in ['O', 'J']:
    for i in ['before', 'after']:
        aux = df[(df.Time == i) & (df.Group == g)]#.replace(np.nan, 0.0).reset_index().iloc[:,2:-3]
        X_fmri = aux.select_dtypes(include=float)
        aux['STD'] = X_fmri.std(axis=1)
        aux['MEAN'] = X_fmri.mean(axis=1)

        #aux = aux.join(X_fmri.apply(ut.calculate_properties, axis=1).apply(pd.Series))

        y = aux.reset_index().iloc[:,0]

        A = ut.reconstruct_symmetric_matrix(N, X_fmri.iloc[:,:].mean(axis=0))
        train_data, val_data = ut.create_graph(X_fmri, X_fmri, y, y,size=N, method={'knn_group' : ut.compute_KNN_graph(A, 15)})#, method={'threshold': 0.8})
        train_loader, val_loader = ut.create_batch(train_data, val_data, batch_size=1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        model = GCN(N, 3).to(device)
        model.load_state_dict(torch.load('/Users/rodrigo/Post-Grad/Ising_GNN/Data/model_params_333_TRUE.pth'))
        model.eval()

        y_pred_aux_age = []
        for y_i in val_loader:
            y_pred_aux_age.append((model(y_i))[1].detach().numpy().ravel()[0])

        aux['TEMP'] = y_pred_aux_age
        metrics = pd.concat([metrics, aux.loc[:, ['Group', 'Subject', 'Time', 'MEAN', 'STD', 'TEMP']]])
        # metrics = pd.concat([metrics, aux.loc[:,['Group', 'Subject', 'Time', 'MEAN','STD','TEMP',
        #                                          'Weighted Clustering Coefficient', 'Weighted Eigenvector Centrality',
        #                                          'Weighted Density', 'Assortativity (Weight Correlation)'
        #                                          ]]])


        # my_dict_age = {}
        # for i in range(len(y.values)):
        #     my_dict_age[y.values[i]] = y_pred_aux_age[i]

# Inter-subject
# sns.boxplot(metrics, y='TEMP', x='Group', hue='Time')
# plt.show()

# Intra-subject
# Apply the function to each group using groupby
result = ut.calculate_diff(metrics, "TEMP")

result = pd.merge(result.reset_index(), HRS, how='left', on=['Group', 'Subject'])

group_o = result[result.Group == 'O']
group_j = result[result.Group == 'J']

from scipy import stats
print(stats.ttest_ind(group_o['Change'], group_j['Change']))

plt.figure()
# Display the resulting DataFrame
sns.boxplot(result, x='Group', y='Change')
sns.swarmplot(result, x='Group', y='Change')
plt.ylabel('Delta T')
plt.show()

plt.figure()
ayahuasca = group_j[['Subject','after']]
placebo = group_o[['Subject', 'after']]
arr = [ayahuasca.after, placebo.after]
print(stats.ttest_ind(ayahuasca.after, placebo.after))
g = sns.boxplot(arr)
g.set_xticks(range(len(arr))) # <--- set the ticks first
g.set_xticklabels(['Ayahuasca', 'Placebo'])
plt.ylabel('Temperature')
plt.ylim(2.20, 2.55)
plt.show()


ayahuasca = group_j[['Subject', 'Change']]
placebo = group_o[['Subject', 'Change']]

plt.figure(figsize=(10,5))
# Calculate the mean
mean_value = ayahuasca['Change'].mean()

# Normalize the values to create a gradient
#norm = plt.Normalize(ayahuasca['Change'].min(), ayahuasca['Change'].max())
norm = plt.Normalize(-.150, .150)
sm = plt.cm.ScalarMappable(cmap="Purples", norm=norm)
colors = [sm.to_rgba(val) for val in ayahuasca['Change']]

# Plotting
plt.title('Ayahuasca')
plt.axhline(y=0, color='grey', linestyle='-')
sns.barplot(data=ayahuasca, x='Subject', y='Change', palette=colors)
plt.ylabel('Delta T')
plt.axhline(y=ayahuasca['Change'].median(), color='black', linestyle='dotted', label='Median')
plt.axhline(y=mean_value, color='grey', linestyle='--', label='Mean')
plt.ylim(-.170,.170)
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.title('Placebo')
# Calculate the mean
mean_value = placebo['Change'].mean()

# Normalize the values to create a gradient
#norm = plt.Normalize(placebo['Change'].min(), placebo['Change'].max())
norm = plt.Normalize(-.150, .150)
sm = plt.cm.ScalarMappable(cmap="Purples", norm=norm)
colors = [sm.to_rgba(val) for val in ayahuasca['Change']]
plt.axhline(y=0, color='grey', linestyle='-')
sns.barplot(data=pd.DataFrame(placebo).reset_index(), x='Subject', y='Change', palette=colors)
plt.ylabel('Delta T')
plt.axhline(y=placebo['Change'].median(), color='black', linestyle='dotted', label='Median')
plt.axhline(y=mean_value, color='black', linestyle='--', label='Mean')
plt.ylim(-.170,.170)
plt.legend()
plt.show()

X = group_j[['Volition_average','Cognition_average', 'Perception_average','Somaesthesia_average',
             'Intensity_average','Affect_average', 'Volition_std', 'Cognition_std', 'Perception_std', 'Somaesthesia_std',
             'Intensity_std', 'Affect_std']]

for dep_variable in ['Change', 'after','before']:
    for group in ['O', 'J']:

        print('################ Group {}, Depedent Variable {} ##################'.format(group, dep_variable))
        y = result[result.Group == group][dep_variable]
        X = result[result.Group == group][['Volition_average', 'Cognition_average', 'Perception_average', 'Somaesthesia_average',
                     'Intensity_average', 'Affect_average', 'Volition_std', 'Cognition_std', 'Perception_std',
                     'Somaesthesia_std',
                     'Intensity_std', 'Affect_std']]

        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        y = (y - y.min(axis=0)) / (y.max(axis=0) - y.min(axis=0))

        import statsmodels.api as sm

        X = sm.add_constant(X)
        mod = sm.OLS(y, X).fit()
        pred = mod.predict(X)

        print(mod.summary())

