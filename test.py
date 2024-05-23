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
N = 333


condition = (
    ((df['Group'] == 'J') & (df['Subject'].isin([7, 9, 18]))) |     # Examinate subject 18
    ((df['Group'] == 'O') & (df['Subject'].isin([1, 7])))
)

# Inverting the condition to keep rows that do NOT match the condition
df = df[~condition]

# df_before_o = df[(df.Time == 'before') & (df.Group == 'O')].replace(np.nan, 0.0).reset_index().iloc[:,2:-3]
# df_after_o = df[(df.Time == 'after') & (df.Group == 'O')].replace(np.nan, 0.0).reset_index().iloc[:,2:-3]
#
# df_before_j = df[(df.Time == 'before') & (df.Group == 'J')].replace(np.nan, 0.0).reset_index().iloc[:,2:-3]
# df_after_j = df[(df.Time == 'after') & (df.Group == 'J')].replace(np.nan, 0.0).reset_index().iloc[:,2:-3]
#
# plt.boxplot([df_before_o.std(axis=1),df_after_o.std(axis=1),df_before_j.std(axis=1),df_after_j.std(axis=1)])

metrics = pd.DataFrame()

def calculate_properties(size, row):
    graph = nx.from_numpy_array(np.matrix(ut.reconstruct_symmetric_matrix(size, row)))
    df_graph, global_properties = ut.graph_properties(graph)

    return {
    'Weighted Clustering Coefficient' : df_graph['Weighted Clustering Coefficient'].mean(),
    'Weighted Eigenvector Centrality' : df_graph['Weighted Eigenvector Centrality'].mean(),
    #'Weighted Closeness Centrality' : df_graph['Weighted Closeness Centrality'].mean(),
    'Weighted Density' : global_properties['Weighted Density'],
    'Assortativity (Weight Correlation)' : global_properties['Assortativity (Weight Correlation)']
    }

for g in ['O', 'J']:
    for i in ['before', 'after']:
        aux = df[(df.Time == i) & (df.Group == g)]#.replace(np.nan, 0.0).reset_index().iloc[:,2:-3]
        X_fmri = aux.select_dtypes(include=float)
        aux['STD'] = X_fmri.std(axis=1)
        aux['MEAN'] = X_fmri.mean(axis=1)

        #aux = aux.join(X_fmri.apply(calculate_properties, axis=1).apply(pd.Series))

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
def calculate_diff(group_df, feature):
    result = pd.DataFrame()

    for group in ['O', 'J']:
        # Pivot the table so that subjects are rows and 'before'/'after' are columns
        pivot_df = group_df[group_df.Group == group].pivot(index="Subject", columns="Time", values=feature)
        # Drop rows with missing values
        pivot_df = pivot_df.dropna()
        # Calculate the change in TEMP (after - before)
        pivot_df["Change"] = pivot_df["after"] - pivot_df["before"]
        pivot_df['Group'] = group

        result = pd.concat([result, pivot_df])
    return result

# Apply the function to each group using groupby
result = calculate_diff(metrics, "TEMP")

group_o = result[result.Group == 'O']
group_j = result[result.Group == 'J']

from scipy import stats
print(stats.ttest_ind(group_o['Change'], group_j['Change']))

plt.figure()
# Display the resulting DataFrame
sns.boxplot(result.reset_index(), x='Group', y='Change')
sns.swarmplot(result.reset_index(), x='Group', y='Change')
plt.ylabel('Delta T')
plt.show()

plt.figure()
ayahuasca = group_j.after
placebo = group_o.after
arr = [ayahuasca,placebo]
print(stats.ttest_ind(ayahuasca, placebo))
g = sns.boxplot(arr)
g.set_xticks(range(len(arr))) # <--- set the ticks first
g.set_xticklabels(['Ayahuasca', 'Placebo'])
plt.ylabel('Temperature')
plt.ylim(2.20, 2.55)
plt.show()


ayahuasca = pd.DataFrame(group_j.Change).reset_index()
placebo = pd.DataFrame(group_o.Change).reset_index()

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