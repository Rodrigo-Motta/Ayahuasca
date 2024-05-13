import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image as nimg
from nilearn import input_data
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import coo_matrix
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import networkx as nx


def parcel(file_path, parcel_path):

    # reading parcellation
    parcel = nib.load(parcel_path)

    parcel_dim = len(np.unique(parcel.get_fdata())) - 1


    t_r = 2

    img = nib.load(file_path).slicer[:,:,:,int(t_r*2):]
    #
    # Print dimensions of functional image and atlas image

    print("Size of functional image:", img.shape)
    print("Size of atlas image:", parcel.shape)

    resampled_parcel = nimg.resample_to_img(parcel, img, interpolation = 'nearest', fill_value=0)
    #resampled_GOR = nimg.resample_img(GOR, affine, target_shape=test_load.get_fdata()[:,:,:,0].shape,
    #                                  interpolation = 'nearest', fill_value=0)

    print('Resampled GOR', resampled_parcel.shape)

    # Get the label numbers from the atlas
    atlas_labels = np.unique(resampled_parcel.get_fdata().astype(int))

    # Get number of labels that we have
    NUM_LABELS = len(atlas_labels)
    #print('Num labels', NUM_LABELS)

    masker = input_data.NiftiLabelsMasker(labels_img=resampled_parcel,
                                          standardize=True,
                                          verbose=1,
                                          detrend=True,
                                          low_pass=0.08,
                                          high_pass=0.009,
                                          t_r=2)

    cleaned_and_averaged_time_series = masker.fit_transform(img)

    # Get the label numbers from the atlas
    atlas_labels = np.unique(resampled_parcel.get_fdata().astype(int))

    # Get number of labels that we have
    NUM_LABELS = len(atlas_labels)

    if NUM_LABELS != parcel_dim:

        # Remember fMRI images are of size (x,y,z,t)
        # where t is the number of timepoints
        num_timepoints = img.shape[3]

        # Create an array of zeros that has the correct size
        final_signal = np.zeros((num_timepoints, parcel_dim + 1))  # NUM_LABELS))

        # Get regions that are kept
        regions_kept = np.array(masker.labels_).astype(int)

        # Fill columns matching labels with signal values
        final_signal[:, regions_kept] = cleaned_and_averaged_time_series

        # Excluding ROI = 0 that does not exist
        final_signal = final_signal[:, 1:]

        return pd.DataFrame(final_signal).replace(np.nan, 0.0)

    else:
        return pd.DataFrame(cleaned_and_averaged_time_series).replace(np.nan,0.0)


def parcel_exception(file_path, parcel_path):
    # reading Gordon parcellation
    parcel = nib.load(parcel_path)

    parcel_dim = len(np.unique(parcel.get_fdata())) - 1

    t_r = 2

    img = nib.load(file_path).slicer[:,:,:,int(t_r*2):]

    solver = nib.load('/Users/rodrigo/Side-Projects/Ayahuasca/Data/Controle/O/6/before/swau6_RS.nii').slicer[:,:,:,2:]

    # Access the image data and header
    data = img.get_fdata()
    header = solver.header

    # Modify the description field in the header
    header['descrip'] = 'Modified image header'

    # Save the modified image with the new header
    modified_image = nib.Nifti1Image(data, solver.affine, header)

    resampled_parcel = nimg.resample_to_img(parcel, modified_image, interpolation='nearest', fill_value=0)
    #
    # Print dimensions of functional image and atlas image

    print("Size of functional image:", img.shape)
    print("Size of atlas image:", parcel.shape)

    print('Resampled GOR', resampled_parcel.shape)

    # Get the label numbers from the atlas
    atlas_labels = np.unique(resampled_parcel.get_fdata().astype(int))

    # Get number of labels that we have
    NUM_LABELS = len(atlas_labels)
    #print('Num labels', NUM_LABELS)

    masker = input_data.NiftiLabelsMasker(labels_img=resampled_parcel,
                                          standardize=True,
                                          verbose=1,
                                          detrend=True,
                                          low_pass=0.08,
                                          high_pass=0.009,
                                          t_r=2)

    cleaned_and_averaged_time_series = masker.fit_transform(modified_image)

    # Get the label numbers from the atlas
    atlas_labels = np.unique(resampled_parcel.get_fdata().astype(int))

    # Get number of labels that we have
    NUM_LABELS = len(atlas_labels)

    if NUM_LABELS != parcel_dim:

        # Remember fMRI images are of size (x,y,z,t)
        # where t is the number of timepoints
        num_timepoints = modified_image.shape[3]

        # Create an array of zeros that has the correct size
        final_signal = np.zeros((num_timepoints, parcel_dim + 1))  # NUM_LABELS))

        # Get regions that are kept
        regions_kept = np.array(masker.labels_).astype(int)

        # Fill columns matching labels with signal values
        final_signal[:, regions_kept] = cleaned_and_averaged_time_series

        # Excluding ROI = 0 that does not exist
        final_signal = final_signal[:, 1:]

        return pd.DataFrame(final_signal).replace(np.nan, 0.0)

    else:
        return pd.DataFrame(cleaned_and_averaged_time_series).replace(np.nan,0.0)

def correlation_matrix(df_time_series):
    return pd.DataFrame(df_time_series).corr()#.values.ravel()

def remove_triangle(df):
    # Remove triangle of a symmetric matrix and the diagonal

    df = df.astype(float)
    df.values[np.triu_indices_from(df, k=1)] = np.nan
    df = ((df.T).values.reshape((1, (df.shape[0]) ** 2)))
    df = df[~np.isnan(df)]
    df = df[df != 1]
    return (df).reshape((1, len(df)))

def reconstruct_symmetric_matrix(size, upper_triangle_array, diag=1):

    result = np.zeros((size, size))
    result[np.triu_indices_from(result, 1)] = upper_triangle_array
    result = result + result.T
    np.fill_diagonal(result, diag)
    return result

def compute_KNN_graph(matrix, k_degree=10):
    '''
    Calculate the adjacency matrix from the connectivity matrix
    '''

    matrix = np.abs(matrix)
    idx = np.argsort(-matrix)[:, 0:k_degree]
    matrix.sort()
    matrix = matrix[:, ::-1]
    matrix = matrix[:, 0:k_degree]

    A = adjacency(matrix, idx).astype(np.float32)

    return A


def adjacency(dist, idx):
    m, k = dist.shape
    assert m, k == idx.shape
    assert dist.min() >= 0

    # Weight matrix.
    I = np.arange(0, m).repeat(k)
    J = idx.reshape(m * k)
    V = dist.reshape(m * k)
    W = coo_matrix((V, (I, J)), shape=(m, m))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    return W.todense()


def create_graph(X_train, X_test, y_train, y_test, size=190, method={'knn': 10}):
    train_data = []
    val_data = []

    # Creating train data in pyG DATA structure
    for i in range((X_train.shape[0])):

        # Transforming into a correlation matrix
        Adj = reconstruct_symmetric_matrix(size, X_train.iloc[i, :].values)

        # Copying the Adj matrix for operations to define edge_index
        A = Adj.copy()

        Adj = torch.from_numpy(Adj).float()

        if method == None:
            A = A

        elif list(method.keys())[0] == 'knn':
            # Using k-NN to define Edges
            A = compute_KNN_graph(A, method['knn'])

        elif list(method.keys())[0] == 'threshold':
            A[A < method['threshold']] = 0
            Adj[Adj < method['threshold']] = 0

        elif list(method.keys())[0] == 'knn_group':
            A = method['knn_group']

        # Removing self connections
        np.fill_diagonal(A, 0)
        A = torch.from_numpy(A).float()

        # getting the edge_index
        edge_index_A, edge_attr_A = dense_to_sparse(A)

        train_data.append(Data(x=Adj, edge_index=edge_index_A, edge_attr=edge_attr_A.reshape(len(edge_attr_A), 1),
                               y=torch.tensor(float(y_train.iloc[i]))))

    # Creating test data in pyG DATA structure
    for i in range((X_test.shape[0])):

        # Transforming into a correlation matrix
        Adj = reconstruct_symmetric_matrix(size, X_test.iloc[i, :].values)

        # Copying the Adj matrix for operations to define edge_index
        A = Adj.copy()

        Adj = torch.from_numpy(Adj).float()

        if method == None:
            A = A

        elif list(method.keys())[0] == 'knn':
            # Using k-NN to define Edges
            A = compute_KNN_graph(A, method['knn'])

        elif list(method.keys())[0] == 'threshold':
            A[A < method['threshold']] = 0
            Adj[Adj < method['threshold']] = 0

        elif list(method.keys())[0] == 'knn_group':
            A = method['knn_group']

        # Removing self connections
        np.fill_diagonal(A, 0)
        A = torch.from_numpy(A).float()

        # getting the edge_index
        edge_index_A, edge_attr_A = dense_to_sparse(A)

        val_data.append(Data(x=Adj, edge_index=edge_index_A, edge_attr=edge_attr_A.reshape(len(edge_attr_A), 1),
                             y=torch.tensor(float(y_test.iloc[i]))))

    return train_data, val_data


def create_batch(train_data, val_data, batch_size):
    train_loader = DataLoader(train_data, batch_size)  # Shuffle=True

    val_loader = DataLoader(val_data)  # Shuffle=True

    return train_loader, val_loader

def graph_properties(graph):
    # Assume the graph is already loaded or created
    # `graph` is a NetworkX graph where the edge weights represent semantic similarity

    # Weighted Degree (Strength)
    weighted_degrees = dict(graph.degree(weight='weight'))

    # Weighted Clustering Coefficient
    weighted_clustering_coefficients = nx.clustering(graph, weight='weight')

    # Weighted Betweenness Centrality
    #weighted_betweenness = nx.betweenness_centrality(graph, weight='weight')

    # Weighted Closeness Centrality
    #weighted_closeness = nx.closeness_centrality(graph, distance='weight')

    # Weighted Modularity (Community Detection via Louvain method)
    #partition = community_louvain.best_partition(graph, weight='weight')

    # Weighted Eigenvector Centrality
    weighted_eigenvector = nx.eigenvector_centrality(graph, weight='weight')

    # Combine individual properties into a DataFrame
    df = pd.DataFrame({
        'Weighted Degree': weighted_degrees,
        'Weighted Clustering Coefficient': weighted_clustering_coefficients,
        #'Weighted Betweenness Centrality': weighted_betweenness,
        #'Weighted Closeness Centrality': weighted_closeness,
        #'Community (Partition)': partition,
        'Weighted Eigenvector Centrality': weighted_eigenvector
    })

    # Adding global metrics that are not node-specific
    total_weight = sum(data['weight'] for u, v, data in graph.edges(data=True))
    num_possible_edges = len(graph) * (len(graph) - 1) / 2
    weighted_density = total_weight / num_possible_edges
    weight_correlations = nx.degree_pearson_correlation_coefficient(graph, weight='weight')

    global_metrics = {
        'Average Weighted Degree': sum(weighted_degrees.values()) / len(graph),
        'Weighted Density': weighted_density,
        'Assortativity (Weight Correlation)': weight_correlations
    }


    return df, global_metrics
