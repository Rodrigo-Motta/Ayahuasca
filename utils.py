import numpy as np
import pandas as pd
import os
import nibabel as nib
from nilearn import image as nimg
from nilearn import input_data

def parcel(file_path):
    # reading Gordon parcellation
    GOR = nib.load('/Users/rodrigo/Side-Projects/Ayahuasca/Parcels_MNI_333.nii')

    test_load = nib.load(file_path)
    #
    # Print dimensions of functional image and atlas image

    print("Size of functional image:", test_load.shape)
    print("Size of atlas image:", GOR.shape)

    resampled_GOR = nimg.resample_to_img(GOR, test_load, interpolation = 'nearest', fill_value=0)

    print('Resampled GOR', resampled_GOR.shape)

    resampled = nimg.resample_img(GOR, test_load.affine,target_shape=test_load.get_fdata()[:,:,:,0].shape, interpolation = 'nearest', fill_value=0)

    # Get the label numbers from the atlas
    atlas_labels = np.unique(resampled_GOR.get_fdata().astype(int))

    # Get number of labels that we have
    NUM_LABELS = len(atlas_labels)
    #print('Num labels', NUM_LABELS)

    masker = input_data.NiftiLabelsMasker(labels_img=resampled_GOR,
                                          standardize=True,
                                          verbose=1,
                                          detrend=True)

    cleaned_and_averaged_time_series = masker.fit_transform(test_load)

    # Get the label numbers from the atlas
    atlas_labels = np.unique(resampled_GOR.get_fdata().astype(int))

    # Get number of labels that we have
    NUM_LABELS = len(atlas_labels)

    if NUM_LABELS != 333:

        # Remember fMRI images are of size (x,y,z,t)
        # where t is the number of timepoints
        num_timepoints = test_load.shape[3]

        # Create an array of zeros that has the correct size
        final_signal = np.zeros((num_timepoints, 333 + 1))  # NUM_LABELS))

        # Get regions that are kept
        regions_kept = np.array(masker.labels_).astype(int)

        # Fill columns matching labels with signal values
        final_signal[:, regions_kept] = cleaned_and_averaged_time_series

        # Excluding ROI = 0 that does not exist
        final_signal = final_signal[:, 1:]

        return final_signal

    else:
        return pd.DataFrame(cleaned_and_averaged_time_series)


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