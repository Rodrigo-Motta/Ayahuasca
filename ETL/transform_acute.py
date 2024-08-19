import os
from utils import parcel, correlation_matrix, remove_triangle, parcel_exception
import pandas as pd
import numpy as np

# Define paths
path = r'/Users/rodrigo/Documents/data/Ayahuasca/data/ayahuasca_acute_preacute/Ayahuasca_acute/func/'
parcel_path = '/Users/rodrigo/Documents/data/INPD/Parcels_MNI_333.nii'
time_series_save_path = '/Users/rodrigo/Documents/data/Ayahuasca/data/acute_pre_processed/time_series/'

# Create directory for saving time series if it doesn't exist
os.makedirs(time_series_save_path, exist_ok=True)

groups = os.listdir(path)
groups.sort()
df = pd.DataFrame()

for group in groups[1:]:
    subjects = os.listdir(path + group)
    subjects.sort()
    for subject in subjects[1:]:
        file = os.listdir(path + group + '/' + subject)
        file.sort()
        nii_file = [filename for filename in file if filename.endswith('.nii')]
        mov_file = [filename for filename in file if filename.endswith('trans.txt')]
        img_path = path + group + '/' + subject + '/' + nii_file[0]
                
        # Load the motion parameters from the file
        mov = np.loadtxt(path + group + '/' + subject + '/' + mov_file[0])

        # Calculate the mean of each column
        mov = np.mean(mov, axis=0)

        # Create a DataFrame with appropriate column names
        column_names = ['Translation_X', 'Translation_Y', 'Translation_Z', 
                        'Rotation_X', 'Rotation_Y', 'Rotation_Z']

        # Generate time series
        time_series = parcel(img_path, parcel_path)
        
        # Save the time series to a file
        time_series_filename = f"{group}_{subject}_time_series.csv"
        pd.DataFrame(time_series).to_csv(os.path.join(time_series_save_path, time_series_filename), index=False)
        
        # Calculate correlation matrix
        corr = correlation_matrix(time_series)

        df_aux = pd.DataFrame(remove_triangle(corr))
        df_aux['Group'] = group
        df_aux['Subject'] = subject
        df_aux[column_names] = mov
        df = pd.concat([df, df_aux], axis=0)

# Replace NaNs with 0.0
df = df.replace(np.nan, 0.0)

# Save the correlation matrices to a CSV file
df.to_csv('/Users/rodrigo/Documents/data/Ayahuasca/data/acute_pre_processed/corr_matrices_gordon.csv', index=False)


