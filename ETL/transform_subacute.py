import os
from utils import parcel, correlation_matrix, remove_triangle, parcel_exception
import pandas as pd
import numpy as np

# Define paths
path = r'/Users/rodrigo/Documents/data/Ayahuasca/data/subacute/'
parcel_path = '/Users/rodrigo/Documents/data/INPD/Parcels_MNI_333.nii'
time_series_save_path = '/Users/rodrigo/Documents/data/Ayahuasca/data/subacute_pre_processed/time_series/'


groups = os.listdir(path)
groups.sort()
df = pd.DataFrame()

# Create directory for saving time series if it doesn't exist
os.makedirs(time_series_save_path, exist_ok=True)

for group in groups[1:]:
    subjects = os.listdir(path + group)
    subjects.sort()
    for subject in subjects[1:]:
        time = os.listdir(path + group + '/' + subject)
        time.sort()
        time = [filename for filename in time if not filename.startswith('.')]
        for t in time:
            file = os.listdir(path + group + '/' + subject + '/' + t)
            file.sort()
            if len(file) != 0:
                nii_file = [filename for filename in file if filename.endswith('.nii')]
                mov_file = [filename for filename in file if filename.endswith('RS.txt')]

                # Load the motion parameters from the file
                mov = np.loadtxt(path + group + '/' + subject + '/' + t + '/'+ mov_file[0])

                # Calculate the mean of each column
                mov = np.mean(mov, axis=0)

                # Create a DataFrame with appropriate column names
                column_names = ['Translation_X', 'Translation_Y', 'Translation_Z', 
                                'Rotation_X', 'Rotation_Y', 'Rotation_Z']

                img_path = path + group + '/' + subject + '/' + t + '/' + nii_file[0]
                if all(word in img_path for word in ["Placebo", "before"]) == True:
                    time_series = parcel_exception(img_path, parcel_path)
                else:
                    time_series = parcel(img_path, parcel_path)

                # Save the time series to a file
                time_series_filename = f"{group}_{subject}_time_series.csv"
                pd.DataFrame(time_series).to_csv(os.path.join(time_series_save_path, time_series_filename), index=False)
                
                corr = correlation_matrix(time_series)
                df_aux = pd.DataFrame(remove_triangle(corr))
                df_aux['Group'] = group
                df_aux['Subject'] = subject
                df_aux['Time'] = t
                df_aux[column_names] = mov
                df = pd.concat([df, df_aux], axis=0)


df = df.replace(np.nan, 0.0)
df.to_csv('/Users/rodrigo/Documents/data/Ayahuasca/data/subacute_pre_processed/corr_matrices_gordon.csv', index=False)


