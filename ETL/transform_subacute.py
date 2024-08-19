import os
from utils import parcel, correlation_matrix, remove_triangle, parcel_exception
import pandas as pd
import numpy as np

path = r'/Users/rodrigo/Side-Projects/Ayahuasca/Data/Controle/'
#parcel_path = '/Users/rodrigo/Side-Projects/Ayahuasca/Parcels_MNI_333.nii'
parcel_path = '/Users/rodrigo/Side-Projects/Ayahuasca/CC200.nii'

groups = os.listdir(path)
groups.sort()
df = pd.DataFrame()

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
                img_path = path + group + '/' + subject + '/' + t + '/' + nii_file[0]
                if all(word in img_path for word in ["Controle", "O", "before"]) == True:
                    time_series = parcel_exception(img_path, parcel_path)
                else:
                    time_series = parcel(img_path, parcel_path)
                corr = correlation_matrix(time_series)
                df_aux = pd.DataFrame(remove_triangle(corr))
                df_aux['Group'] = group
                df_aux['Subject'] = subject
                df_aux['Time'] = t
                df = pd.concat([df, df_aux], axis=0)


df = df.replace(np.nan, 0.0)
df.to_csv('/Users/rodrigo/Side-Projects/Ayahuasca/Data/corr_matrices_cc200.csv')


