import os
from utils import parcel, correlation_matrix, remove_triangle
import pandas as pd

path = r'/Users/rodrigo/Side-Projects/Ayahuasca/Data/Controle/'
groups = os.listdir(path)
groups.sort()
df = pd.DataFrame()

for group in groups[1]:
    subjects = os.listdir(path + group)
    subjects.sort()
    for subject in subjects[1:]:
        time = os.listdir(path + group + '/' + subject)
        time.sort()
        for t in time[1:]:
            file = os.listdir(path + group + '/' + subject + '/' + t)
            file.sort()
            if len(file) != 0:
                nii_file = [filename for filename in file if filename.endswith('.nii')]

                time_series = parcel(path + group + '/' + subject + '/' + t + '/' + nii_file[0])
                corr = correlation_matrix(time_series)
                df_aux = pd.DataFrame(remove_triangle(corr))
                df_aux['Group'] = group
                df_aux['Subject'] = subject
                df_aux['Time'] = t
                df = pd.concat([df, df_aux], axis=0)

df.to_csv('/Users/rodrigo/Side-Projects/Ayahuasca/Data/corr_matrices.csv')


