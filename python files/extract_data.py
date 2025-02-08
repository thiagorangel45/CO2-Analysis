import csv

# # List of file names

# #2024
# file_names = ['30CO2_1', '30CO2_2.2', '30CO2_3.3', '30CO2_6.9', '30CO2_10', '30CO2_22', '30CO2_OFF', '30CO205SF6_1','30CO205SF6_2.2', '30CO205SF6_3.3', \
#                 '30CO205SF6_6.9', '30CO205SF6_10', '30CO205SF6_OFF', '40CO2_1', '40CO2_2.2', '40CO2_4.6', '40CO2_6.9', '40CO2_10', '40CO2_22',\
#                 '40CO2_100','40CO2_OFF', 'STDMX_1', 'STDMX_2.2', 'STDMX_3.3', 'STDMX_6.9', 'STDMX_10', 'STDMX_22','STDMX_100', 'STDMX_OFF', 'WP_30CO2_1',\
#                 'WP_30CO2_2.2', 'WP_30CO2_3.3', 'WP_30CO2_10', 'WP_30CO2_22', 'WP_30CO2_OFF', 'WP_30CO205SF6_1', \
#                 'WP_30CO205SF6_2.2', 'WP_30CO205SF6_3.3', 'WP_30CO205SF6_6.9', 'WP_30CO205SF6_10', 'WP_30CO205SF6_22', 'WP_30CO205SF6_OFF', 'WP_40CO2_1',\
#                 'WP_40CO2_2.2','WP_40CO2_3.3', 'WP_40CO2_4.6', 'WP_40CO2_6.9', 'WP_40CO2_10', 'WP_40CO2_22', 'WP_40CO2_OFF', 'WP_STDMX_1', 'WP_STDMX_2.2', 'WP_STDMX_3.3',\
#                 'WP_STDMX_6.9', 'WP_STDMX_10', 'WP_STDMX_22', 'WP_STDMX_OFF']

#2023
file_names = ['30CO2_1', '30CO2_2.2', '30CO2_3.3', '30CO2_4.6', '30CO2_6.9', '30CO2_10', '30CO2_22', '30CO2_OFF', '30CO205SF6_1', \
                '30CO205SF6_2.2', '30CO205SF6_3.3', '30CO205SF6_4.6', '30CO205SF6_6.9', '30CO205SF6_10', '30CO205SF6_22', '30CO205SF6_OFF', \
                '40CO2_3.3', '40CO2_4.6', '40CO2_6.9', '40CO2_10', '40CO2_22', '40CO2_OFF', 'STDMX_1', 'STDMX_2.2', 'STDMX_3.3', 'STDMX_4.6', \
                'STDMX_6.9', 'STDMX_10', 'STDMX_22', 'STDMX_OFF', 'WP_30CO2_1', 'WP_30CO2_2.2', 'WP_30CO2_3.3', 'WP_30CO2_10', 'WP_30CO2_22', 'WP_30CO2_OFF']

data = {'HV': {}, 'eff': {}, 'err': {}, 'noise_gamma': {}, 'gamma_cs': {}, 'gamma_cs_err': {}, 'muon_cs': {}, 'muon_cs_err': {}}

# Iterate over the file names
for file_name in file_names:
    run = f'{file_name}.csv'  
    data['HV'][file_name] = []
    data['eff'][file_name] = []
    data['err'][file_name] = []
    data['noise_gamma'][file_name] = []
    data['gamma_cs'][file_name] = []
    data['gamma_cs_err'][file_name] = []
    data['muon_cs'][file_name] = []
    data['muon_cs_err'][file_name] = []
    
    with open('Data_2023/' + run, "r") as csv_file:  
        csv_reader = csv.reader(csv_file)
        # Skip the header row
        next(csv_reader)

        for row in csv_reader:
            # Convert data to floats
            data['HV'][file_name].append(float(row[0]))
            data['eff'][file_name].append(float(row[14]))  
            data['err'][file_name].append(float(row[15]))
            data['noise_gamma'][file_name].append(float(row[16]))
            data['gamma_cs'][file_name].append(float(row[9]))
            data['gamma_cs_err'][file_name].append(float(row[13]))
            data['muon_cs'][file_name].append(float(row[8]))
            data['muon_cs_err'][file_name].append(float(row[12]))


# import csv
# import os

# # List of file names (2024)
# file_names = ['30CO2_1', '30CO2_2.2', '30CO2_3.3', '30CO2_6.9', '30CO2_10', '30CO2_22', '30CO2_OFF', '30CO205SF6_1', '30CO205SF6_2.2', '30CO205SF6_3.3', 
#                 '30CO205SF6_6.9', '30CO205SF6_10', '30CO205SF6_OFF', '40CO2_1', '40CO2_2.2', '40CO2_4.6', '40CO2_6.9', '40CO2_10', '40CO2_22',
#                 '40CO2_100','40CO2_OFF', 'STDMX_1', 'STDMX_2.2', 'STDMX_3.3', 'STDMX_6.9', 'STDMX_10', 'STDMX_22','STDMX_100', 'STDMX_OFF']

# def load_data():
#     data = {'HV': {}, 'eff': {}, 'err': {}, 'noise_gamma': {}, 'gamma_cs': {}, 'gamma_cs_err': {}, 'muon_cs': {}, 'muon_cs_err': {}}
    
#     for file_name in file_names:
#         if file_name.startswith('WP'):
#             continue
        
#         run = f'{file_name}.csv'  
#         data['HV'][file_name] = []
#         data['eff'][file_name] = []
#         data['err'][file_name] = []
#         data['noise_gamma'][file_name] = []
#         data['gamma_cs'][file_name] = []
#         data['gamma_cs_err'][file_name] = []
#         data['muon_cs'][file_name] = []
#         data['muon_cs_err'][file_name] = []
        
#         with open('Data_2024/' + run, "r") as csv_file:  
#             csv_reader = csv.reader(csv_file)
#             next(csv_reader)  # Skip the header row
            
#             for row in csv_reader:
#                 data['HV'][file_name].append(float(row[0]))
#                 data['eff'][file_name].append(float(row[14]))  
#                 data['err'][file_name].append(float(row[15]))
#                 data['noise_gamma'][file_name].append(float(row[16]))
#                 data['gamma_cs'][file_name].append(float(row[9]))
#                 data['gamma_cs_err'][file_name].append(float(row[13]))
#                 data['muon_cs'][file_name].append(float(row[8]))
#                 data['muon_cs_err'][file_name].append(float(row[12]))

#     return data
