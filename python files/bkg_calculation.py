import csv
import numpy as np
from extract_data import data

# # scans

# 2024 scans at WP
scans = ['WP_30CO2_1','WP_30CO2_2.2', 'WP_30CO2_3.3', 'WP_30CO2_10', 'WP_30CO2_22', 'WP_30CO2_OFF', 'WP_30CO205SF6_1',\
         'WP_30CO205SF6_2.2', 'WP_30CO205SF6_3.3', 'WP_30CO205SF6_6.9', 'WP_30CO205SF6_10', 'WP_30CO205SF6_22', 'WP_30CO205SF6_OFF',\
         'WP_40CO2_1', 'WP_40CO2_2.2','WP_40CO2_3.3', 'WP_40CO2_4.6', 'WP_40CO2_6.9', 'WP_40CO2_10', 'WP_40CO2_22', 'WP_40CO2_OFF',\
         'WP_STDMX_1', 'WP_STDMX_2.2', 'WP_STDMX_3.3', 'WP_STDMX_6.9', 'WP_STDMX_10', 'WP_STDMX_22', 'WP_STDMX_OFF']

# 2023 scans at WP
# scans = ['WP_30CO2_1', 'WP_30CO2_2.2', 'WP_30CO2_3.3', 'WP_30CO2_4.6', 'WP_30CO2_6.9', 'WP_30CO2_10', 'WP_30CO2_22', 'WP_30CO2_OFF', 'WP_30CO205SF6_1', \
#                 'WP_30CO205SF6_2.2', 'WP_30CO205SF6_3.3', 'WP_30CO205SF6_4.6', 'WP_30CO205SF6_6.9', 'WP_30CO205SF6_10', 'WP_30CO205SF6_22', 'WP_30CO205SF6_OFF', \
#                  'WP_40CO2_3.3', 'WP_40CO2_4.6', 'WP_40CO2_6.9', 'WP_40CO2_10', 'WP_40CO2_22', 'WP_40CO2_OFF', 'WP_STDMX_1', 'WP_STDMX_2.2', 'WP_STDMX_3.3',  \
#                  'WP_STDMX_4.6',  'WP_STDMX_6.9', 'WP_STDMX_10', 'WP_STDMX_22', 'WP_STDMX_OFF']


with open('background_rates_2024.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(['Scan', 'HV', 'Background Rate'])

    # Iterate over each scan
    for scan in scans:
        scan_name = scan.replace('WP_', '')
        HV = data['HV'][scan]
        noise_gamma = np.array(data['noise_gamma'][scan])
        gamma_cs = np.array(data['gamma_cs'][scan])

        # Calculate background rate
        bkg = noise_gamma / (gamma_cs * 1000)

        # Iterate over each HV value and its corresponding background rate
        for hv, background_rate in zip(HV, bkg):
            # Write the scan name, HV value, and background rate to the CSV file
            writer.writerow([scan_name, hv, background_rate])
