import csv
from extract_data import data
import numpy as np
import math
import matplotlib.pyplot as plt 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoLocator, AutoMinorLocator
import os
import mplhep as hep
from datetime import datetime
from scipy.optimize import curve_fit


font0 = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 38,
            } # for plot title
font1 = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 16,
            } # for legend
markers = ['o', '^', 's', '*', 'd']
colors = ['black', 'magenta', 'red', 'blue', 'green']
mixtures = {"30CO2" : "64% TFE + 30% CO2 + 5% iC4H10 + 1.0% SF6",
            "30CO205SF6" : "64.5% TFE + 30% CO2 + 5% iC4H10 + 0.5% SF6",
            "40CO2" : "54% TFE + 40% CO2 + 5% iC4H10 + 1.0% SF6",
            "STDMX" : "95.2% TFE + 4.5% iC4H10 + 0.83% SF6"
}

def read_background_rates(filename):
    background_rates = {}
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            scan_name, hv_str, background_rate_str, wp_str, eff_str, muoncs_str, gammacs_str, muoncs_err_str, gammacs_err_str = row
            wp = float(wp_str) / 1000
            background_rate = float(background_rate_str)
            if scan_name not in background_rates:
                background_rates[scan_name] = []
            background_rates[scan_name].append((wp, background_rate))
    return background_rates

def plot_scan(scan, wp):
    HV = np.array(data['HV'][scan])
    eff = data['eff'][scan]
    noise_gamma = np.array(data['noise_gamma'][scan])
    gamma_cs = np.array(data['gamma_cs'][scan])
    gamma_cs_err =  np.array(data['gamma_cs_err'][scan]) 
    wp = background_rates[scan][0][0]  # Get the wp for the current scan
    HV_adj = HV - (wp*1000) 

    if scan not in plot_scan_colors:
        color = plot_scan_colors[scan] = colors[len(plot_scan_colors) % len(colors)]
    color = plot_scan_colors[scan]

    if scan not in plot_scan_markers:
        marker = plot_scan_markers[scan] = markers[len(plot_scan_markers) % len(markers)]
    marker = plot_scan_markers[scan]

    labels = []
    for wp, background_rate in background_rates.get(scan, []):
        labels.append(f"WP = {wp} [kV], Background rate at WP= {background_rate:.2f} kHz/cm2")
    label = '\n'.join(labels)
    plt.errorbar(HV_adj, gamma_cs, yerr=gamma_cs_err, fmt=marker, markersize=11, color=color, label=label)

background_rates = read_background_rates('background_rates_2024.csv')
plot_scan_colors = {}
plot_scan_markers = {}

#Setting the plot
hep.style.use("CMS")
figure, ax = plt.subplots(figsize=(14, 14))
plt.xlabel(r'HV$_{\mathrm{eff}}$ - HV$_{\mathrm{WP}}$ [V]', fontsize=38)
plt.ylabel('Gamma cluster size', fontsize=38)
plt.grid(ls='--')

ax.xaxis.set_major_locator(AutoLocator())
ax.yaxis.set_major_locator(AutoLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='major', direction='in', length=10, width=2.0, labelsize=12)
ax.tick_params(which='minor', direction='in', length=5, width=1.0)
plt.yticks(fontproperties='DejaVu Sans', size=20, weight='bold')  
plt.xticks(fontproperties='DejaVu Sans', size=20, weight='bold') 
plt.xlim(-1050, 600)
plt.ylim(0.8, 2.75)

# Input scan names from user
scans = input("Enter the names of the scans you want to plot, separated by commas: ").split(',')
#name =  datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
name = input("Enter a name for the plot: ")

for scan in scans:
    wp = background_rates.get(scan, [(None, None)])[0][0]
    plot_scan(scan.strip(), wp)

if not os.path.exists('Plots_2024/'):
    os.makedirs('Plots_2024/')

# Plotting 
prefixes = set(scan.split('_')[0].strip() for scan in scans)
mixture_labels = [mixtures.get(prefix, f"Unknown Mixture ({prefix})") for prefix in prefixes]

# Generate label based on scan names
label = " + ".join(set(mixture_labels))
label += "\nTest Beam April 2024  \nThreshold: 60 fC \n1.4 mm double gap RPC"
ax.text(0.035, 0.78, label, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=22, linespacing=1.5)

plt.text(250, 2.75+0.03, "GIF++", font0)
hep.cms.text("Preliminary", fontsize=32)
plt.legend(loc='upper left', prop=font1, frameon=False)
#plt.axhline(y=2.5, color='black', linestyle='--')
plt.savefig(os.path.join('Plots_2024/' + name + ".png"))

#plt.show()

