import csv
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, AutoMinorLocator
import numpy as np
import math
import mplhep as hep
from scipy.optimize import curve_fit
from extract_data import data

# Fonts
font0 = {'family': 'DejaVu Sans', 'weight': 'bold', 'size': 38}
font1 = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 20}
font2 = {'family': 'DejaVu Sans', 'weight': 'bold', 'size': 17}

markers = ['o', '^', 's', '*', 'd']
colors = ['black', 'magenta', 'red', 'blue', 'green']
mixtures = {"30CO2": "30% CO2 + 1.0% SF6", "30CO205SF6": "30% CO2 + 0.5% SF6", "40CO2": "40% CO2 + 1.0% SF6", "STDMX": "Standard gas mixture"}

def find_index_of_value(HV, target_value):
    rounded_HV = math.ceil(target_value / 100) * 100
    try:
        return HV.index(rounded_HV)
    except ValueError:
        decremented_value = target_value - 100
        while decremented_value >= min(HV):
            rounded_decremented_value = math.ceil(decremented_value / 100) * 100
            try:
                return HV.index(rounded_decremented_value)
            except ValueError:
                decremented_value -= 100
        return None
    
def read_data(filename):
    background_rates = {}
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            scan_name, hv_str, background_rate_str, wp_str, eff_str, muoncs_str, gammacs_str, muoncs_err_str, gammacs_err_str = row
            wp = float(wp_str) / 1000
            background_rate = float(background_rate_str)
            eff = float(eff_str)
            if scan_name not in background_rates:
                background_rates[scan_name] = []
            background_rates[scan_name].append((eff, background_rate))
    return background_rates 

def plot_scan(scan):
    HV = data['HV'][scan]
    eff = data['eff'][scan]
    err = data['err'][scan]
    noise_gamma = data['noise_gamma'][scan]
    gamma_cs = data['gamma_cs'][scan]

    def func(HV, E, L, H):
        return E / (1 + np.exp(L * (H - HV)))

    initial_guess = [0.98, 0.01, 7000]
    popt, pcov = curve_fit(func, HV, eff, p0=initial_guess, bounds=([0, -np.inf, -np.inf], [1.00, np.inf, np.inf]))
    E, L, H = popt

    knee = H - (math.log(1 / 0.95 - 1)) / L
    WP = knee + 120
    eff_WP = func(WP, E, L, H)

    a = round(E * 100)
    b = WP / 1000
    c = round(eff_WP * 100)

    if scan not in plot_scan_colors:
        color = plot_scan_colors[scan] = colors[len(plot_scan_colors) % len(colors)]
    color = plot_scan_colors[scan]

    if scan not in plot_scan_markers:
        marker = plot_scan_markers[scan] = markers[len(plot_scan_markers) % len(markers)]
    marker = plot_scan_markers[scan]

    parts = scan.split("_")
    mixture = parts[0]
    d = mixtures.get(mixture, f"Gas mixture: {mixture}")

    label = f"{d}, plateau = {a} %, WP = {b:.2f} kV, Eff(WP) = {c} %"
    plt.errorbar(HV, eff, yerr=err, fmt=marker, markersize=11, color=color, label=label)

    x = np.linspace(min(HV), max(HV), 100)
    y = func(x, E, L, H)
    plt.plot(x, y, linewidth=3, color=color)
    
background_rates = read_data('2024_data_atWP.csv')
plot_scan_colors = {}
plot_scan_markers = {}

# Setting the plot
hep.style.use("CMS")
figure, ax = plt.subplots(figsize=(13, 13))
plt.xlabel(r'HV$_{\mathrm{eff}}$ [V]', fontsize=36)
plt.ylabel('Muon efficiency', fontsize=36)
plt.grid(ls='--')

ax.xaxis.set_major_locator(AutoLocator())
ax.yaxis.set_major_locator(AutoLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='major', direction='in', length=10, width=2.0, labelsize=12)
ax.tick_params(which='minor', direction='in', length=5, width=1.0)
plt.yticks(fontproperties='DejaVu Sans', size=20, weight='bold')  
plt.xticks(fontproperties='DejaVu Sans', size=20, weight='bold') 
plt.xlim(5800, 7400)
plt.ylim(0, 1.2)

# Input scan names from user
scans = input("Enter the names of the scans you want to plot, separated by commas: ").split(',')
name = input("Enter a name for the plot: ")

for scan in scans:
    plot_scan(scan.strip())

if not os.path.exists('Plots_2024/'):
    os.makedirs('Plots_2024/')

stdmx_background_rate = "N/A"
for scan in scans:
    scan = scan.strip()
    if 'STDMX' in scan:
        stdmx_background_rate = background_rates.get(scan, [(None, "N/A")])[0][1]
        break

# Plotting 
label = f"Background rate: {stdmx_background_rate if isinstance(stdmx_background_rate, float) else 'N/A'} kHz/cm²"
ax.text(0.025, 0.8, label + "\nTest Beam April 2024 \nThreshold: 60 fC \n1.4 mm double gap RPC", transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=24)

plt.text(7050, 1.21+0.01, "GIF++", font0)
hep.cms.text("Preliminary", fontsize=32)
plt.legend(loc='upper left', prop=font2, frameon=False)
plt.axhline(y=1, color='black', linestyle='--')
plt.savefig(os.path.join('Plots_2024/' + name + ".png"))

plt.show()
