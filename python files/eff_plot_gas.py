import csv
import os
import matplotlib.pyplot as plt 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoLocator, AutoMinorLocator
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import math
import mplhep as hep
from extract_data import data


""""
Before running the scripit: 
- change text according to gas mixture being used
- double check if TB from 2023/24 and change path directory accordingly
"""

#Fonts
font0 = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 34,
            } # for plot title
font1 = {'family': 'DejaVu Sans',
            'weight': 'normal',
            'size': 20,
            } # for text info
font2 = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 15,
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
        header = next(reader)  # Pular a linha de cabeçalho
        
        # Verifique o número de colunas no cabeçalho
        print(f"Colunas no cabeçalho: {header}")

        for row in reader:
            # Imprima a linha atual para ver seu conteúdo
            print(f"Linha atual: {row}")

            if len(row) < 9:  # Verifique se há pelo menos 9 colunas
                print(f"Advertência: Linha com dados insuficientes: {row}")
                continue  # Pular linhas que não têm dados suficientes

            scan_name, hv_str, background_rate_str, wp_str, eff_str, muoncs_str, gammacs_str, muoncs_err_str, gammacs_err_str = row
            
            try:
                wp = float(wp_str) / 1000
                background_rate = float(background_rate_str)
            except ValueError as e:
                print(f"Erro ao converter valores: {e}")
                continue

            if scan_name not in background_rates:
                background_rates[scan_name] = []
            background_rates[scan_name].append((wp, background_rate))
    return background_rates


def find_index_of_value(HV, target_value):
    rounded_HV = math.ceil(target_value / 100) * 100
    try:
        return HV.index(rounded_HV)
    except ValueError:
        print(f"Value {rounded_HV} not found in HV column.")
        decremented_value = target_value - 100
        while decremented_value >= min(HV):
            rounded_decremented_value = math.ceil(decremented_value / 100) * 100
            try:
                return HV.index(rounded_decremented_value)
            except ValueError:
                print(f"Value {rounded_decremented_value} not found in HV column.")
                decremented_value -= 100
        print("No corresponding value found in HV column.")
        return None
        

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
    
    gamma_noise = None
    index_WP = find_index_of_value(HV, WP)

    if index_WP is not None:
        eff_WP = eff[index_WP]
        gamma_noise = noise_gamma[index_WP]
        #gamma_c = gamma_cs[index_WP]
    else:
        print("WP outside the HV range")

    if gamma_noise is not None: 
        a = round(E * 100)
        b = WP / 1000
        c = round(eff_WP * 100) 
        #d = gamma_noise / (gamma_c * 1000) 
    else:
        print("Background rate value not available") 

    if scan not in plot_scan_colors:
        color = plot_scan_colors[scan] = colors[len(plot_scan_colors) % len(colors)]
    color = plot_scan_colors[scan]

    if scan not in plot_scan_markers:
        marker = plot_scan_markers[scan] = markers[len(plot_scan_markers) % len(markers)]
    marker = plot_scan_markers[scan]

    labels = []
    for wp, background_rate in background_rates.get(scan, []):
        labels.append(f"bkg rate at WP= {background_rate:.2f} kHz/cm2, WP = {b:.2f} kV, Eff(WP) = {c} %")
    label = '\n'.join(labels)
    plt.errorbar(HV, eff, yerr=err, fmt=marker, markersize=11, color=color, label=label)

    # if d < 0.01:
    #     label = f"no bkg rate, plateau = {a} %, WP = {b:.2f} kV, Eff(WP) = {c} %"
    #     plt.errorbar(HV, eff, yerr=err, fmt=marker, markersize=11, color=color, label=label)
    # else:
    #     label = f"bkg rate = {d:.1f} Hz/cm2, plateau = {a} %, WP = {b:.2f} kV, Eff(WP) = {c} %"
    #     plt.errorbar(HV, eff, yerr=err, fmt=marker, markersize=11, color=color, label=label)

    x = np.linspace(min(HV), max(HV), 100)
    y = func(x, E, L, H)
    plt.plot(x, y, linewidth=3, color=color)

background_rates = read_background_rates('background_rates_2024.csv')
plot_scan_colors = {}
plot_scan_markers = {}

#Setting the plot
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
plt.xlim(5800, 7300)
plt.ylim(0, 1.2)

# Input scan names from user
scans = input("Enter the names of the scans you want to plot, separated by commas: ").split(',')
#name =  datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
name = input("Enter a name for the plot: ")

for scan in scans:
    plot_scan(scan.strip())

if not os.path.exists('Plots_2024/'):
    os.makedirs('Plots_2024/')

# Plotting 
prefixes = set(scan.split('_')[0].strip() for scan in scans)
mixture_labels = [mixtures.get(prefix, f"Unknown Mixture ({prefix})") for prefix in prefixes]

# Generate label based on scan names
label = " + ".join(set(mixture_labels))
label += "\nTest Beam April 2024 \nThreshold: 60 fC \n1.4 mm double gap RPC"
ax.text(0.035, 0.78, label, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=19, linespacing=1.5)

plt.text(7050, 1.21+0.01, "GIF++", font0)
hep.cms.text("Preliminary", fontsize=32)
plt.legend(loc='upper left', prop=font2, frameon=False)
plt.axhline(y=1, color='black', linestyle='--')
plt.savefig(os.path.join('Plots_2024/' + name + ".png"))

plt.show()



