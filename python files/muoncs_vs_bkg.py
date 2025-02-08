import csv
import os
import re
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.ticker import AutoLocator, AutoMinorLocator

font0 = {'family': 'DejaVu Sans',
         'weight': 'bold',
         'size': 38,
         }  # "GIF++"
mixtures = {"30CO2": "64% TFE + 30% CO2 + 5% iC4H10 + 1.0% SF6",
            "30CO205SF6": "64.5% TFE + 30% CO2 + 5% iC4H10 + 0.5% SF6",
            "40CO2": "54% TFE + 40% CO2 + 5% iC4H10 + 1.0% SF6",
            "STDMX": "95.2% TFE + 4.5% iC4H10 + 0.83% SF6"
            }

colors = {
    "30CO2": 'blue',
    "30CO205SF6": 'green',
    "40CO2": 'orange',
    "STDMX": 'purple'
}

def read_data(filename):
    background_rates = {}
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            # Skip rows that don't have the expected number of columns or contain empty values
            if len(row) < 7 or any(cell == '' for cell in row[:7]):
                continue
            scan_name, hv_str, background_rate_str, wp_str, eff_str, muoncs_str, gammacs_str, muoncs_err_str, gammacs_err_str = row
            try:
                cs = float(muoncs_str)
                background_rate = float(background_rate_str)
                cs_err = float(muoncs_err_str)
            except ValueError:
                # Skip rows with invalid float conversion
                continue
            if scan_name not in background_rates:
                background_rates[scan_name] = []
            background_rates[scan_name].append((cs, background_rate, cs_err))

    return background_rates

def filter_gas_mixtures(background_rates, gas):
    gas_mixtures = {}
    for scan_name, rates in background_rates.items():
        if gas + '_' in scan_name:
            gas_mixtures[scan_name] = rates
    return gas_mixtures

def plot_background_rates(all_gas_mixtures_2023, all_gas_mixtures_2024):
    hep.style.use("CMS")
    figure, ax = plt.subplots(figsize=(12, 12))

    plotted_labels = set()

    for gas, gas_mixtures_2023 in all_gas_mixtures_2023.items():
        for scan_name, rates in gas_mixtures_2023.items():
            css, background_rates, cs_errs = zip(*rates)
            label = f'{mixtures[gas]}' if gas not in plotted_labels else '_nolegend_'
            ax.errorbar(background_rates, css, yerr=cs_errs, fmt='o', color=colors[gas], label=label, markersize=8)
            plotted_labels.add(gas)

    for gas, gas_mixtures_2024 in all_gas_mixtures_2024.items():
        for scan_name, rates in gas_mixtures_2024.items():
            css, background_rates, cs_errs = zip(*rates)
            label = f'{mixtures[gas]}' if gas not in plotted_labels else '_nolegend_'
            ax.errorbar(background_rates, css, yerr=cs_errs, fmt='o', ecolor=colors[gas], mfc='none', mec=colors[gas], label=label, markersize=8)
            plotted_labels.add(gas)

    # Set plot properties on ax object
    ax.set_ylabel('Muon cluster size', fontproperties='DejaVu Sans', size=20, weight='bold')
    ax.set_xlabel('Background Gamma Rate', fontproperties='DejaVu Sans', size=20, weight='bold')
    ax.grid(ls='--')

    ax.xaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major', direction='in', length=10, width=2.0, labelsize=12)
    ax.tick_params(which='minor', direction='in', length=5, width=1.0)

    plt.yticks(fontproperties='DejaVu Sans', size=20, weight='bold')
    plt.xticks(fontproperties='DejaVu Sans', size=20, weight='bold')
    plt.ylim(1.4, 2.6)
    plt.xlim(-0.1, 2.2)

    # Create custom legend for gas mixtures
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    legend = ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', fontsize = 17)

    # Add custom text for year indicators
    text_label = "Filled circle: 2023\nOpen circle: 2024"
    ax.text(0.75, 0.75, text_label, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=17, linespacing=1.5)

    plt.text(1.7, 2.6+0.015, "GIF++", fontdict=font0)
    hep.cms.text("Preliminary", fontsize=32)

    plt.savefig(os.path.join('Plots_23v24', "muoncs_vs_bkg.png"))

if __name__ == "__main__":
    eff_wp_calc_2023 = read_data('2023_data_atWP.csv')
    eff_wp_calc_2024 = read_data('2024_data_atWP.csv')
    
    all_gas_mixtures_2023 = {gas: filter_gas_mixtures(eff_wp_calc_2023, gas) for gas in mixtures.keys()}
    all_gas_mixtures_2024 = {gas: filter_gas_mixtures(eff_wp_calc_2024, gas) for gas in mixtures.keys()}
    
    plot_background_rates(all_gas_mixtures_2023, all_gas_mixtures_2024)