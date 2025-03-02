import os
import pandas as pd
import numpy as np
import ROOT
import math

markers = [20, 21, 22, 23, 24]  
colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen+2, ROOT.kMagenta, ROOT.kOrange+7]

def get_files(data_folder):
    mixtures = ['STDMX', '30CO2', '30CO205SF6', '40CO2']
    
    num_files = {
        "STDMX": ["STDMX_OFF.csv", "STDMX_22.csv", "STDMX_10.csv", "STDMX_6.9.csv", "STDMX_3.3.csv", "STDMX_2.2.csv", "STDMX_1.csv"],
        "30CO2": ["30CO2_OFF.csv", "30CO2_22.csv", "30CO2_10.csv", "30CO2_6.9.csv", "30CO2_3.3.csv", "30CO2_2.2.csv", "30CO2_1.csv"],
        "30CO205SF6": ["30CO205SF6_OFF.csv", "30CO205SF6_22.csv", "30CO205SF6_10.csv", "30CO205SF6_6.9.csv", "30CO205SF6_3.3.csv", "30CO205SF6_2.2.csv", "30CO205SF6_1.csv"],
        "40CO2": ["40CO2_OFF.csv", "40CO2_22.csv", "40CO2_10.csv", "40CO2_6.9.csv", "40CO2_3.3.csv"]
    }
    
    csv_files_2024 = {mixture: [] for mixture in mixtures}

    for mixture in mixtures:
        for file_name in num_files[mixture]:
            full_path = os.path.join(data_folder, file_name)
            csv_files_2024[mixture].append(full_path)
    
    return csv_files_2024

def create_eff_graph(df, index):
    gr = ROOT.TGraphErrors(len(df),
                           np.array(df['HV_top'].values, dtype=float), 
                           np.array(df['efficiency'].values, dtype=float),
                           ROOT.nullptr, 
                           np.array(df['eff_error'].values, dtype=float))
    gr.SetMarkerStyle(index % 10 + 20)
    gr.SetMarkerColor(index + 1)
    gr.SetLineColor(index + 1)
    return gr

def fit_sigmoid(graph, df, index):
    sigmoid = ROOT.TF1(f"sigmoid_{index}", "[0]/(1+ TMath::Exp(-[1]*(x-[2])))", df['HV_top'].min(), df['HV_top'].max())
    sigmoid.SetParNames("Emax", "Lambda", "HV50")
    sigmoid.SetParameters(0.9, 0.01, 7000)
    sigmoid.SetLineColor(index + 1)
    graph.Fit(sigmoid, "R")
    return sigmoid

def extract_fit_parameters(fits, HV_ref=9500):
    params = []
    for sigmoid in fits:
        Emax = sigmoid.GetParameter(0)
        Lambda = sigmoid.GetParameter(1)
        HV50 = sigmoid.GetParameter(2)
        HV95 = sigmoid.GetX(HV_ref)
        WP = HV50 - math.log(1 / 0.95 - 1) / Lambda + 150.
        params.append({'Emax': Emax, 'Lambda': Lambda, 'HV50': HV50, 'HV95': HV95, 'WP': WP})
    return params

def process_files(csv_files):
    graphs, fits = [], []
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file)
        gr = create_eff_graph(df, i)
        sigmoid = fit_sigmoid(gr, df, i)
        graphs.append(gr)
        fits.append(sigmoid)
    return fits

def extract_ABS_Emax(params, csv_files):
    Emax = {mixture: [p['Emax'] for p in param_list] for mixture, param_list in params.items()}
    ABS = {mixture: [] for mixture in csv_files.keys()}
    for mixture, files in csv_files.items():
        for file in files:
            try:
                value = file.split("/")[-1].split("_", 1)[1].replace(".csv", "")
                ABS[mixture].append(25 if value == "OFF" else float(value))
            except (IndexError, ValueError):
                ABS[mixture].append(0)
    print(Emax, ABS)
    return Emax, ABS

def plot_ABS_vs_Emax(Emax_2024, Emax_2023, ABS_2024, ABS_2023):
    # Criando o gráfico de 2024 com preenchimento
    gr_2024_STDMX = ROOT.TGraph(len(ABS_2024['STDMX']), np.array(ABS_2024['STDMX'], dtype=float), np.array(Emax_2024['STDMX'], dtype=float))
    gr_2024_30CO2 = ROOT.TGraph(len(ABS_2024['30CO2']), np.array(ABS_2024['30CO2'], dtype=float), np.array(Emax_2024['30CO2'], dtype=float))
    gr_2024_30CO205SF6 = ROOT.TGraph(len(ABS_2024['30CO205SF6']), np.array(ABS_2024['30CO205SF6'], dtype=float), np.array(Emax_2024['30CO205SF6'], dtype=float))
    gr_2024_40CO2 = ROOT.TGraph(len(ABS_2024['40CO2']), np.array(ABS_2024['40CO2'], dtype=float), np.array(Emax_2024['40CO2'], dtype=float))

    gr_2024_STDMX.SetMarkerStyle(20)
    gr_2024_STDMX.SetMarkerColor(ROOT.kBlack)
    gr_2024_STDMX.SetMarkerSize(1.2)
    gr_2024_STDMX.SetFillColor(ROOT.kBlack)
    
    gr_2024_30CO2.SetMarkerStyle(21)
    gr_2024_30CO2.SetMarkerColor(ROOT.kRed)
    gr_2024_30CO2.SetMarkerSize(1.2)
    gr_2024_30CO2.SetFillColor(ROOT.kRed)
    
    gr_2024_30CO205SF6.SetMarkerStyle(22)
    gr_2024_30CO205SF6.SetMarkerColor(ROOT.kBlue)
    gr_2024_30CO205SF6.SetMarkerSize(1.2)
    gr_2024_30CO205SF6.SetFillColor(ROOT.kBlue)
    
    gr_2024_40CO2.SetMarkerStyle(23)
    gr_2024_40CO2.SetMarkerColor(ROOT.kGreen+2)
    gr_2024_40CO2.SetMarkerSize(1.2)
    gr_2024_40CO2.SetFillColor(ROOT.kGreen+2)
    
    # Criando o gráfico de 2023 sem preenchimento
    gr_2023_STDMX = ROOT.TGraph(len(ABS_2023['STDMX']), np.array(ABS_2023['STDMX'], dtype=float), np.array(Emax_2023['STDMX'], dtype=float))
    gr_2023_30CO2 = ROOT.TGraph(len(ABS_2023['30CO2']), np.array(ABS_2023['30CO2'], dtype=float), np.array(Emax_2023['30CO2'], dtype=float))
    gr_2023_30CO205SF6 = ROOT.TGraph(len(ABS_2023['30CO205SF6']), np.array(ABS_2023['30CO205SF6'], dtype=float), np.array(Emax_2023['30CO205SF6'], dtype=float))
    gr_2023_40CO2 = ROOT.TGraph(len(ABS_2023['40CO2']), np.array(ABS_2023['40CO2'], dtype=float), np.array(Emax_2023['40CO2'], dtype=float))

    gr_2023_STDMX.SetMarkerStyle(24)
    gr_2023_STDMX.SetMarkerColor(ROOT.kBlack)
    gr_2023_STDMX.SetMarkerSize(1.2)
    
    gr_2023_30CO2.SetMarkerStyle(25)
    gr_2023_30CO2.SetMarkerColor(ROOT.kRed)
    gr_2023_30CO2.SetMarkerSize(1.2)
    
    gr_2023_30CO205SF6.SetMarkerStyle(26)
    gr_2023_30CO205SF6.SetMarkerColor(ROOT.kBlue)
    gr_2023_30CO205SF6.SetMarkerSize(1.2)
    
    gr_2023_40CO2.SetMarkerStyle(32)
    gr_2023_40CO2.SetMarkerColor(ROOT.kGreen+2)
    gr_2023_40CO2.SetMarkerSize(1.2)
    
    # Criando o Canvas
    c1 = ROOT.TCanvas("c1", "Emax vs ABS", 800, 600)
    c1.SetGrid()
    gr_2024_STDMX.Draw("AP")
    gr_2024_30CO2.Draw("P SAME")
    gr_2024_30CO205SF6.Draw("P SAME")
    gr_2024_40CO2.Draw("P SAME")
    
    gr_2023_STDMX.Draw("P SAME")
    gr_2023_30CO2.Draw("P SAME")
    gr_2023_30CO205SF6.Draw("P SAME")
    gr_2023_40CO2.Draw("P SAME")
    
    c1.SaveAs("Emax_vs_ABS_2024_vs_2023.png")
    
def main():
    data_folder_2024 = "data_2024"
    data_folder_2023 = "data_2023"
    
    csv_files_2024 = get_files(data_folder_2024)
    csv_files_2023 = get_files(data_folder_2023)

    params_2024, params_2023 = {}, {}
    
    for mixture, files in csv_files_2024.items():
        fits = process_files(files)
        params_2024[mixture] = extract_fit_parameters(fits)

    for mixture, files in csv_files_2023.items():
        fits = process_files(files)
        params_2023[mixture] = extract_fit_parameters(fits)
    
    Emax_2024, ABS_2024 = extract_ABS_Emax(params_2024, csv_files_2024)
    Emax_2023, ABS_2023 = extract_ABS_Emax(params_2023, csv_files_2023)
    
    plot_ABS_vs_Emax(Emax_2024, Emax_2023, ABS_2024, ABS_2023)
    
if __name__ == "__main__":
    main()