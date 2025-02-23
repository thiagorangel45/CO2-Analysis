import ROOT
import pandas as pd
import os
import math
from ROOT import TLine, TLegend
import numpy as np

HV_ref = 95  
colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen+2, ROOT.kMagenta, ROOT.kOrange, ROOT.kCyan, ROOT.kBlack]
markers = [20, 21, 22, 23, 24, 25, 26]

def get_files(data_folder, year):
    num_files = int(input(f"Quantos arquivos deseja analisar para o ano {year}? "))
    csv_files = []
    
    for j in range(num_files):
        file_name = input(f"Digite o nome do arquivo {j+1} do ano {year} (ex: STDMX.csv): ")
        full_path = os.path.join(data_folder, file_name)
        
        if not os.path.isfile(full_path):
            print(f"Erro: Arquivo '{file_name}' não encontrado. Pulando...")
        else:
            csv_files.append(full_path)
    
    return csv_files

def create_eff_graph(df, index):
    gr = ROOT.TGraphErrors(len(df),
                           np.array(df['HV_top'].values, dtype=float), 
                           np.array(df['efficiency'].values, dtype=float),
                           ROOT.nullptr, 
                           np.array(df['eff_error'].values, dtype=float))
    gr.SetMarkerStyle(markers[index % len(markers)])
    gr.SetMarkerColor(colors[index % len(colors)])
    gr.SetLineColor(colors[index % len(colors)])
    return gr

def fit_sigmoid(graph, df, index):
    sigmoid = ROOT.TF1(f"sigmoid_{index}", "[0]/(1+ TMath::Exp(-[1]*(x-[2])))", df['HV_top'].min(), df['HV_top'].max())
    sigmoid.SetParNames("Emax", "Lambda", "HV50")
    sigmoid.SetParameters(0.9, 0.01, 7000)
    sigmoid.SetLineColor(colors[index % len(colors)])
    graph.Fit(sigmoid, "R")
    return sigmoid

def extract_fit_parameters(fits):
    Emax, Lambda, HV50, HV95, WP = [], [], [], [], []
    for sigmoid in fits:
        Emax.append(sigmoid.GetParameter(0))
        Lambda.append(sigmoid.GetParameter(1))
        HV50.append(sigmoid.GetParameter(2))
        HV95.append(sigmoid.GetX(HV_ref))
        WP.append(HV50[-1] - math.log(1 / 0.95 - 1) / Lambda[-1] + 150.)
    return Emax, Lambda, HV50, HV95, WP

def create_ABS_graph(Emax_2024, Emax_2023, csv_files_2024, csv_files_2023):
    ABS_2024, ABS_2023 = [], []

    for item in csv_files_2024:
        try:
            value = item.split("/")[-1].split("_", 1)[1].replace(".csv", "")
            ABS_2024.append(25 if value == "OFF" else float(value))
        except (IndexError, ValueError):
            ABS_2024.append(0)

    for item in csv_files_2023:
        try:
            value = item.split("/")[-1].split("_", 1)[1].replace(".csv", "")
            ABS_2023.append(25 if value == "OFF" else float(value))
        except (IndexError, ValueError):
            ABS_2023.append(0)

    gr_2024 = ROOT.TGraph(len(ABS_2024))
    gr_2023 = ROOT.TGraph(len(ABS_2023))

    for i, (x, y) in enumerate(zip(ABS_2024, Emax_2024)):
        gr_2024.SetPoint(i, x, y)

    for i, (x, y) in enumerate(zip(ABS_2023, Emax_2023)):
        gr_2023.SetPoint(i, x, y)

    return gr_2024, gr_2023

def process_files(csv_files, file_offset=0):
    graphs, fits = [], []
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file)
        gr = create_eff_graph(df, i + file_offset)
        sigmoid = fit_sigmoid(gr, df, i + file_offset)
        graphs.append(gr)
        fits.append(sigmoid)
    return graphs, fits

def plot_results(gr_2024, gr_2023):
    c2 = ROOT.TCanvas("c2", "Emax vs ABS", 800, 600)

    gr_2024.SetMarkerStyle(21)
    gr_2024.SetMarkerColor(ROOT.kBlue)
    gr_2024.GetXaxis().SetTitle("ABS")
    gr_2024.GetYaxis().SetTitle("Efficiency")

    gr_2023.SetMarkerStyle(22)
    gr_2023.SetMarkerColor(ROOT.kRed)

    gr_2024.Draw("AP")
    gr_2023.Draw("P same")

    legend = ROOT.TLegend(0.7, 0.2, 0.9, 0.3)
    legend.AddEntry(gr_2024, "2024", "p")
    legend.AddEntry(gr_2023, "2023", "p")
    legend.Draw()

    c2.Draw()
    c2.SaveAs("ABS_vs_EFF.png")

def main():
    data_folder_1 = "data_2024"
    data_folder_2 = "data_2023" 
    
    csv_files_2024 = get_files(data_folder_1, 2024)
    csv_files_2023 = get_files(data_folder_2, 2023)
    
    graphs_2024, fits_2024 = process_files(csv_files_2024)    
    graphs_2023, fits_2023 = process_files(csv_files_2023, file_offset=len(csv_files_2024))
    
    Emax_2024, _, _, _, _ = extract_fit_parameters(fits_2024)
    Emax_2023, _, _, _, _ = extract_fit_parameters(fits_2023)
    gr_2024, gr_2023 = create_ABS_graph(Emax_2024, Emax_2023, csv_files_2024, csv_files_2023)
    plot_results(gr_2024, gr_2023)

    print("Processamento concluído.")

if __name__ == "__main__":
    main()
