import ROOT
import pandas as pd
import os
import math
from ROOT import TLine, TLegend
import numpy as np

HV_ref = 0.95  # Definição do HV de referência para HV95
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
                           np.zeros(len(df), dtype=float),  # Erro em X (substitui None)
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

def create_ABS_graph(Emax, csv_files):
    ABS = []
    for item in csv_files:
        try:
            value = item.split("/")[-1].split("_", 1)[1].replace(".csv", "")
            ABS.append(25 if value == "OFF" else float(value))
        except (IndexError, ValueError):
            ABS.append(0)

    gr2 = ROOT.TGraph(len(ABS))
    for i, (x, y) in enumerate(zip(ABS, Emax)):
        gr2.SetPoint(i, x, y)  # Corrigido: `gr2` em vez de `gr`
    
    return gr2

def process_files(csv_files, file_offset=0):
    graphs, fits = [], []
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file)
        gr = create_eff_graph(df, i + file_offset)
        sigmoid = fit_sigmoid(gr, df, i + file_offset)
        graphs.append(gr)
        fits.append(sigmoid)
    return graphs, fits

def plot_results(graphs, fits, gr2):
    c1 = ROOT.TCanvas("c1", "Efficiency vs ABS", 800, 600)
    c1.SetGrid()

    legend = ROOT.TLegend(0.15, 0.15, 0.45, 0.35)
    legend.SetHeader("Efficiency Curves", "C")

    first_graph = True
    for i, (graph, fit) in enumerate(zip(graphs, fits)):
        if first_graph:
            graph.SetTitle("Efficiency vs HV; HV (V); Efficiency")
            graph.Draw("AP")
            first_graph = False
        else:
            graph.Draw("P SAME")
        fit.Draw("SAME")
        legend.AddEntry(graph, f"Dataset {i+1}", "p")

    legend.Draw()
    c1.Draw()
    c1.SaveAs("Efficiency_vs_HV.png")

    # Criando segundo gráfico
    c2 = ROOT.TCanvas("c2", "Emax vs ABS", 800, 600)
    c2.SetGrid()
    
    gr2.SetTitle("Emax vs ABS; ABS (units); Emax")
    gr2.SetMarkerStyle(21)
    gr2.SetMarkerColor(ROOT.kBlue)
    gr2.Draw("AP")

    c2.Draw()
    c2.SaveAs("ABS_vs_EFF.png")

def main():
    data_folder_1 = "data_2024"
    data_folder_2 = "data_2024" 
    
    num_periods = int(input("Quantos períodos deseja analisar? "))
    period_files = []
    
    csv_files_1 = get_files(data_folder_1, 2024)
    csv_files_2 = get_files(data_folder_2, 2023)
    period_files.append((csv_files_1, csv_files_2))
    
    graphs, fits = [], []
    
    for period_idx, (csv_files_1, csv_files_2) in enumerate(period_files):
        print(f"Analisando arquivos do período {period_idx + 1}:")
        g1, f1 = process_files(csv_files_1, file_offset=0)
        g2, f2 = process_files(csv_files_2, file_offset=len(csv_files_1))
        
        graphs.extend(g1 + g2)
        fits.extend(f1 + f2)

    Emax, _, _, _, _ = extract_fit_parameters(fits)
    gr2 = create_ABS_graph(Emax, csv_files_1 + csv_files_2)

    plot_results(graphs, fits, gr2)
    print("Processamento concluído.")

if __name__ == "__main__":
    main()