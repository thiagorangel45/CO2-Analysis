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
    csv_WP_files = []
    
    for j in range(num_files):
        file_name = input(f"Digite o nome do arquivo {j+1} do ano {year} (ex: STDMX.csv): ")
        full_path = os.path.join(data_folder, file_name)
        
        if not os.path.isfile(full_path):
            print(f"Erro: Arquivo '{file_name}' não encontrado. Pulando...")
        else:
            csv_files.append(full_path)
            base_name, ext = os.path.splitext(file_name)
            csv_WP_files.append(os.path.join(data_folder, f"{base_name}_WP{ext}"))
        
    return csv_files, csv_WP_files

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

def process_files(csv_files, file_offset=0):
    graphs, fits = [], []
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file)
        gr = create_eff_graph(df, i + file_offset)
        sigmoid = fit_sigmoid(gr, df, i + file_offset)
        graphs.append(gr)
        fits.append(sigmoid)
    return graphs, fits


def extract_bkg(csv_WP_files):
    bkg = []
    for file in csv_WP_files:
        df = pd.read_csv(file)
        # Garantir que estamos acessando as colunas corretamente
        if 'noiseGammaRate' in df.columns and 'gamma_CS' in df.columns:
            # Calcular o valor de bkg
            bkg_value = df['noiseGammaRate'] / (df['gamma_CS'] * 1000)
            # Assumir que estamos interessados apenas em valores escalares, então pegar o primeiro valor
            bkg.append(bkg_value.iloc[0])  # Acessar o primeiro valor
        else:
            print(f"Colunas 'noiseGammaRate' ou 'gamma_CS' não encontradas no arquivo {file}.")
    return bkg



def create_bkg_graph(Emax_2024, Emax_2023, bkg_2024, bkg_2023):
    gr_2024 = ROOT.TGraph(len(Emax_2024))
    gr_2023 = ROOT.TGraph(len(Emax_2023))

    # Garantir que os dados sejam arrays numpy de tipo float
    Emax_2024 = np.array(Emax_2024, dtype=float)
    Emax_2023 = np.array(Emax_2023, dtype=float)
    bkg_2024 = np.array(bkg_2024, dtype=float)
    bkg_2023 = np.array(bkg_2023, dtype=float)

    print(Emax_2024, Emax_2023, bkg_2024, bkg_2023)

    for i in range(len(bkg_2024)):
        x = float(bkg_2024[i])  
        y = float(Emax_2024[i])  
        gr_2024.SetPoint(i, x, y)  

    for i in range(len(bkg_2023)):
        x = float(bkg_2023[i])  
        y = float(Emax_2023[i])  
        gr_2023.SetPoint(i, x, y) 

    return gr_2024, gr_2023



def plot_results(gr_2024, gr_2023):
    c2 = ROOT.TCanvas("c2", "Emax vs ABS", 800, 600)

    gr_2024.SetMarkerStyle(21)
    gr_2024.SetMarkerColor(ROOT.kBlue)
    gr_2024.GetXaxis().SetTitle("ABS")
    gr_2024.GetYaxis().SetTitle("Efficiency")

    gr_2023.SetMarkerStyle(22)
    gr_2023.SetMarkerColor(ROOT.kRed)

    gr_2024.GetXaxis().SetRangeUser(0, 2.5)
    gr_2023.GetXaxis().SetRangeUser(0, 2.5)
    gr_2024.GetYaxis().SetRangeUser(0.88, 1)
    gr_2023.GetYaxis().SetLimits(0.88, 1)

    gr_2024.Draw("AP")
    gr_2023.Draw("P same")

    legend = ROOT.TLegend(0.7, 0.2, 0.9, 0.3)
    legend.AddEntry(gr_2024, "2024", "p")
    legend.AddEntry(gr_2023, "2023", "p")
    legend.Draw()

    c2.Draw()
    c2.SaveAs("bkg_vs_EFF.png")

def main():
    data_folder_1 = "data_2024"
    data_folder_2 = "data_2023" 
    
    csv_files_2024, csv_WP_files_2024 = get_files(data_folder_1, 2024)
    csv_files_2023, csv_WP_files_2023 = get_files(data_folder_2, 2023)
    
    graphs_2024, fits_2024 = process_files(csv_files_2024)    
    graphs_2023, fits_2023 = process_files(csv_files_2023, file_offset=len(csv_files_2024))
    
    Emax_2024, _, _, _, _ = extract_fit_parameters(fits_2024)
    Emax_2023, _, _, _, _ = extract_fit_parameters(fits_2023)
    bkg_2024, bkg_2023 = extract_bkg(csv_WP_files_2024), extract_bkg(csv_WP_files_2023)
    gr_2024, gr_2023 = create_bkg_graph(Emax_2024, Emax_2023, bkg_2024, bkg_2023)
    plot_results(gr_2024, gr_2023)

    print("Processamento concluído.")

if __name__ == "__main__":
    main()