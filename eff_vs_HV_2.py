import ROOT
import pandas as pd
import os
import math
from ROOT import TLine, TLegend

data_folder = "data_2024"  # Nome da pasta onde estão os arquivos .CSV

num_files = int(input("Quantos scans deseja analisar? "))

csv_files = []
csv_WP_files = []
HV_ref = 95  

for i in range(num_files):
    file_name = input(f"Digite o nome do arquivo {i+1} (ex: STDMX_1.csv): ")
    full_path = os.path.join(data_folder, file_name)
    if not os.path.isfile(full_path):
        print(f"Erro: Arquivo '{file_name}' não encontrado na pasta {data_folder}. Pulando...")
        continue
    
    csv_files.append(full_path)
    base_name, ext = os.path.splitext(file_name)
    csv_WP_files.append(os.path.join(data_folder, f"{base_name}_WP{ext}"))

print("Arquivos WP correspondentes:", csv_WP_files)

c1 = ROOT.TCanvas("c1", "Efficiency vs HV_top with Fits", 630, 600)
mg = ROOT.TMultiGraph()
colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen+2, ROOT.kMagenta, ROOT.kOrange, ROOT.kCyan, ROOT.kBlack]
markers = [20, 21, 22, 23, 24, 25, 26]

graphs = []
fits = []

for i, file in enumerate(csv_files):
    df = pd.read_csv(file)

    if not {'HV_top', 'efficiency', 'eff_error'}.issubset(df.columns):
        print(f"Erro: Colunas esperadas não encontradas em '{file}'. Pulando...")
        continue
    
    n = len(df)
    gr = ROOT.TGraphErrors(n, 
                           df['HV_top'].values.astype(float), 
                           df['efficiency'].values.astype(float),
                           ROOT.nullptr,  
                           df['eff_error'].values.astype(float))
    
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    
    gr.SetMarkerStyle(marker)
    gr.SetMarkerColor(color)
    gr.SetLineColor(color)
    gr.SetTitle(file.replace(".csv", ""))
    
    mg.Add(gr)
    graphs.append(gr)

    sigmoid = ROOT.TF1(f"sigmoid_{i}", "[0]/(1+ TMath::Exp(-[1]*(x-[2])))", df['HV_top'].min(), df['HV_top'].max())
    sigmoid.SetParNames("Emax", "Lambda", "HV50")
    sigmoid.SetParameters(0.9, 0.01, 7000)
    sigmoid.SetLineColor(color)
    
    gr.Fit(sigmoid, "R")
    fits.append(sigmoid)

Emax, Lambda, HV50, HV95, WP = [], [], [], [], []

for sigmoid in fits:
    Emax.append(sigmoid.GetParameter(0))
    Lambda.append(sigmoid.GetParameter(1))
    HV50.append(sigmoid.GetParameter(2))
    
    HV95.append(sigmoid.GetX(HV_ref))
    WP.append(HV50[-1] - math.log(1 / 0.95 - 1) / Lambda[-1] + 150.)

legend = TLegend(0.12, 0.68, 0.3, 0.89)
legend.SetTextFont(42)
legend.SetBorderSize(0) 
legend.SetFillStyle(4000)
legend.SetFillColor(0)  
legend.SetTextSize(0.02376)  

gas_mixtures = [
    "Standard gas mixture",
    "30% CO_{2} + 1.0% SF_{6}",
    "30% CO_{2} + 0.5% SF_{6}",
    "40% CO_{2} + 1.0% SF_{6}"
]

gas_index = 0
for i, file in enumerate(csv_WP_files):
    if not os.path.isfile(file):
        print(f"Erro: Arquivo WP '{file}' não encontrado. Pulando...")
        continue
    
    df = pd.read_csv(file)
    if not {'noiseGammaRate', 'gamma_CS'}.issubset(df.columns):
        print(f"Erro: Colunas esperadas não encontradas em '{file}'. Pulando...")
        continue
    
    txt = df['noiseGammaRate'][0] / (df['gamma_CS'][0] * 1000)
    eff_text = f"plateau = {Emax[i]:.0%}, WP = {(WP[i]/1000):.2f} kV, {gas_mixtures[gas_index]}, Eff(WP) = {fits[i].Eval(WP[i]):.0%}"
    gas_index = (gas_index + 1) % len(gas_mixtures)
    
    legend.AddEntry(graphs[i], eff_text, "p")

mg.GetYaxis().SetRangeUser(0, 1.4)
mg.GetXaxis().SetRangeUser(6000, 7400)
mg.Draw("AP")

line = TLine(6000, 1., 7600, 1.)
line.SetLineColor(1)
line.SetLineStyle(9)
line.SetLineWidth(2)
line.Draw()
legend.Draw()  

c1.Draw()
c1.SaveAs("eff_vs_HV_with_fits.pdf")