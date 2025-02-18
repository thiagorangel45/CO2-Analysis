import ROOT
import pandas as pd
import os
import math
from ROOT import TLine, TLegend

data_folder = "data_2024" # altere aqui o nome da pasta onde estão os arquivos .CSV

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

    if 'HV_top' not in df.columns or 'efficiency' not in df.columns or 'eff_error' not in df.columns:
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

Emax, Emax_err, Lambda, Lambda_err, HV50, HV50_err, HV95, HV95_err, WP = [], [], [], [], [], [], [], [], []

for sigmoid in fits:
    Emax_ = sigmoid.GetParameter(0)
    Emax.append(Emax_)
    Emax_err.append(sigmoid.GetParError(0))     
    Lambda.append(sigmoid.GetParameter(1))  
    Lambda_err.append(sigmoid.GetParError(1))
    HV50.append(sigmoid.GetParameter(2))  
    HV50_err.append(sigmoid.GetParError(2)) 

    # Calcular HV95
    HV95_val = sigmoid.GetX(HV_ref)
    HV95.append(HV95_val)
    HV95_err.append((math.log(19) / sigmoid.GetParameter(1)**2) * sigmoid.GetParError(1) + sigmoid.GetParError(2))
    WP.append(HV50[-1] - math.log(1 / 0.95 - 1) / Lambda[-1] + 150.)

legend = TLegend(0.12, 0.68, 0.3, 0.89)
legend.SetTextFont(42)
legend.SetBorderSize(0) 
legend.SetFillStyle(4000)
legend.SetFillColor(0)  
legend.SetTextSize(0.02376)  

for i, file in enumerate(csv_WP_files):
    if not os.path.isfile(file):
        print(f"Erro: Arquivo WP '{file}' não encontrado. Pulando...")
        continue
    
    df = pd.read_csv(file)
    if 'noiseGammaRate' not in df.columns or 'gamma_CS' not in df.columns:
        print(f"Erro: Colunas esperadas não encontradas em '{file}'. Pulando...")
        continue
    
    if (df['noiseGammaRate'][0] / (df['gamma_CS'][0] * 1000) == 0):
        eff_i_STDMX = f'plateau = {Emax[i]:.0%}, WP = {(WP[i]/1000):.2f} kV, no background gamma rate, Eff(WP) = {fits[i].Eval(WP[i]):.0%}'
    else:
        txt = df['noiseGammaRate'][0] / (df['gamma_CS'][0] * 1000)
        eff_i_STDMX = f'plateau = {Emax[i]:.0%}, WP = {(WP[i]/1000):.2f} kV, bkg gamma rate = {txt:.1f} kHz/cm^{{2}}, Eff(WP) = {fits[i].Eval(WP[i]):.0%}'
    
    legend.AddEntry(graphs[i], eff_i_STDMX, "p")


mg.GetYaxis().SetRangeUser(0, 1.4)
mg.GetXaxis().SetRangeUser(6300, 7600)
mg.Draw("AP")

line = TLine(6300, 1., 7600, 1.)
line.SetLineColor(1)
line.SetLineStyle(9)
line.SetLineWidth(2)
line.Draw()
legend.Draw()  

c1.Draw()
c1.SaveAs("eff_vs_HV_with_fits.pdf")