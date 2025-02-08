import ROOT
import pandas as pd
import os
import math
from ROOT import TLine, TLegend

num_files = int(input("Quantos scans deseja analisar? "))

csv_files = []
csv_WP_files = []
HV_ref = 95

for i in range(num_files):
    file_name = input(f"Digite o nome do arquivo {i+1} (ex: STDMX_1.csv): ")
    csv_files.append(file_name)

    base_name, ext = os.path.splitext(file_name)
    csv_WP_files.append(f"{base_name}_WP{ext}")

print("Arquivos WP correspondentes:", csv_WP_files)

# Canvas e gráficos
c1 = ROOT.TCanvas("c1", "Efficiency vs HV_top with Fits", 630, 600)
mg = ROOT.TMultiGraph()

colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen+2, ROOT.kMagenta, ROOT.kOrange, ROOT.kCyan, ROOT.kBlack]
markers = [20, 21, 22, 23, 24, 25, 26]

graphs = []
fits = []

for i, file in enumerate(csv_files):
    df = pd.read_csv(file)
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

    # Ajuste sigmoid
    sigmoid = ROOT.TF1(f"sigmoid_{i}", "[0]/(1+ TMath::Exp(-[1]*(x-[2])))", df['HV_top'].min(), df['HV_top'].max())
    sigmoid.SetParNames("Emax", "Lambda", "HV50")
    sigmoid.SetParameters(0.9, 0.01, 7000)
    sigmoid.SetLineColor(color)
    gr.Fit(sigmoid, "R")
    fits.append(sigmoid)

Emax = []
Emax_err = []
Lambda = []
Lambda_err = []
HV50 = []
HV50_err = []
HV95 = []
HV95_err = []
WP = []

for sigmoid in fits:
    Emax_ = 1 - math.sqrt((1 - sigmoid.GetParameter(0))**2)
    Emax.append(Emax_)
    
    Emax_err.append(sigmoid.GetParError(0))     
    Lambda.append(sigmoid.GetParameter(1))  
    Lambda_err.append(sigmoid.GetParError(1))
    HV50.append(sigmoid.GetParameter(2))  
    HV50_err.append(sigmoid.GetParError(2)) 
    HV95_val = sigmoid.GetX(HV_ref)
    HV95.append(HV95_val)
    HV95_err.append((math.log(19) / sigmoid.GetParameter(1)**2) * sigmoid.GetParError(1) + sigmoid.GetParError(2))  # erro HV95

for Emax_, Lambda_, HV50_ in zip(Emax, Lambda, HV50):
    WP.append(HV50_ - math.log(1 / 0.95 - 1) / Lambda_ + 150.)

legend = TLegend(0.12, 0.68, 0.3, 0.89)
legend.SetTextFont(42)
legend.SetBorderSize(0) 
legend.SetFillStyle(4000)
legend.SetFillColor(0)  
legend.SetTextSize(0.02376)  

for i, file in enumerate(csv_WP_files):
    df = pd.read_csv(file)
    if i == 0:
        eff_i_STDMX = f'plateau = {Emax[i]:.0%}, WP = {(WP[i]/1000):.2f} kV, no hit bkg gamma rate, Eff(WP) = {fits[i].Eval(WP[i]):.0%}'
        legend.AddEntry(graphs[i], eff_i_STDMX, "p")
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