import ROOT
import pandas as pd
import os
import math

# Configurações gerais
data_folder = "data_2024" 
HV_ref = 95  
colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen+2, ROOT.kMagenta, ROOT.kOrange, ROOT.kCyan, ROOT.kBlack]
markers = [20, 21, 22, 23, 24, 25, 26]

def get_file_list(num_files):
    csv_files = []
    csv_WP_files = []
    for i in range(num_files):
        file_name = input(f"Digite o nome do arquivo {i+1} (ex: STDMX_1.csv): ")
        full_path = os.path.join(data_folder, file_name)
        if not os.path.isfile(full_path):
            print(f"Erro: Arquivo '{file_name}' não encontrado na pasta {data_folder}. Pulando...")
            continue
        csv_files.append(full_path)
        base_name, ext = os.path.splitext(file_name)
        csv_WP_files.append(os.path.join(data_folder, f"{base_name}_WP{ext}"))
    return csv_files, csv_WP_files

def create_graph(df, index):
    gr = ROOT.TGraphErrors(len(df),
                           df['HV_top'].values.astype(float), 
                           df['efficiency'].values.astype(float),
                           ROOT.nullptr,  
                           df['eff_error'].values.astype(float))
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


def plot_legends(csv_WP_files, graphs, fits, Emax, WP):
    mixture = [item.split("/")[-1].split("_")[0] for item in csv_WP_files]
    print(mixture)

    legend = ROOT.TLegend(0.12, 0.68, 0.3, 0.89)
    legend.SetTextFont(42)
    legend.SetBorderSize(0) 
    legend.SetFillStyle(4000)
    legend.SetFillColor(0)  
    legend.SetTextSize(0.02376) 

    if all(p == mixture[0] for p in mixture):
        for i, file in enumerate(csv_WP_files):
            if not os.path.isfile(file):
                print(f"Erro: Arquivo WP '{file}' não encontrado. Pulando...")
                continue
            
            df = pd.read_csv(file)
            if not {'noiseGammaRate', 'gamma_CS'}.issubset(df.columns):
                print(f"Erro: Colunas esperadas não encontradas em '{file}'. Pulando...")
                continue
            
            txt = df['noiseGammaRate'][0] / (df['gamma_CS'][0] * 1000)
            eff_text = f"plateau = {Emax[i]:.0%}, WP = {(WP[i]/1000):.2f} kV, bkg gamma rate = {txt:.1f} kHz/cm^{{2}}, Eff(WP) = {fits[i].Eval(WP[i]):.0%}"
            
            legend.AddEntry(graphs[i], eff_text, "p")
    
    else:
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

    return legend  


def plot_heads():
    cms_tex = ROOT.TLatex()
    cms_tex.SetNDC()
    cms_tex.SetTextFont(61)
    cms_tex.SetTextSize(0.04)
    cms_tex.DrawLatex(0.10, 0.905, "CMS MUON")

    cms_tex_1 = ROOT.TLatex()
    cms_tex_1.SetNDC()
    cms_tex_1.SetTextFont(61)
    cms_tex_1.SetTextSize(0.04)
    cms_tex_1.DrawLatex(0.80, 0.905, "GIF++")

    cms_tex_2 = ROOT.TLatex()
    cms_tex_2.SetNDC()
    cms_tex_2.SetTextFont(52)
    cms_tex_2.SetTextSize(0.04)
    cms_tex_2.DrawLatex(0.32, 0.905, "Preliminary")


def plot_results(graphs, fits, legend):
    """ Cria e exibe o gráfico final """
    c1 = ROOT.TCanvas("c1", "Efficiency vs HV_top with Fits", 630, 600)
    mg = ROOT.TMultiGraph()
    for gr in graphs:
        mg.Add(gr)

    mg.GetYaxis().SetRangeUser(0, 1.4)
    mg.GetXaxis().SetRangeUser(6000, 7400)
    mg.GetXaxis().SetTitle("HV_{eff} [V]")
    mg.GetYaxis().SetTitle("Efficiency")
    mg.Draw("AP")

    line = ROOT.TLine(6250, 1., 7400, 1.)
    line.SetLineColor(1)
    line.SetLineStyle(9)
    line.SetLineWidth(2)
    line.Draw()

    legend.Draw()
    plot_heads() 
    c1.Draw()
    c1.SaveAs("eff_vs_HV_with_fits.pdf")


def main():
    num_files = int(input("Quantos scans deseja analisar? "))
    csv_files, csv_WP_files = get_file_list(num_files)
    
    graphs, fits = [], []
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file)
        if not {'HV_top', 'efficiency', 'eff_error'}.issubset(df.columns):
            print(f"Erro: Colunas esperadas não encontradas em '{file}'. Pulando...")
            continue
        
        gr = create_graph(df, i)
        graphs.append(gr)
        fits.append(fit_sigmoid(gr, df, i))
    
    Emax, Lambda, HV50, HV95, WP = extract_fit_parameters(fits)
    legend = plot_legends(csv_WP_files, graphs, fits, Emax, WP)   
    plot_heads()
    plot_results(graphs, fits, legend)
    
if __name__ == "__main__":
    main()