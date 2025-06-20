import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os 
import sys
sys.path.append("prepare_data")
from create_drp_dict import *
if not os.path.exists('./Figures/'):
    os.makedirs('./Figures/')
    
drug_response_dict, drug_name, cell_name, drug_smiles_dict, drug_figureprints_dict = read_dr_dict()

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]
###upsampling the data
## For each cell line, fit a norm distribution with N(ic_50,0.2)
def drug_ic50_upsampling():
    drug_ic50_dict = {drug:[] for drug in drug_name}
    def flatten_list(nested_list):
        return [item for sublist in nested_list for item in sublist]
    for idx,(cell,drug,ic50,norm_ic50) in enumerate(drug_response_dict):
    ###upsampling the data
    ## For each cell line, fit a norm distribution with N(ic_50,0.2)
        normal_dist_sampling = np.random.normal(ic50,0.2,100)
        drug_ic50_dict[drug].append(normal_dist_sampling)
    drug_ic50_dict = {drug:flatten_list(drug_ic50_dict[drug]) for drug in drug_name}
    return drug_ic50_dict
def plot_drug_ic50(save = True):
    drug_ic50_dict = drug_ic50_upsampling()
    drug_id_name = pd.read_csv('./Data/GDSC_data/drugid_name.txt',sep='\t',header=None)
    drug_id_name_dict = {item[1]:item[0] for item in drug_id_name.values}
    drug_name_id_dict = {item[0]:item[1] for item in drug_id_name.values}
    for drug_id in drug_name:
        drug_name = drug_name_id_dict[drug_id]
        min_ic50 = np.min(drug_ic50_dict[drug_id])
        max_ic50 = np.max(drug_ic50_dict[drug_id])
        tmp = np.array(drug_ic50_dict[drug_id])
        min_int = int(min_ic50)
        max_int = int(max_ic50)
        sns.histplot(tmp,bins=50)
        # plt.hist(df, bins=interval)
        # plt.xticks(rotation=45)
        plt.xlabel('IC50')
        plt.ylabel('Frequency')
        title_font = fm.FontProperties(family='DejaVu Sans', style='normal', size=14, weight='bold', stretch='normal')
        plt.title(f'Distribution of IC50 for {drug_name}', fontproperties=title_font)
        plt.style.use('ggplot')
        # plt.figure(dpi=100, figsize=(200, 1))
        plt.autoscale(enable=True, axis='x', tight=True)
        # plt.show()
        if save :
            plt.savefig(f"./Figures/{drug_name}.svg", bbox_inches='tight')
        plt.close()


def binarization_drug_resposne():        
    kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
    t = 0.05
    norm_t = norm.ppf(t)        
    def Get_threshold_drug(drug_id):
        tmp = np.array(drug_ic50_dict[drug_id])
        kde.fit(tmp[:, None])
        min_ic50 = np.min(tmp)
        max_ic50 = np.max(tmp)
        x_d = np.linspace(min_ic50, max_ic50, 1000)
        logprob = kde.score_samples(x_d[:, None])
        score = np.exp(logprob)
        max_pos = np.argmax(score.ravel())
        mu = x_d.ravel()[max_pos]
        ### Check the first derivative
        diff = np.gradient(score)
        sdiff = np.sign(diff)
        zc = np.where(sdiff[:-1] != sdiff[1:])
        if zc is not None:
            theta_pos = zc[0][0]
            theta = x_d[theta_pos]    
            cond_1 = theta < mu
            cond_2 = score[:theta_pos].sum()/score.sum() > 0.05
            cond = cond_1 & cond_2
            if cond:
                print('Theta generate by 1st derivative')
            else :
                print('Theta generate failed by 1st derivative, used f_min instead')
                theta = min_ic50
        else: 
            print('Theta generate failed by 1st derivative, used f_min instead')
            theta = min_ic50
        sigma = np.absolute(np.mean([theta,mu]))
        b = norm_t * sigma + mu ##https://www.nature.com/articles/srep36812  ic50<b means sensitive, others resistant
        return b
    drug_id_name = pd.read_csv('./Data/GDSC_data/drugid_name.txt',sep='\t',header=None)
    drug_id_name_dict = {item[1]:item[0] for item in drug_id_name.values}
    drug_name_id_dict = {item[0]:item[1] for item in drug_id_name.values}
    drug_ic50_dict = drug_ic50_upsampling()
    drug_threshold = {drug_id : Get_threshold_drug(drug_id) for drug_id in drug_name_id_dict.keys()}
    drug_response_binary = []
    num = 0
    for line in drug_response_dict:
        cell_id, drug_id, ic50, norm_ic50 = line
        threshold = drug_threshold[drug_id]
        if ic50 < threshold:
            label = 1
            num += 1
        else: label = 0
        drug_response_binary.append([drug_id , cell_id, label])
    def save_data():
        np.save(os.path.join('./Data/DRP_dataset/', 'drug_threshold.npy'), drug_threshold)
        np.save(os.path.join('./Data/DRP_dataset/', 'drug_response_binary.npy'), drug_response_binary)
    save_data()
    return drug_threshold, drug_threshold
def load_binary_ic50():
    drug_response_binary = np.load('./Data/DRP_dataset/drug_response_binary.npy',allow_pickle=True)
    drug_threshold = np.load('./Data/DRP_dataset/drug_threshold.npy',allow_pickle=True).item()
    return drug_response_binary, drug_threshold

if __name__ == '__main__':
    # drug_threshold, drug_threshold = binarization_drug_resposne()
    drug_response_binary, drug_threshold = load_binary_ic50()
    