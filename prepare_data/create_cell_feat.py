import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import ipdb
from sklearn.preprocessing import MinMaxScaler
import os

def get_cell_feat():

    ############################       GDSC data       ###
    cell_info_file = './GDSC/GDSC_data/Table_S5_GDSC_Cell_line_annotation.csv'
    gene_path = './GDSC/GDSC_data/Table_S1_GDSC_Gene_expression.csv'
    mut_path = './GDSC/GDSC_data/Table_S3_GDSC_Mutation.csv'
    copy_path = './GDSC/GDSC_data/Table_S4_GDSC_Copy_number_variation.csv'
    cell_info = pd.read_csv(cell_info_file, index_col=0)[['PRIMARY_SITE', 'COSMIC_ID']]
    cell_info_dict = cell_info.set_index('COSMIC_ID')['PRIMARY_SITE'].T.to_dict()
    

    gene_exp = pd.read_csv(gene_path, index_col=0)
    # cell_gene_names = list(gene_exp.columns)/
    mutation = pd.read_csv(mut_path, index_col=0)
    copy_number = pd.read_csv(copy_path, index_col=0)
    cell_info_dict = cell_info.set_index('COSMIC_ID')['PRIMARY_SITE'].T.to_dict()

    #############################       GDSC data       ######################################


    #############################       CCLE data       ######################################

    # def extract_and_convert(input_string):
    #     digits = ''.join(char for char in input_string if char.isdigit())
        
    #     first_non_zero_index = -1
    #     for i, digit in enumerate(digits):
    #         if digit != '0':
    #             first_non_zero_index = i
    #             break
        
    #     if first_non_zero_index != -1:
    #         extracted_digits = digits[first_non_zero_index:]
    #         return int(extracted_digits)
    #     else:
    #         return 0
    
    # root = os.getcwd()
    
    # cell_info_file = root + '/CCLE/CCLE_data/Table_S25_CCLE_Cell_line_annotation-u.csv'
    # path_ccle_id_name = root + '/CCLE/CCLE_data/Cell_lines_annotations_20181226.txt'   
    # gene_path = root + '/CCLE/CCLE_data/Table_S13_CCLE_Gene_expression.csv'
    # mut_path = root + '/CCLE/CCLE_data/Table_S17_CCLE_Mutation.csv'
    # copy_path = root + '/CCLE/CCLE_data/Table_S18_CCLE_Copy_number_variation.csv'
    # path_ccle_id_name = root + '/CCLE/CCLE_data/Cell_lines_annotations_20181226.txt'    
    
    # ccle_id_name = pd.read_csv(path_ccle_id_name,  sep='\t')
    # def re_ccle(x): return extract_and_convert(ccle_id_name[ccle_id_name['CCLE_ID'] == x]['depMapID'].item())

    # cell_info = pd.read_csv(cell_info_file, index_col=0)[['PRIMARY_SITE', 'ID']]
    # cell_info['ID'] = cell_info['ID'].apply(lambda x: extract_and_convert(x))
    # cell_info_dict = cell_info.set_index('ID')['PRIMARY_SITE'].T.to_dict()
    # gene_exp = pd.read_csv(gene_path, index_col=0)
    # gene_exp.rename(index={x: re_ccle(x)
    #                           for x in gene_exp.index}, inplace=True)
    
    # mutation = pd.read_csv(mut_path, index_col=0)
    # copy_number = pd.read_csv(copy_path, index_col=0)

    # mutation.rename(index={x: re_ccle(x)
    #                           for x in mutation.index}, inplace=True)
    # copy_number.rename(index={x: re_ccle(x)
    #                           for x in copy_number.index}, inplace=True)
    # cell_info_dict = cell_info.set_index('ID')['PRIMARY_SITE'].T.to_dict()

    #############################       CCLE data       ######################################

    
    from sklearn.preprocessing import MinMaxScaler
    gene_exp_scaler = MinMaxScaler()
    gene_exp_data = gene_exp_scaler.fit_transform(gene_exp.values)
    gene_exp = pd.DataFrame(gene_exp_data, index=gene_exp.index, columns=gene_exp.columns)

    from sklearn.feature_selection import VarianceThreshold

    if gene_exp.shape[1] > 8000:
        std = VarianceThreshold()
        variance_ = std.fit(gene_exp.values).variances_
        var_index = np.argsort(-variance_)[:8000]

        # genes = np.array(gene_exp.columns)[var_index]
        
        # import dill 
        # with open('gene_exp_8000.pkl', 'wb') as f:
        #     dill.dump(genes, f)

        gene_exp = gene_exp.iloc[:, var_index]

        print("gene expression data shape: ", gene_exp.shape)

    # ipdb.set_trace()

    mut_dict = {key: list(value.values()) for key, value in mutation.to_dict(orient='index').items()}
    cnv_dict = {key: list(value.values()) for key, value in copy_number.to_dict(orient='index').items()}

    classes = sorted(list(set(cell_info_dict.values())))
    # import ipdb
    # ipdb.set_trace()
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    cell_type_dict = {}
    
    for sample_id, class_name in cell_info_dict.items():
        one_hot_vector = np.zeros(len(classes))
        one_hot_vector[class_to_index[class_name]] = 1
        cell_type_dict[sample_id] = one_hot_vector

    gen_dict = {key: list(value.values()) for key, value in gene_exp.to_dict(orient='index').items()}

    print("finish loading cell data!")


    return mut_dict, cnv_dict, gen_dict, cell_type_dict


if __name__ =='__main__':

    # cell_info_file = './Data/GDSC_data/Table_S5_GDSC_Cell_line_annotation.csv'
    # gene_path = './Data/GDSC_data/Table_S1_GDSC_Gene_expression.csv'
    # mut_path = './Data/GDSC_data/Table_S3_GDSC_Mutation.csv'
    # copy_path = './Data/GDSC_data/Table_S4_GDSC_Copy_number_variation.csv'
      

    mut_dict, cnv_dict, gen_dict, cell_type_dict = get_cell_feat()

