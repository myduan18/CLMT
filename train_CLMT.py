import argparse
from prepare_data.DRP_loader import *
from train_model.train_test import *
from prepare_data.create_cell_feat import *
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

root = os.getcwd()

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size (default: 256)')
    # parser.add_argument('--embed_dim',type = int, default= 256 )
    parser.add_argument('--dropout_rate',type = float, default= 0.3) # 0.5
    parser.add_argument('--layer_num',type = int, default= 2 )
    parser.add_argument('--hidden_dim',type = int, default= 128)
    parser.add_argument('--readout',type = str, default= 'mean')
    parser.add_argument('--train_type',type = str, default= 'mix')
    # parser.add_argument('--train_type',type = str, default= 'db' )

    parser.add_argument('--device',type = int, default= 0)
    parser.add_argument('--seed',type = int, default=1234)#1234 
    parser.add_argument('--num_workers',type = int, default= 4)
    parser.add_argument('--epochs',type = int, default= 300)
    parser.add_argument('--lrf',type = float, default= 0.1)
    parser.add_argument('--lr',type = float, default= 1e-3) # 1e-4 db 5e-5 mix 5e-4
    parser.add_argument('--use_norm_ic50',type = str, default= 'False')
    parser.add_argument('--use_regulizer',type = str, default= 'True')
    
    parser.add_argument('--use_regulizer_cell',type = str, default= 'True')
    parser.add_argument('--use_regulizer_drug',type = str, default= 'True')
    parser.add_argument('--use_drug_path_way',type = str, default= 'True')

    parser.add_argument('--regular_weight', type = float, default= 0.5)
    parser.add_argument('--cl_weight', type = float, default= 1e-2) # GDSC 5e-3, CCLE 1e-2
    # parser.add_argument('--balance_weight', type = float, default= 0.1)
    
    parser.add_argument('--check_step', type = int, default= 2)
    parser.add_argument('--early_stop_count', type = int, default= 5)
    parser.add_argument('--view_dim', type= int, default=256)
    parser.add_argument('--part', type= int, default=0) ##cross validation part 0,1,2,3,4
    parser.add_argument('--scheduler_type',type = str, default= 'ML', help= 'OP(OnPlateau) or ML(Multistep)')
    parser.add_argument('--drug_ablation', type= str, default= 'False')

    parser.add_argument('--pathway_dim', type=int, default='100', help='each pathway contains gene number')
    parser.add_argument('--gene_embed_dim', type=int, default='64', help='gene embedding')
    # parser.add_argument('--drug_embed_dim', type=int, default='64', help='drug_embedding')

    parser.add_argument('--drug_embed_dim', type=int, default='256', help='drug_embedding')
    
    parser.add_argument('--graph_embedding_dim', type=int, default='256', help='output embedding dim') 

    parser.add_argument('--dataset', type=str, default='GDSC', help='dataset type: "GDSC" or "CCLE"')

    # parser.add_argument('--drug_desc_path', type=str, default=root+ '/GDSC/drug_descriptions.csv', help='drug description path')
    # parser.add_argument('--cell_desc_path', type=str, default=root+ '/GDSC/cell_descriptions.json', help='cell description path')

    parser.add_argument('--drug_desc_path', type=str, default=root+ '/GDSC/drug_descriptions.csv', help='drug description path')
    parser.add_argument('--cell_desc_path', type=str, default='/home/hai/hai_disk/Mia/CLMT/pathway/gdsc_cell_descriptions.json', help='cell description path')

    # parser.add_argument('--drug_desc_path', type=str, default=root+ '/CCLE/drug_descriptions.csv', help='drug description path')
    # parser.add_argument('--cell_desc_path', type=str, default='/home/hai/hai_disk/Mia/CLMT/pathway/ccle_cell_descriptions.json', help='cell description path')

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu") 
    # device = torch.device('cpu')
    drug_response_dict, drug_name, cell_name, drug_smiles_dict, drug_figureprints_dict = read_dr_dict()
    drug_atom_dict, drug_bond_dict = load_drug_feat()
    mut_dict, cnv_dict, gen_dict, cell_type_dict  = get_cell_feat() 

    train_dict, val_dict, test_dict = n_fold_split(drug_name=drug_name, cell_name= cell_name, 
                                                   drug_response_dict= drug_response_dict, 
                                                    type= args.train_type, 
                                                    seed = args.seed, test_portion=0.1)
    i = args.part
    if args.train_type == 'mix' or args.train_type == 'cb':
        train_idx, val_idx, test_idx = train_dict[i], val_dict[i], test_dict

        # import ipdb
        # ipdb.set_trace()
        
    else:
        train_idx, val_idx, test_idx = train_dict[i], val_dict[i], test_dict[i]

    # import ipdb
    # ipdb.set_trace()

    # train_set = multi_DRP_dataset(drp_idx = train_idx,use_norm_ic50= args.use_norm_ic50)
    # val_set = multi_DRP_dataset(drp_idx = val_idx,use_norm_ic50= args.use_norm_ic50)
    # test_set = multi_DRP_dataset(drp_idx = test_idx, use_norm_ic50= args.use_norm_ic50)

    # dataset = 'CCLE'
    # dataset = 'GDSC'
    dataset = args.dataset

    data, cell_name_dict, drug_name_dict = process_data(gen_dict, mut_dict, cnv_dict, drug_figureprints_dict, dataset=dataset, k=20)

    # ipdb.set_trace()

    train_set = graph_drp_dataset(drug_response_dict, data, train_idx, batch_size=args.batch_size)
    val_set   = graph_drp_dataset(drug_response_dict, data, val_idx, batch_size=args.batch_size)
    test_set  = graph_drp_dataset(drug_response_dict, data, test_idx, batch_size=args.batch_size)



    log_file = 'loss_log.txt'
    model_save_path = f'./save_model/save_model.pth'


    train_multi_view_model(args, train_set, val_set, test_set, i, cell_name_dict, drug_name_dict, log_file, model_save_path)   