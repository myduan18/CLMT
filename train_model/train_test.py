import sys
sys.path.append('.')
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, DataLoader
from prepare_data.create_cell_feat import *
from prepare_data.create_drug_feat import *
from prepare_data.create_drp_dict import *
from prepare_data.DRP_loader import *
# from base_line.TGSA_model.tgsa_model import *
from torch_geometric.data import Batch
import torch.optim as opt
from torch_geometric.nn import  graclus
from tqdm import tqdm
from itertools import product
from train_model.utils import get_polynomial_decay_schedule_with_warmup
import argparse

from model.drp_model.model import CLMT
import torch.nn as nn

drug_atom_dict,drug_bond_dict = load_drug_feat()
drug_response_dict, drug_name, cell_name, drug_smiles_dict, drug_figureprints_dict = read_dr_dict()
mut_dict, cnv_dict, gen_dict, cell_type_dict  = get_cell_feat() 
       

def cross_entropy_one_hot(input, target):
    batch_size = input.shape[0]
    num_target = input.shape[1]
    target = target.reshape(batch_size,num_target)
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)


def inter_contrastive(cell_feature1, drug_feature1, cell_feature2, drug_feature2):

    import torch.nn.functional as F

    inter_ssl_temperature = 0.1  

    drug_feature1 = F.normalize(drug_feature1, p=2, dim=1)
    cell_feature1 = F.normalize(cell_feature1, p=2, dim=1)
    drug_feature2 = F.normalize(drug_feature2, p=2, dim=1)
    cell_feature2 = F.normalize(cell_feature2, p=2, dim=1)

    drug_pos_score = torch.multiply(drug_feature1, drug_feature2).sum(dim=1)
    cell_pos_score = torch.multiply(cell_feature1, cell_feature2).sum(dim=1)

    batch_size = drug_feature1.shape[0]
    mask = (torch.ones(batch_size, batch_size) - torch.eye(batch_size)).cuda()

    drug_sim_matrix = torch.matmul(drug_feature1, drug_feature2.t())
    cell_sim_matrix = torch.matmul(cell_feature1, cell_feature2.t())

    drug_neg_matrix = drug_sim_matrix.clone()
    drug_neg_matrix = drug_neg_matrix * mask - 1e9 * (1 - mask) 
    cell_neg_matrix = cell_sim_matrix.clone()
    cell_neg_matrix = cell_neg_matrix * mask - 1e9 * (1 - mask)

    drug_log_prob = drug_pos_score / inter_ssl_temperature - torch.logsumexp(drug_neg_matrix / inter_ssl_temperature, dim=1)
    cell_log_prob = cell_pos_score / inter_ssl_temperature - torch.logsumexp(cell_neg_matrix / inter_ssl_temperature, dim=1)

    drug_ssl_loss = -drug_log_prob.mean()
    cell_ssl_loss = -cell_log_prob.mean()

    # import ipdb
    # ipdb.set_trace()

    return drug_ssl_loss + cell_ssl_loss


def total_loss(out_dict, ic50,  
               cell_type, drug_type, drug_pathway, 
               epoch, args):


    out = out_dict['pred']
    
    cell_dual, drug_dual, cell_graph, drug_graph, cell_text, drug_text = out_dict['features']

    pred_loss = nn.MSELoss()(out, ic50)


    cell_type_pred = out_dict['cell_regulizer']
    drug_type_pred = out_dict['drug_regulizer']
    drug_pathway_pred = out_dict['drug_pathway']

    inter_contrastive_loss_1 = inter_contrastive(cell_dual, drug_dual, cell_graph, drug_graph)
    inter_contrastive_loss_2 = inter_contrastive(cell_dual, drug_dual, cell_text, drug_text)


    alpha = args.cl_weight
    beta = args.regular_weight



    class_l = 0.0
    drug_l = 0.0
    drug_pathway_l = 0.0

    if args.use_regulizer_drug == 'True':
        drug_l = nn.MSELoss()(drug_type_pred, drug_type.view(-1,1))
    if args.use_regulizer_cell == 'True':

        class_l = cross_entropy_one_hot(cell_type_pred, cell_type)

    if args.use_drug_path_way == 'True':
        drug_pathway_l = cross_entropy_one_hot(drug_pathway_pred, drug_pathway)
   
    

    t_loss = pred_loss + alpha*(inter_contrastive_loss_1 + inter_contrastive_loss_2) + beta*(class_l + drug_l + drug_pathway_l)

    return t_loss, pred_loss, class_l, drug_l, drug_pathway_l, inter_contrastive_loss_1, inter_contrastive_loss_2

def graph_batch_from_smiles(smiles_list):
    
    from ogb.utils import smiles2graph
    edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
    graphs = [smiles2graph(x) for x in smiles_list]
    for idx, graph in enumerate(graphs):
        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])
        lstnode += graph['num_nodes']
        batch.append(np.ones(graph['num_nodes'], dtype=np.int64) * idx)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0)
    }
    result = {k: torch.from_numpy(v) for k, v in result.items()}
    result['num_nodes'] = lstnode
    return Data(**result)

from rdkit.Chem import BRICS

def get_frac(SMILES):
    
    tempF = set()
    try:
        m = BRICS.BRICSDecompose(Chem.MolFromSmiles(SMILES))
        for frac in m:
            tempF.add(frac) 
    except:
        pass


    return tempF

def get_batch_fracs(batch_smiles_list):


    fracs_set_unique = set()
    fracs_num = list()
    fracs_list = []

    for SMILES in batch_smiles_list:

        fracs = get_frac(SMILES)
        fracs_set_unique = fracs_set_unique | fracs
        fracs_list.append(list(fracs))
        fracs_num.append(len(fracs))

    fracs_list_unique = list(fracs_set_unique)

    indices = []
    count = 0
    for drug_fracs in fracs_list: 
        e_d_indices = []
        for f in drug_fracs:
            e_d_indices.append(fracs_list_unique.index(f))
        count+=len(e_d_indices)
        indices.append(e_d_indices)

    assert count == sum(fracs_num)

    return fracs_list_unique, indices


def train_step(model, train_loader, optimizer, epoch, device, args, cell_name_dict, drug_name_dict):
    # enable automatic mixed precision
    from torch.cuda.amp import GradScaler

    model.train()
    scaler = GradScaler()

    y_true, preds = [], []
    

    # for batch in train_loader:
    for batch in tqdm(train_loader, desc="Training", unit="batch"):


        src_n_id = batch['cell_line'].n_id
        dst_n_id = batch['drug'].n_id
        src, dst =  batch['cell_line', 'response', 'drug'].edge_label_index


        batch_cell_name = [cell_name_dict[i] for i in src_n_id[src]]
        batch_drug_name = [drug_name_dict[i] for i in dst_n_id[dst]]

        ic50 = batch['cell_line', 'response', 'drug'].edge_label.to(device)


        drug_atom_list = [drug_atom_dict[i] for i in batch_drug_name]
        drug_bond_list = [drug_bond_dict[i] for i in batch_drug_name]
        
        drug_atom = Batch.from_data_list(drug_atom_list)
        drug_bond = Batch.from_data_list(drug_bond_list)

        cell_gen = torch.tensor(np.array([gen_dict[i] for i in batch_cell_name])).float()
        cell_type = torch.tensor(np.array([cell_type_dict[i] for i in batch_cell_name]))

        cell_mut = torch.tensor([mut_dict[i] for i in batch_cell_name]).int()
        cell_cnv = torch.tensor([cnv_dict[i] for i in batch_cell_name]).int()

        drug_atom = drug_atom.to(device)  
        drug_bond = drug_bond.to(device)

        cell_gen = cell_gen.to(device)
        cell_mut = cell_mut.to(device)
        cell_cnv = cell_cnv.to(device)
        cell_type = cell_type.to(device)

        ic50 = ic50.to(device)  


        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            out_dict = model(batch, 
                            drug_atom, drug_bond, 
                            cell_gen, cell_mut, cell_cnv,
                            batch_cell_name, batch_drug_name,
                            device)
            
            out = out_dict['pred'].to(device)

            loss, pred_loss, class_l, drug_l, drug_pathway_l, inter_contrastive_loss_1, inter_contrastive_loss_2 =  total_loss(out_dict, ic50.view(-1, 1).float(), 
                            cell_type,
                            drug_atom.threshold,
                            drug_atom.path_way,
                            epoch,
                            args)
        
        

        y_true.append(ic50.view(-1, 1).float())
        preds.append(out.float().cpu())
        # perform backward pass and optimizer step using the scaler

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
  
        # scheduler.step()
    y_true = torch.cat(y_true, dim=0).cpu().detach().numpy()
    y_pred = torch.cat(preds, dim=0).cpu().detach().numpy()
    rmse = mean_squared_error(y_true,y_pred, squared=False)
    pcc = pearsonr(y_true.flatten(), y_pred.flatten())[0]
    r_2 = r2_score(y_true.flatten(), y_pred.flatten())
    MAE = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    # writer.add_scalar("Loss", rmse, epoch)
    # writer.add_scalar("Accuracy/train/rmse", rmse, epoch)
    # writer.add_scalar("Accuracy/train/mae", MAE, epoch)
    # writer.add_scalar("Accuracy/train/pcc", pcc, epoch)
    # writer.add_scalar("Accuracy/train/r_2", r_2, epoch)
    print(optimizer.param_groups[0]['lr'])
    # rmse=0.0
    # pcc=0.0
    return rmse, pcc, loss

@torch.no_grad()
def test_step(args, model, loader, device, cell_name_dict, drug_name_dict):
    model.eval()
    y_true, preds = [], []


    for batch in tqdm(loader, desc="Testing", unit="batch"):

    # for batch in loader:

        src_n_id = batch['cell_line'].n_id
        dst_n_id = batch['drug'].n_id
        src, dst =  batch['cell_line', 'response', 'drug'].edge_label_index


        batch_cell_name = [cell_name_dict[i] for i in src_n_id[src]]
        batch_drug_name = [drug_name_dict[i] for i in dst_n_id[dst]]

        # print('test! ', count, len(batch_cell_name))

        ic50 = batch['cell_line', 'response', 'drug'].edge_label.to(device)

        drug_atom_list = [drug_atom_dict[i] for i in batch_drug_name]
        drug_bond_list = [drug_bond_dict[i] for i in batch_drug_name]
        
        drug_atom = Batch.from_data_list(drug_atom_list)
        drug_bond = Batch.from_data_list(drug_bond_list)


        cell_gen = torch.tensor(np.array([gen_dict[i] for i in batch_cell_name])).float()
        cell_type = torch.tensor(np.array([cell_type_dict[i] for i in batch_cell_name]))

        cell_mut = torch.tensor([mut_dict[i] for i in batch_cell_name]).int()
        cell_cnv = torch.tensor([cnv_dict[i] for i in batch_cell_name]).int()
 
    # for data in tqdm(loader):

    #     x_dict, edge_index_dict, batch_edge_index = None, None, None

    #     drug_atom, drug_bond, smiles, cell_gen, cell_mut, cell_cnv, cell_type, ic50 = data

        drug_atom = drug_atom.to(device)  
        drug_bond = drug_bond.to(device)

        cell_gen = cell_gen.to(device)
        cell_mut = cell_mut.to(device)
        cell_cnv = cell_cnv.to(device)

        cell_type = cell_type.to(device)
        ic50 = ic50.to(device)  

        out_dict = model(batch, 
                        drug_atom, drug_bond, 
                        cell_gen, cell_mut, cell_cnv,
                        batch_cell_name, batch_drug_name,
                        device)
            
        y_true.append(ic50.view(-1, 1).float())
        out = out_dict['pred'].to(device)
        preds.append(out.float().cpu())
        
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(preds, dim=0).cpu().numpy()
    rmse = mean_squared_error(y_true,y_pred, squared=False)
    pcc = pearsonr(y_true.flatten(), y_pred.flatten())[0]
    r_2 = r2_score(y_true.flatten(), y_pred.flatten())
    MAE = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    return rmse,pcc,r_2,MAE



def train_multi_view_model(args, train_set, val_set, test_set, i, cell_name_dict, drug_name_dict, log_file, path):
    
    device = args.device
    lr = args.lr
    batch_size = args.batch_size
 
    n_epochs = args.epochs
    print(path)


    model_config = {
                'graph_embedding_dim': args.graph_embedding_dim, 
                'dropout_rate': args.dropout_rate,
                'pathway_dim' : args.pathway_dim,
                'drug_embed_dim': args.drug_embed_dim, 

                'use_regulizer_cell' : args.use_regulizer_cell, 
                'use_drug_path_way' : args.use_drug_path_way,
                'use_regulizer_drug' : args.use_regulizer_drug,
                'drug_desc_path': args.drug_desc_path,
                'cell_desc_path': args.cell_desc_path

               }

    # global_para = {
    #         'embed_dim': 256, 'dropout_rate': 0.4,'hidden_dim' : 128,
    #         'layer_num': 2, 'readout': 'mean', 'JK' : 'True'
    #                 }

    # cell_protein_embedding = torch.tensor(np.array([i for i in protein_embeddings_dict.values()]), dtype=torch.float32)
    
    # import ipdb
    # ipdb.set_trace()

    root = os.getcwd()
    bert_model_name = root + '/NeuML/pubmedbert-base-embeddings'

    global_para = {
        'embed_dim': 256, 'dropout_rate': 0.4,'hidden_dim' : 128,
        'layer_num': 2, 'readout': 'mean', 'JK' : 'True'
                }
    
    gen_dim = len(next(iter(gen_dict.values())))
    mut_dim = len(next(iter(mut_dict.values())))
    cnv_dim = len(next(iter(cnv_dict.values())))
    cell_dim = len(next(iter(cell_type_dict.values())))
    drug_pathway_dim = len(next(iter(drug_atom_dict.values())).path_way)

    # import ipdb
    # ipdb.set_trace()
    
    omic_para = {'gen_dim': gen_dim, 'mut_dim': mut_dim, 'cnv_dim':cnv_dim, 'cell_dim': cell_dim, 'drug_pathway_dim': drug_pathway_dim}


    model = CLMT(
        global_para,
        omic_para,
        bert_model_name,
        device,
        model_config
    ).to(device)
    # elif args.use_raw_gene == 'False': 
    #     if args.drug_ablation == 'False':
    #         model = DRP_multi_view_ablation(mut_cluster, cnv_cluster, ge_cluster, model_config).to(device)
    #     elif  args.drug_ablation == 'True':
    #         model = DRP_multi_view_ablation_drug(mut_cluster, cnv_cluster, ge_cluster, model_config).to(device)
    # # model = torch.compile(model)
    optimizer = opt.AdamW(model.parameters(), lr=lr, weight_decay= 0.005) # 0.001
        # scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,num_warmup_steps=50, num_training_steps=n_epochs, lr_end = 1e-4, power=1)
    # elif optimizer_name == 'SGD': 
        # optimizer = opt.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2)
    # cos_lr = lambda x : ((1+math.cos(math.pi* x /100) )/2)*(1-args.lrf) + args.lrf
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cos_lr)

    print('Begin Training')
    

    train_loader = graph_drp_loader(train_set,batch_size= batch_size,shuffle = True)
    val_loader = graph_drp_loader(val_set,batch_size= batch_size,shuffle = True)
    test_loader = graph_drp_loader(test_set,batch_size= batch_size,shuffle = True)

    # import dill
    # with open('test_loader.pkl','wb') as f:
    #     dill.dump(test_loader, f)

    epoch_len = len(str(n_epochs))
    best_val_pcc = -1

    early_stop_count = 0 
    best_epoch = 0 
    best_mae = 100
    best_val_pcc = -1
    best_val_rmse = 100

    scheduler_type = args.scheduler_type

    if args.scheduler_type == 'OP':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5 , verbose=True, min_lr= 0.05 * args.lr, factor= 0.1)
    elif args.scheduler_type == 'ML':
        scheduler = opt.lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.1)


    with open(log_file, "a") as f:
    
        for epoch in range(n_epochs):

            if early_stop_count < args.early_stop_count :

                train_rmse, train_pcc, loss = train_step(model, train_loader, optimizer,  epoch, device, args, cell_name_dict, drug_name_dict)
                

                f.write(f"Epoch {epoch+1},"
                            f"train_rmse: {train_rmse:.4f},"
                            f"train_pcc: {train_pcc:.5f},"
                            f"Total Loss: {loss.item():.4f}, \n"           
                        )

                if args.scheduler_type == 'ML':
                    scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] '  + 
                            f'train_rmse: {train_rmse:.5f} ' +
                            f'train_pcc: {train_pcc:.5f} ' +  f'lr : {current_lr}')         
                
                print(print_msg)

                if epoch % args.check_step == 0:
                    val_rmse, val_pcc, val_r_2, val_mae = test_step(args, model, val_loader, device, cell_name_dict, drug_name_dict)

                    if args.scheduler_type == 'OP':
                        scheduler.step(val_rmse)

                    print(f'Test for Epoch: {epoch:03d},  val_rmse:{val_rmse:.4f}, val_PCC: {val_pcc:.4f}')

                    if val_rmse < best_val_rmse or val_pcc > best_val_pcc:
                        early_stop_count = 0
                        best_val_pcc = val_pcc
                        best_val_rmse = val_rmse
                        # best_r_2 = val_r_2
                        # best_mae = val_mae
                        best_epoch = epoch
                        
                        test_rmse, test_pcc, test_r_2, test_mae = test_step(args, model,test_loader, device, cell_name_dict, drug_name_dict)

                        print(f'Test PCC: {test_pcc:.4f}')
                        print(f'Test RMSE: {test_rmse:.4f}, Test R_2: {test_r_2:.4f}, Test MAE: {test_mae:.4f}')
                        
                        #   Save the last batch loss
                        f.write(f"Epoch {epoch+1},"
                                    f"test_rmse: {test_rmse:.4f},"
                                    f"test_pcc: {test_pcc:.5f},"
                                    f"test_r_2: {test_r_2:.5f},"
                                    f"test_mae: {test_mae:.5f}\n"
                                )
                        f.flush()  
                         
                        torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict' : optimizer.state_dict(),
                                }, path)
                    else: 
                        early_stop_count += 1 
                        print(f'Early stopping encounter : {early_stop_count}  times')
                    if early_stop_count >= args.early_stop_count:
                        print('Early stopping!')
                        break
        print("__________________________________________________________")
        

        print(f'Best epoch: {best_epoch:03d}, Best PCC: {test_pcc:.4f}')
        print(f'Best RMSE: {test_rmse:.4f}, Best R_2: {test_r_2:.4f}, Best MAE: {test_mae:.4f}')

        f.write(f"Epoch {epoch+1},"
            f"test_rmse: {test_rmse:.4f},"
            f"test_pcc: {test_pcc:.5f},"
            f"test_r_2: {test_r_2:.5f},"
            f"test_mae: {test_mae:.5f}\n"
        )


        # load the model and record the response results

