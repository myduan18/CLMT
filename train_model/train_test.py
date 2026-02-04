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

# from model.drp_model.model_u import CLMT

import torch.nn as nn

drug_atom_dict,drug_bond_dict = load_drug_feat()
drug_response_dict, drug_name, cell_name, drug_smiles_dict, drug_figureprints_dict = read_dr_dict()
mut_dict, cnv_dict, gen_dict, cell_type_dict  = get_cell_feat() 
       
# grad_log_file = "grad_alignment_log.csv"

def cross_entropy_one_hot(input, target):
    batch_size = input.shape[0]
    num_target = input.shape[1]
    target = target.reshape(batch_size,num_target)
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)


def cross_contrastive(cell_feature1, drug_feature1, cell_feature2, drug_feature2):

    import torch.nn.functional as F

    cross_ssl_temperature = 0.3

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

    drug_log_prob = drug_pos_score / cross_ssl_temperature - torch.logsumexp(drug_neg_matrix / cross_ssl_temperature, dim=1)
    cell_log_prob = cell_pos_score / cross_ssl_temperature - torch.logsumexp(cell_neg_matrix / cross_ssl_temperature, dim=1)

    drug_ssl_loss = -drug_log_prob.mean()
    cell_ssl_loss = -cell_log_prob.mean()

    # import ipdb
    # ipdb.set_trace()

    return drug_ssl_loss + cell_ssl_loss


def total_loss(out_dict, ic50,  
               cell_type, drug_type, drug_pathway, 
               epoch, args):


    out = out_dict['pred']
    pred_loss = nn.MSELoss()(out, ic50)
    

    cell_dual, drug_dual, cell_graph, drug_graph, cell_text, drug_text = out_dict['features']

    cell_type_pred = out_dict['cell_regulizer']
    drug_type_pred = out_dict['drug_regulizer']
    drug_pathway_pred = out_dict['drug_pathway']

    alpha = args.cl_weight
    beta = args.regular_weight

    class_l = 0.0
    drug_l = 0.0
    drug_pathway_l = 0.0

    intra_contrastive_loss, cross_contrastive_loss = 0.0, 0.0

    if args.use_regulizer_drug == 'True':
        drug_l = nn.MSELoss()(drug_type_pred, drug_type.view(-1,1))
    if args.use_regulizer_cell == 'True':

        class_l = cross_entropy_one_hot(cell_type_pred, cell_type)

    if args.use_drug_path_way == 'True':
        drug_pathway_l = cross_entropy_one_hot(drug_pathway_pred, drug_pathway)

    
    intra_contrastive_loss = cross_contrastive(cell_dual, drug_dual, cell_graph, drug_graph)
    cross_contrastive_loss = cross_contrastive(cell_dual, drug_dual, cell_text, drug_text)

    # cross_contrastive_loss_3 = cross_contrastive(cell_dual, drug_dual, cell_graph, drug_graph)
    # ipdb.set_trace()

    contrastive_loss = alpha*(intra_contrastive_loss + cross_contrastive_loss)

    regularize_loss  = beta*(class_l + drug_l + drug_pathway_l)

    t_loss = pred_loss + contrastive_loss + regularize_loss

    return t_loss, pred_loss, intra_contrastive_loss, cross_contrastive_loss, regularize_loss

def train_step(model, train_loader, optimizer, epoch, device, args, cell_name_dict, drug_name_dict):
    # enable automatic mixed precision



    def check_grad_align(pred_loss, cl_loss):
        shared = [p for n, p in model.named_parameters() 
                 if any(k in n for k in [                 
                 'GNN_drug',         
                 'gene_linear',      
                 'heterograph',      
                 'cell_text_linear', 
                 'drug_text_linear', 
                 ])and p.requires_grad]
        if not shared: return None
        
        model.zero_grad()
        pred_loss.backward(retain_graph=True)
        g1 = torch.cat([p.grad.view(-1) for p in shared if p.grad is not None])
        
        model.zero_grad()
        cl_loss.backward(retain_graph=True)
        g2 = torch.cat([p.grad.view(-1) for p in shared if p.grad is not None])
        
        import torch.nn.functional as F

        cos_z = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()


        # g_z_task = shared.grad.detach().clone()
        # model.zero_grad()

        # pred_loss.backward(retain_graph=True)
        # g_z_con = shared.grad.detach().clone()
        # model.zero_grad()

        # cos_z = torch.nn.functional.cosine_similarity(
        #     g_z_task.view(shared.size(0), -1),
        #     g_z_con.view(shared.size(0), -1),
        #     dim=1
        # ).mean()


        return cos_z 

    model.train()

    batch_drugs = []
    batch_cells = []
    ic50s = []

    y_true, preds = [], []
    
    batch_idx = 0
    
    optimizer.zero_grad()

    total_loss_sum = 0.0
    pred_loss_sum = 0.0

    intra_contrastive_loss_sum = 0.0
    cross_contrastive_loss_sum = 0.0

    regularize_loss_sum = 0.0
    num_batches = 0

    # for batch in train_loader:
    for batch in tqdm(train_loader, desc="Training", unit="batch"):
        
        batch_idx += 1
        num_batches += 1 

        src_n_id = batch['cell_line'].n_id
        dst_n_id = batch['drug'].n_id
        src, dst =  batch['cell_line', 'response', 'drug'].edge_label_index


        batch_cell_name = [cell_name_dict[i] for i in src_n_id[src]]
        batch_drug_name = [drug_name_dict[i] for i in dst_n_id[dst]]


        ic50 = batch['cell_line', 'response', 'drug'].edge_label.to(device)
        batch_drugs.extend(batch_drug_name)
        batch_cells.extend(batch_cell_name)
        ic50s.extend(ic50.view(-1, 1).cpu().detach().numpy())

        # print(len(batch_drug_name))


        drug_atom_list = [drug_atom_dict[i] for i in batch_drug_name]
        drug_bond_list = [drug_bond_dict[i] for i in batch_drug_name]

        # ipdb.set_trace()
        
        drug_atom = Batch.from_data_list(drug_atom_list)
        drug_bond = Batch.from_data_list(drug_bond_list)

        cell_gen = torch.tensor(np.array([gen_dict[i] for i in batch_cell_name])).float()
        cell_type = torch.tensor(np.array([cell_type_dict[i] for i in batch_cell_name]))

        cell_mut = torch.tensor([mut_dict[i] for i in batch_cell_name]).float()
        cell_cnv = torch.tensor([cnv_dict[i] for i in batch_cell_name]).float()

        drug_atom = drug_atom.to(device)  
        drug_bond = drug_bond.to(device)

        cell_gen = cell_gen.to(device)
        cell_mut = cell_mut.to(device)
        cell_cnv = cell_cnv.to(device)
        cell_type = cell_type.to(device)

        ic50 = ic50.to(device)  
        # with torch.cuda.amp.autocast():

        out_dict = model(batch, 
                        drug_atom, drug_bond, 
                        cell_gen, cell_mut, cell_cnv,
                        batch_cell_name, batch_drug_name,
                        device)
        
        out = out_dict['pred'].to(device)

        loss, pred_loss, intra_contrastive_loss, cross_contrastive_loss, regularize_loss =  total_loss(out_dict, ic50.view(-1, 1).float(), 
                        cell_type,
                        drug_atom.threshold,
                        drug_atom.path_way,
                        epoch,
                        args)
        
        # if batch_idx % 198 == 0:
        #     cl_loss = args.cl_weight * (intra_contrastive_loss + cross_contrastive_loss)
        #     cos_sim = check_grad_align(pred_loss, cl_loss)
        #     if cos_sim is not None:
        #         print(f"  [Alignment] {cos_sim:.4f} {'✓' if cos_sim>0 else '✗'}")

        #         line = f"{batch_idx},{cos_sim:.6f}\n"
        #         print(f"  [Alignment] {line.strip()}")
        #         with open('grad_alignment_log.csv', "a") as grad_f:
        #             grad_f.write(line)
        #             grad_f.flush()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss_sum += loss.item()
        pred_loss_sum += pred_loss.item()

        intra_contrastive_loss_sum += intra_contrastive_loss.item()
        cross_contrastive_loss_sum += cross_contrastive_loss.item()

        regularize_loss_sum += regularize_loss.item()

        y_true.append(ic50.view(-1, 1).float())
        preds.append(out.float().cpu())
    
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


    avg_total_loss = total_loss_sum / num_batches
    avg_pred_loss = pred_loss_sum / num_batches

    avg_intra_CL = intra_contrastive_loss_sum / num_batches
    avg_cross_CL = cross_contrastive_loss_sum / num_batches

    avg_regularize_loss = regularize_loss_sum / num_batches

    # exit()

    return rmse, pcc, avg_total_loss, avg_pred_loss, avg_intra_CL, avg_cross_CL, avg_regularize_loss


@torch.no_grad()
def test_step(args, model, loader, device, cell_name_dict, drug_name_dict):
    model.eval()
    y_true, preds = [], []
    cell_names, drug_names = [], []


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

        cell_mut = torch.tensor([mut_dict[i] for i in batch_cell_name]).float()
        cell_cnv = torch.tensor([cnv_dict[i] for i in batch_cell_name]).float()

        cell_names.extend(batch_cell_name)
        drug_names.extend(batch_drug_name)
 
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
    return rmse,pcc,r_2,MAE, y_true, y_pred, cell_names, drug_names



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

    # ipdb.set_trace()

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

    loss_log_file = 'loss_components_text.csv'
    
    # 写入CSV表头
    with open(loss_log_file, 'w') as log_f:
        log_f.write('epoch,total_loss,pred_loss,intra_CL_loss,cross_CL_loss,regularize_loss\n')

    if args.scheduler_type == 'OP':
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5 , verbose=True, min_lr= 0.05 * args.lr, factor= 0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5 , verbose=True, min_lr= 0.05 * args.lr, factor= 0.5)

    elif args.scheduler_type == 'ML':
        scheduler = opt.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=0.1)


    with open(log_file, "a") as f:                        
    #     # ========================================================
    
        for epoch in range(n_epochs):

            if early_stop_count < args.early_stop_count :

                train_rmse, train_pcc, loss, pred_loss, avg_intra_CL, avg_cross_CL, regularize_loss = train_step(
                    model, train_loader, optimizer,  epoch, device, args, cell_name_dict, drug_name_dict)
                
                # train_rmse, train_pcc, loss = train_step(model, train_loader, optimizer,  epoch, device, args, cell_name_dict, drug_name_dict)

                with open(loss_log_file, 'a') as loss_f:
                    loss_f.write(f'{epoch},{loss:.6f},{pred_loss:.6f},')
                    loss_f.write(f'{avg_intra_CL:.6f},{avg_cross_CL:.6f},{regularize_loss:.6f}\n')

                # ipdb.set_trace()

                f.write(f"Epoch {epoch+1},"
                            f"train_rmse: {train_rmse:.4f},"
                            f"train_pcc: {train_pcc:.5f},"
                            f"Total Loss: {loss:.4f}, \n"           
                        )

                # exit()
                if args.scheduler_type == 'ML':
                    scheduler.step()
                    
                current_lr = optimizer.param_groups[0]['lr']
                print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] '  + 
                            f'train_rmse: {train_rmse:.5f} ' +
                            f'train_pcc: {train_pcc:.5f} ' +  f'lr : {current_lr}')         
                
                print(print_msg)

                if epoch % args.check_step == 0:
                    val_rmse, val_pcc, _, _, _, _, _, _ = test_step(args, model, val_loader, device, cell_name_dict, drug_name_dict)

                    f.write(f"Epoch {epoch+1},"
                                    f"val_rmse: {val_rmse:.4f},"
                                    f"val_pcc: {val_pcc:.5f}\n"
                    )
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
                        
                        test_rmse, test_pcc, test_r_2, test_mae, test_y_true, test_y_pred, cell_names, drug_names = test_step(
                            args, model,test_loader, device, cell_name_dict, drug_name_dict)

                        best_test_pcc = test_pcc
                        best_test_rmse = test_rmse  
                        best_test_r2 = test_r_2
                        best_test_mae = test_mae


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
        

        print(f'Best epoch: {best_epoch:03d}, Best PCC: {best_test_pcc:.4f}')
        print(f'Best RMSE: {best_test_rmse:.4f}, Best R_2: {best_test_r2:.4f}, Best MAE: {best_test_mae:.4f}')

        pd.DataFrame([cell_names, drug_names, test_y_true.flatten(), test_y_pred.flatten()]).T.to_csv(
            f'./test_epoch_resnet_{epoch}.csv', index=False, header=['cell_name', 'drug_name', 'y_true', 'y_pred'])

        f.write(f"Best_Epoch {best_epoch},"
                f"best_test_rmse: {best_test_rmse:.4f},"
                f"best_test_pcc: {best_test_pcc:.5f},"
                f"best_test_r_2: {best_test_r2:.5f},"
                f"best_test_mae: {best_test_mae:.5f}\n")


        # load the model and record the response results
