import os 
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
from tqdm import tqdm
import numpy as np
import time
import re

import torch
import torch.optim as optim
import torch.nn.utils as utils
import torch.distributed as dist
import torch.multiprocessing as mp

from configure import args
from dataset_process import MoleculeDataset, MoleculeDatasetWrapper
from llama_model import TopModel
from criterion import NT_Xent, LOSS_FUNCTION_MATCH_DICT, EVALUE_FUNCTION_MATCH_DICT


def read_cot_from_file(file_path):
    print('loading cot texts from file ', file_path, '...')
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    pattern = r"idx:\s*(\d+)\nsmiles:\s*(.*?)\n(.*?)(?=\n=======================================================)"

    # search for all the pattern
    matches = re.findall(pattern, text, re.DOTALL)

    # save the idx and corresponding text block in a dictionary
    data = {}
    for match in matches:
        idx = int(match[0]) 
        smiles = match[1].strip()  
        text_block = match[2].strip()  
        data[idx] = {"smiles": smiles, "text": text_block}
    return data


def cot_text_generation(args, train_dataloader, valid_dataloader, test_dataloader):
    top_model.eval()
    
    all_cot_thinking_process_text = {}
    log_file = args.datasets[0] + '_cot_thinking_process_text.txt'
    
    print('train cot text generation...')
    with open(log_file, 'a') as file:
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            with torch.cuda.amp.autocast():
                smiles_prompt = batch['smiles_prompt'] # .to(args.device)
                cot_prompt = batch['cot_prompt'] # .to(args.device)
                predict_prompt = batch['predict_prompt'] # .to(args.device)
                mol_idx = batch['idx']
                smiles = batch['smile']
                
                for i in range(len(smiles_prompt)):
                    smiles_cot_prompt = smiles_prompt[i] + '\n' + predict_prompt[i] + " " + cot_prompt[i]
                    cot_thinking_process_text = top_model.cot_process_generate([smiles_cot_prompt])
                    all_cot_thinking_process_text[mol_idx[i].item()] = cot_thinking_process_text[0]
        
                    file.write("idx: %s\n" % str(mol_idx[i].item()))
                    file.write("smiles: %s\n" % smiles[i])
                    file.write("%s\n\n" % all_cot_thinking_process_text[mol_idx[i].item()])
                    file.write("=======================================================\n")
                    
    print('valid cot text generation...')
    with open(log_file, 'a') as file:
        for step, batch in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
            with torch.cuda.amp.autocast():
                smiles_prompt = batch['smiles_prompt'] # .to(args.device)
                cot_prompt = batch['cot_prompt'] # .to(args.device)
                predict_prompt = batch['predict_prompt'] # .to(args.device)
                mol_idx = batch['idx']
                smiles = batch['smile']
                
                for i in range(len(smiles_prompt)):
                    smiles_cot_prompt = smiles_prompt[i] + '\n' + predict_prompt[i] + " " + cot_prompt[i]
                    cot_thinking_process_text = top_model.cot_process_generate([smiles_cot_prompt])
                    all_cot_thinking_process_text[mol_idx[i].item()] = cot_thinking_process_text[0]
        
                    file.write("idx: %s\n" % str(mol_idx[i].item()))
                    file.write("smiles: %s\n" % smiles[i])
                    file.write("%s\n\n" % all_cot_thinking_process_text[mol_idx[i].item()])
                    file.write("=======================================================\n")
    
    print('test cot text generation...')
    with open(log_file, 'a') as file:
        for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            with torch.cuda.amp.autocast():
                smiles_prompt = batch['smiles_prompt'] # .to(args.device)
                cot_prompt = batch['cot_prompt'] # .to(args.device)
                predict_prompt = batch['predict_prompt'] # .to(args.device)
                mol_idx = batch['idx']
                smiles = batch['smile']
                
                for i in range(len(smiles_prompt)):
                    smiles_cot_prompt = smiles_prompt[i] + '\n' + predict_prompt[i] + " " + cot_prompt[i]
                    cot_thinking_process_text = top_model.cot_process_generate([smiles_cot_prompt])
                    all_cot_thinking_process_text[mol_idx[i].item()] = cot_thinking_process_text[0]
        
                    file.write("idx: %s\n" % str(mol_idx[i].item()))
                    file.write("smiles: %s\n" % smiles[i])
                    file.write("%s\n\n" % all_cot_thinking_process_text[mol_idx[i].item()])
                    file.write("=======================================================\n")
    
    return all_cot_thinking_process_text


def train(args, train_dataloader, cot_thinking_process_text, optimizer):
    top_model.train()
    
    loss_acc = 0
    # use thinking text to predict the label
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        with torch.cuda.amp.autocast():
            smiles_prompt = batch['smiles_prompt'] # .to(args.device)
            predict_prompt = batch['predict_prompt'] # .to(args.device)
            cot_prompt = batch['cot_prompt']
            atom_features_list = batch["atom_features_list"].to(args.device)
            edge_index = batch["edge_index"].to(args.device)
            edge_attr = batch["edge_attr"].to(args.device)
            gnn_batch = batch['batch'].to(args.device)
            print('gnn_batch: ', gnn_batch)
            mol_idx = batch['idx']
            
            prompt = []
            for i in range(len(smiles_prompt)):
                if cot_thinking_process_text[mol_idx[i]]['smiles'].strip().encode('utf-8') != batch['smile'][i].strip().encode('utf-8'):
                    print('smiles not match!')
                    print(cot_thinking_process_text[mol_idx[i]]['smiles'], batch['smile'][i])
                    return
                
                one_prompt = smiles_prompt[i] + '\n' + predict_prompt[i] + '\n\n' 
                if args.use_cot:
                    one_prompt = one_prompt + cot_prompt[i] + '\n' + cot_thinking_process_text[mol_idx[i]]['text']
                prompt.append(one_prompt)
                
            prompt_embd, gnn_emb, _, y_pred = top_model(prompt, atom_features_list, edge_index, edge_attr, gnn_batch)
            y_pred = y_pred.to(torch.float32).cpu()
            y_true = batch['label'].view(len(y_pred), -1).to(torch.float32).unsqueeze(2)

            criterion_contrast = NT_Xent(args.batch_size, args.loss_temperature, 1)
            loss_contrast = criterion_contrast(prompt_embd, gnn_emb)
            
            if batch['task_type'][0] == 'classification':
                y_valid = y_true ** 2 > 0
                y_true = ((1 + y_true) / 2)  # make the y_true back to 1 and 0 instead of i and -1
                masked_y_pred = torch.masked_select(y_pred, y_valid)
                masked_y_true = torch.masked_select(y_true, y_valid)
            else:
                masked_y_pred = y_pred
                masked_y_true = y_true

            loss1 = LOSS_FUNCTION_MATCH_DICT[batch['task'][0]](masked_y_pred, masked_y_true)
            loss = args.lambda_contrast * loss_contrast + loss1
            loss = loss.to(torch.float16)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_acc += loss.detach().item()
    batch_loss = loss_acc / len(train_dataloader)
    print('Loss: ', batch_loss)
    return batch_loss


def eval(args, loader, cot_thinking_process_text, evaluation_mode):
    top_model.eval()
    
    y_true_list = []
    y_pred_list = []
    y_embd_list = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(loader), total=len(loader)):
            with torch.cuda.amp.autocast():
                smiles_prompt = batch['smiles_prompt'] # .to(args.device)
                predict_prompt = batch['predict_prompt'] # .to(args.device)
                cot_prompt = batch['cot_prompt']
                atom_features_list = batch["atom_features_list"].to(args.device)
                edge_index = batch["edge_index"].to(args.device)
                edge_attr = batch["edge_attr"].to(args.device)
                gnn_batch = batch['batch'].to(args.device)
                cot_prompt = batch['cot_prompt']
                mol_idx = batch['idx']
            
                # use thinking text to predict the label
                prompt = []
                for i in range(len(smiles_prompt)):
                    one_prompt = smiles_prompt[i] + '\n' + predict_prompt[i] + '\n\n' + cot_prompt[i] + '\n' + cot_thinking_process_text[mol_idx[i]]['text']
                    prompt.append(one_prompt)
                    
                _, _, y_embd, y_pred = top_model(prompt, atom_features_list, edge_index, edge_attr, gnn_batch)
                y_pred = y_pred.to(torch.float32).cpu()
                y_true = batch['label'].view(len(y_pred), -1).to(torch.float16).unsqueeze(2)
                
                y_true_list.append(y_true.cpu())
                y_pred_list.append(y_pred.cpu())
                y_embd_list.append(y_embd.cpu())

        y_true_list = torch.cat(y_true_list, dim=0)
        y_pred_list = torch.cat(y_pred_list, dim=0)
        y_embd_list = torch.cat(y_embd_list, dim=0)
    
        result = EVALUE_FUNCTION_MATCH_DICT[batch['task'][0]](y_pred_list, y_true_list)
        print(f'{evaluation_mode}\t{result}')
        
        y_true_np = y_true_list.numpy()
        y_pred_np = y_pred_list.numpy()
        y_embd_np = y_embd_list.numpy()
        
    return result, y_true_np, y_pred_np, y_embd_np


def save_model(model, epoch, save_best=False):
    if not args.finetune_model_save_path == '':
        if save_best:
            print('saving best model...')
            torch.save(model, args.finetune_model_save_path + '_epoch' + str(epoch) + '_best.pth')
        else:
            torch.save(model, args.finetune_model_save_path + '_final.pth')
    return


if __name__ == '__main__':
    print('arguments\t', args)
    
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print('cuda: ', torch.__version__)
    print('cuda: ', torch.cuda.is_available())
    
    ############ Set up model ############
    top_model = TopModel(args)
    top_model = top_model.to(args.device)
    
    ########## Set up molecule dataset ##########
    root = '../datasets/'
    dataset = MoleculeDataset(args=args)
    dataset = MoleculeDatasetWrapper(
        dataset=dataset, 
        batch_size=args.batch_size, 
        num_gpu=args.world_size,
        valid_rate=args.valid_rate,
        test_rate=args.test_rate,
        split_type=args.split_type,
        num_workers=args.num_workers,
        split_seed=args.split_seed
        )

    train_dataloader, valid_dataloader, test_dataloader, total_dataloader = dataset.get_data_loaders()
    print(f"len(train_loader):{len(train_dataloader)}")
    print(f"len(valid_loader):{len(valid_dataloader)}")
    print(f"len(test_loader):{len(test_dataloader)}")
    
    ######### Set up optimization ##########
    parameters = list(top_model.parameters())
    optimizer = optim.Adam(parameters, lr=args.llm_lr, weight_decay=args.decay)
    scaler = torch.cuda.amp.GradScaler()
    
    ######### main train ##################
    best_valid = args.best_valid_initial[args.datasets[0]]
    best_valid_test = args.best_valid_test[args.datasets[0]]
    best_model = None
    early_stop_count = 0
    best_loss = 10000
    
    if args.generate_cot:
        cot_thinking_process_text = cot_text_generation(args, train_dataloader, valid_dataloader, test_dataloader)
    else:
        cot_thinking_process_text = read_cot_from_file(args.datasets[0]+'_cot_thinking_process_text.txt')
        representation_file_name = args.datasets[0]
        if args.use_cot == False:
            representation_file_name = representation_file_name + '_no_cot'
        if args.use_multimodal == False:
            representation_file_name = representation_file_name + '_no_mm'
        representation_file_name = representation_file_name + '_repr.npz'
        
        _, _, _, _ = eval(args, test_dataloader, cot_thinking_process_text, 'evaluate')
        for epoch in range(1, args.llm_epochs + 1):
            print('====epoch ' + str(epoch) + '====')
            start_time = time.time()
            loss = train(args, train_dataloader, cot_thinking_process_text, optimizer)

            if epoch % args.eval_every_n_epochs == 0:
                # task_y_train_dict_list = eval(args, train_dataloader, 'train')
                valid_result, valid_y_true, valid_y_pred, valid_y_embd = eval(args, valid_dataloader, cot_thinking_process_text, 'valid')
                test_result, test_y_true, test_y_pred, test_y_embd = eval(args, test_dataloader, cot_thinking_process_text, 'test')
                
                if (valid_result > best_valid and args.dataset_task_type[args.datasets[0]] == 'classification') or \
                    (valid_result < best_valid and args.dataset_task_type[args.datasets[0]] == 'regression'):
                    best_valid = valid_result
                    best_model = top_model.state_dict()
                    best_valid_test = test_result
                    
                    if args.save_repr:
                        train_result, train_y_true, train_y_pred, train_y_embd = eval(args, train_dataloader, cot_thinking_process_text, 'train')
                        np.savez(
                            representation_file_name, 
                            train_y_true=train_y_true, 
                            train_y_pred=train_y_pred, 
                            train_y_embd=train_y_embd, 
                            valid_y_true=valid_y_true, 
                            valid_y_pred=valid_y_pred, 
                            valid_y_embd=valid_y_embd, 
                            test_y_true=test_y_true, 
                            test_y_pred=test_y_pred, 
                            test_y_embd=test_y_embd
                        )
                        print('save best model representation...')
                    
                    save_model(best_model, epoch, save_best=True)
                    early_stop_count = 0
                    
                else:
                    early_stop_count = early_stop_count + args.eval_every_n_epochs
                
                if early_stop_count >= args.early_stop_epochs_llm:
                    print('validation result does not improvement for {} epochs, early stop! '.format(early_stop_count))
                    break
                        
                    
            print('{:.3f}s'.format(time.time() - start_time))
            print()
        print()
    
