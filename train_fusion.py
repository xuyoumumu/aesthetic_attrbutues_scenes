'''
    date:2020/10/15
    融合属性信息和场景预测，预测整体美学评分
'''


import json
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
from pathlib import Path
import argparse
from collections import defaultdict
from utils.data import read_data, create_dataloader
from model.resnet_multask_EVA import resnet_gap_features
from utils.cuda import cudarize
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def setup_data(train_path, val_path, img_folder_path, batch_size):
    train = read_data(train_path, img_folder_path)
    val = read_data(val_path, img_folder_path)
    train_dataset = create_dataloader(train, batch_size=batch_size, is_train=True, shuffle=True)
    val_dataset = create_dataloader(val, batch_size=1, is_train=False, shuffle=False)
    return train_dataset, val_dataset


def setup_model(use_cuda):
    model = resnet_gap_features()
    model = cudarize(model, use_cuda)
    return model
def setup_model_load(use_cuda):
    ssave_path = "F:/IQA/deep-photo-aesthetics-master/checkpoint/att_model" 
    #ssave_path = "F:/IQA/deep-photo-aesthetics-master/checkpoint/0" 

    #checkpoint = "epoch_9.loss_1.3016026020050049.pth"
    checkpoint = "epoch_0.loss_0.5562185472057711.pth"
    model = resnet_gap_features()
    model.load_state_dict(torch.load(f"{ssave_path}/{checkpoint}", map_location=lambda storage, loc: storage))
    model = cudarize(model, use_cuda)
    return model

def create_all_targets(data, attributes):
    targets = []
    for attr in attributes:
        targets.append(data[attr])
    targets = Variable(torch.cat(targets, dim=1))
    #print(targets)
    return targets
def create_score_targets(data, attributes):    #产生label
    targets = []
    #print(attributes)
    for attr in attributes:
        if attr=='score':
            targets.append(data[attr])
    targets = Variable(torch.cat(targets, dim=1)).float()
    return targets

def update_results(epoch, predictions, targets, loss, weights, all_attributes, loss_data_for_df):
    masked_loss = loss.detach().numpy()
    current_batch_size = loss.size()[0]
    #print(current_batch_size)
    #print(all_attributes)
    #print(loss_data_for_df)
    for i in range(current_batch_size):
        loss_data_for_df[all_attributes[0]].append(masked_loss[i])
        loss_data_for_df["epoch"].append(epoch)


def train(train, val, model, loss_weights, n_epochs, use_cuda, save_path,
          fc_lr, fine_tune_lr):
    save_path = Path(save_path)
    scene_key = ['ContentType']
    attribute_keys = ['BalancingElements', 'ColorHarmony', 'Content', 'DoF',
                      'Light',  'Object', 'RuleOfThirds', 'VividColor']
    non_negative_attribute_keys = [ 'score']
    all_attributes = attribute_keys + non_negative_attribute_keys
    base_params_0 = model.weights
    base_params_1 =  model.Score_1.parameters() 
    print(0)
    base_params_2 =  model.Score_2.parameters() 
    base_params_3 =  model.Score_3.parameters() 
    optimizer = torch.optim.Adam([
        {'params': base_params_0},
        {'params': base_params_1},
        {'params': base_params_2},
        {'params': base_params_3},
        ], lr=fc_lr)


    criterion = nn.MSELoss(reduce=False)


   

    for epoch in range(n_epochs):
        score_loss = []
        score_corr = []
        for data in tqdm(train):
            model.train()
            inp = cudarize(Variable(data['image']), use_cuda)
            Score = model(inp)[2]   #使用第二个返回值
            target = cudarize(create_score_targets(data, all_attributes).mean(), use_cuda)
    
            S_0 = []
            loss = criterion(Score, target)
            S = Score.detach().numpy()
            S = S.tolist()
            
            S_0.append(S)
            T_0 = []

            
            T= target.detach().numpy()
            T = T.tolist()
            
           
        
            T_0.append(T)
           
            corr = pearsonr(S, T)
            print(corr)
        
            score_corr.append(corr[0]) 
            
            
            loss_by_attribute = torch.sum(loss, dim=0).unsqueeze(0)
            weights = torch.zeros(1,1)
            weights[0] = 1
            lloss = loss_by_attribute * weights
            lloss = lloss.detach().numpy()
            lloss = lloss.mean()
            score_loss.append(lloss)
            


            # Update gradients
            optimizer.zero_grad()

            # Method 1:
            torch.autograd.backward(loss_by_attribute, weights)

            optimizer.step()

        score_loss = pd.DataFrame(score_loss)
        score_loss = score_loss.mean()
        sscore_loss = float(score_loss)
        

        score_loss.to_csv(f"{save_path}/results.csv",mode = 'a')
        score_corr = pd.DataFrame(score_corr)
        score_corr = score_corr.mean()
        score_corr.to_csv(f"{save_path}/results.csv",mode = 'a')

        torch.save(model.state_dict(), f"{save_path}/score_epoch_{epoch}.loss_{sscore_loss}.pth")


        print(f"\nScore Loss :{sscore_loss}")
        print(f"\nScore corr :{score_corr}")



       
       

        



        
        


        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", default="F:/IQA/deep-photo-aesthetics-master/att_config.json")
    opts = parser.parse_args()
    with open(opts.config_file_path, "r") as fp:
        config = json.load(fp)
    train_dataset, val_dataset = setup_data(config['train_path'],
                                            config['val_path'],
                                            config['img_folder_path'],
                                            config['batch_size'],
                                            )
    model = setup_model_load(config['use_cuda'])
    train(train_dataset, val_dataset, model,
          config['loss_weights'], config['n_epochs'],
          config['use_cuda'], config['save_path'],
          config['fc_lr'], config['fine_tune_lr'])

