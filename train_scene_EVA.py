'''
    date:2020/10/14
    训练多任务网络用于预测场景分布的分支
    使用EVA数据集
'''


import json
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
from pathlib import Path
import argparse
from collections import defaultdict
from utils.scene_data import read_data, create_dataloader
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
    #ssave_path = "F:/IQA/deep-photo-aesthetics-master/checkpoint/scene_model/0" 
    ssave_path = "F:/IQA/deep-photo-aesthetics-master/checkpoint/0" 

    #checkpoint = "epoch_9.loss_1.3016026020050049.pth"
    checkpoint = "epoch_0.loss_0.5605070052608367.pth"
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
    #scene_keys = ['Animal', 'Architecture', 'Human', 'NaturalScene','Still Life']
    scene_key = ['ContentType']
    ignored_params = list(map(id, model.content_weights.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())

    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': model.content_weights.parameters(), 'lr': fc_lr, 'weight_decay': 1e-5}
    ], lr=fine_tune_lr)
    criterion = nn.CrossEntropyLoss(reduce=False)

    # Define the weights that are needed on later
    weights = torch.zeros(1, len(scene_key))
    for i, scene in enumerate(scene_key):
        weight = loss_weights[scene]
        weights[0, i] = weight
        

    weights = cudarize(weights, use_cuda)

    train_loss = []
    train_corr = []
    val_acc = []
    for epoch in range(n_epochs):
        train_loss_data_for_epoch = defaultdict(list)
        val_loss_data_for_epoch = defaultdict(list)

        for data in tqdm(train):
            model.train()
            inp = cudarize(Variable(data['image']), use_cuda)
            predictions = model(inp)[1]
            #print(predictions)
            targets = cudarize(create_all_targets(data, scene_key), use_cuda).squeeze(1)
            #print(targets)
            loss = criterion(predictions, targets)
            #print(loss)
            #loss_by_attribute = torch.sum(loss, dim=0).unsqueeze(0)
            #print(loss_by_attribute)
            # Update results
            update_results(epoch, predictions, targets, loss, weights,
                           scene_key, train_loss_data_for_epoch)

            # Update gradients
            optimizer.zero_grad()
            
            # The two methods below are equivalent!
            
            # Method 1:
            #torch.autograd.backward(loss)
            #print(loss)
            loss = torch.sum(loss) 
            loss.backward()

            # Method 2:
            # masked_loss = loss_by_attribute * weights
            # masked_loss = torch.sum(masked_loss)
            # masked_loss.backward()

            optimizer.step()

        train_loss_df_for_epoch = pd.DataFrame(train_loss_data_for_epoch)
        train_loss.append(train_loss_df_for_epoch)
        #print(train_loss_df_for_epoch.mean())
        print(f"\nTraining Loss :\n{train_loss_df_for_epoch.mean()}")


        acc = 0
        val_num = 2
        if epoch/val_num == 0:
            for data in tqdm(val):
                
                model.eval()
                inp = cudarize(Variable(data['image']), use_cuda)
                predictions = model(inp)[1]
                targets = cudarize(create_all_targets(data, scene_key), use_cuda).squeeze(1)
                predictions = predictions.detach().numpy()
                idx = np.argmax(predictions, axis=1)
                targets = targets.detach().numpy()
                if idx ==targets:
                    acc+=1
            acc_rate = acc/len(val)
            val_acc.append(acc_rate)
            

            print(f"Accuracy : {acc_rate}")
       

        



        
        


        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), f"{save_path}/epoch_{epoch}.loss_{train_loss_df_for_epoch.mean()['ContentType']}.pth")


    train_loss = pd.DataFrame(pd.concat(train_loss))
    #val_loss = pd.DataFrame(pd.concat(val_loss))
    train_loss.to_csv(f"{save_path}/train_results.csv")
    #val_loss.to_csv(f"{save_path}/val_results.csv")

    val_acc_df = pd.DataFrame(val_acc)
    val_acc_df.to_csv(f"{save_path}/train_acc.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", default="F:/IQA/deep-photo-aesthetics-master/EVA_scene_config.json")
    opts = parser.parse_args()
    with open(opts.config_file_path, "r") as fp:
        config = json.load(fp)
    train_dataset, val_dataset = setup_data(config['train_path'],
                                            config['val_path'],
                                            config['img_folder_path'],
                                            config['batch_size'],
                                            )
    model = setup_model(config['use_cuda'])
    train(train_dataset, val_dataset, model,
          config['loss_weights'], config['n_epochs'],
          config['use_cuda'], config['save_path'],
          config['fc_lr'], config['fine_tune_lr'])

