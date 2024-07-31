#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/29 13:45
# @Author  : Anonymous
# @Site    : 
# @File    : main.py
# @Software: PyCharm

# #Desc: first run preprocess

import os
import torch
from config import config
from utils import seed_everything, dill_load
from trainer import DrugTrainer

# from draw.test import buildMPNN

def run_single_model(config):
    print("Hello DrugRec!")
    print(config)

    if config['USE_CUDA']==True:
        os.environ["CUDA_VISIBLE_DEVICES"] = config['GPU']
        device = torch.device("cuda:{}".format(config['GPU']))
    else:
        device = torch.device('cpu')

    print(device)
    # data initial
    data_path = config['ROOT'] + "ready/" + "records_final.pkl"
    voc_path = config['ROOT'] + "ready/" + "voc_final.pkl" # three dict tuple;
    ddi_adj_path = config['ROOT'] + "ready/" + "ddi_A_final.pkl" # side effect matrix
    ddi_mask_path = config['ROOT'] + "ready/" + "ddi_mask_H.pkl" # substructrue-drug
    molecule_path = config['ROOT'] + "ready/" + "atc3toSMILES.pkl" # atc(drug) to smiles lis

    drug_smile_path = config['ROOT'] + "ready/" + "drug_smile.pkl"
    smile_subs_path = config['ROOT'] + "ready/" + "smile_sub_b.pkl"
    smile_sub_voc_path = config['ROOT'] + "ready/" + "smile_sub_voc_b.pkl" # two dict tuple;
    smile_sub_degree_path = config['ROOT'] + "ready/" + "smile_sub_degree_b.pkl"
    smile_sub_recency_path = config['ROOT'] + "ready/" + "smile_sub_recency_b.pkl"

    ddi_adj, ddi_mask_H, data, molecule, voc = dill_load(ddi_adj_path), dill_load(ddi_mask_path), dill_load(data_path), dill_load(molecule_path), dill_load(voc_path)
    drug_smile_matrix, smile_subs_matrix, smile_sub_voc, smile_sub_degree, smile_sub_recency = dill_load(drug_smile_path), dill_load(smile_subs_path), dill_load(smile_sub_voc_path), dill_load(smile_sub_degree_path), dill_load(smile_sub_recency_path)

    # model initial
    trainer = DrugTrainer(config, device, (ddi_adj, ddi_mask_H, data, molecule, voc), (drug_smile_matrix, smile_subs_matrix, smile_sub_voc, smile_sub_degree, smile_sub_recency))

    trainer.main()

    # MPNNSet, N_fingerprint, average_projection = buildMPNN(
    #     molecule, voc["med_voc"].idx2word, 2, device
    # ) # 一个所有药物的三元组；fingerprint的数量，平均矩阵
    # print(molecule)
    # print(MPNNSet)


    print("Everything is OK!")


if __name__ == '__main__':
    config = config
    seed_everything(config['SEED'])
    run_single_model(config)
