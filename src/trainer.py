#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/29 13:54
# @Author  : Anonymous
# @Site    : 
# @File    : trainer.py
# @Software: PyCharm

# #Desc:
import math
import dill
import torch
import os
import time
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from utils import train_test_split, calculate_metrics, ddi_rate_score, llprint
from model.models import DrugRec,SimilarityConstraintLoss

class DrugTrainer():
    def __init__(self, config, device, datasets, new_datasets):
        super(DrugTrainer, self).__init__()
        self.use_cuda = config["USE_CUDA"]
        self.device = device
        self.ddi_adj, self.ddi_mask_H, self.data, self.molecule, self.voc = datasets
        self.drug_smile_matrix, self.smile_subs_matrix, self.smile_sub_voc, self.smile_sub_degree, self.smile_sub_recency = new_datasets

        self.root = config["ROOT"]
        self.task = config["TASK"]
        self.epochs = config["EPOCH"]
        self.embed_dim = config["DIM"]
        self.batch_sz = config["BATCH"]
        self.lr = config["LR"]
        self.wd = config["WD"]
        self.ratio = config["RATIO"]

        self.target_ddi = config["DDI"]
        self.kp = config['KP']
        self.model_name = config["MODEL"]
        self.log = config['LOG']
        self.sim_loss = SimilarityConstraintLoss()
        self.hist = config['HIST']

        self.result = {}

        self.get_data()

    def get_data(self):
        # voc
        self.diag_voc, self.pro_voc, self.med_voc = self.voc["diag_voc"], self.voc["pro_voc"], self.voc["med_voc"]
        self.smile_voc, self.sub_voc = self.smile_sub_voc['smile_voc'], self.smile_sub_voc['sub_voc']
        self.voc_size = (len(self.diag_voc.idx2word), len(self.pro_voc.idx2word), len(self.med_voc.idx2word), len(self.smile_voc), len(self.sub_voc))

        # device
        self.drug_smile_matrix = torch.tensor(self.drug_smile_matrix, dtype=torch.float).to(self.device)
        self.smile_subs_matrix = torch.tensor(self.smile_subs_matrix, dtype=torch.float).to(self.device)
        self.smile_sub_recency = torch.tensor(self.smile_sub_recency, dtype=torch.float).to(self.device)
        self.smile_sub_degree = torch.tensor(self.smile_sub_degree, dtype=torch.float).to(self.device)
        self.ddi_adj = torch.tensor(self.ddi_adj, dtype=torch.float).to(self.device)

        # train-test split
        split_point = int(len(self.data) * self.ratio)
        self.data_train, self.data_valid, self.data_test = train_test_split(self.data, split_point)



    def get_model(self):
        model = DrugRec(self.voc_size, self.embed_dim, self.drug_smile_matrix, self.smile_subs_matrix, self.ddi_adj, (self.smile_sub_recency, self.smile_sub_degree), self.device)
        model = model.to(self.device)
        return model

    def get_opt(self, model):
        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer

           

    def train(self, model, data, criterion, optimizer, epoch):
        print('Training Epoch {}:'.format(epoch))
        model.train()
        ddi_losses_epoch = []
        pre = [0.06 for i in range(self.hist)]
        pre_ddi = [0 for i in range(self.hist)]
        for step, input in enumerate(data): # patient
            loss = 0
            for idx, adm in enumerate(input): # visit
                seq_input = input[: idx+1] # k visit
                loss_bce_target = np.zeros((1, self.voc_size[2])) 
                loss_bce_target[:, adm[2]] = 1
                loss_multi_target = np.full((1, self.voc_size[2]), -1) 
                for idx, item in enumerate(adm[2]):
                    loss_multi_target[0][idx] = item 
                result, loss_ddi = model(seq_input) 
                loss_bce = F.binary_cross_entropy_with_logits(
                    result, torch.FloatTensor(loss_bce_target).to(self.device)
                )
                loss_multi = F.multilabel_margin_loss(
                    torch.sigmoid(result), torch.LongTensor(loss_multi_target).to(self.device)
                )

                sim_reg_loss = 0


                result = torch.sigmoid(result).detach().cpu().numpy()[0]
                result[result >= 0.5] = 1
                result[result < 0.5] = 0
                y_label = np.where(result == 1)[0] # [1, drug_num]
                current_ddi_rate = ddi_rate_score(
                    [[y_label]], self.ddi_adj
                )


                # # stack
                pre.pop(0)
                pre.append(current_ddi_rate)
                avg_ddi = sum(pre) / len(pre)
                
                pre_ddi.pop(0)
                pre_ddi.append(loss_ddi)
                avg_loss_ddi = sum(pre_ddi)/ len(pre_ddi)
                
                ddi_losses_epoch.append(loss_ddi.detach().cpu().item())

                if avg_ddi <= self.target_ddi: 
                    if current_ddi_rate <= self.target_ddi:
                        loss = 0.95 * loss_bce + 0.05 * loss_multi 
                    else:
                        loss = (
                                0.99 * (0.95 * loss_bce + 0.05 * loss_multi)
                                + 0.01 * loss_ddi #
                        )
                else:
                    beta = self.kp * (1 / (avg_ddi / self.target_ddi - 1 + 1e-8))
                    beta = min(math.tanh(beta), 1)

                    loss = (
                            beta * (0.95 * loss_bce + 0.05 * loss_multi)
                            + (1 - beta) * loss_ddi + (1 - beta) * sim_reg_loss
                    )

                
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.1, norm_type=2)
                optimizer.step()

            llprint("\rtraining step: {} / {}".format(step, len(data)))
        print()
        return sum(ddi_losses_epoch) / len(ddi_losses_epoch)



    def eval(self, model, data):
        print('Evaluating:')
        model.eval()
        smm_record = []
        ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
        med_cnt, visit_cnt = 0, 0

        for step, input in enumerate(data):
            y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
            for adm_idx, adm in enumerate(input):
                target_output, _ = model(input[: adm_idx + 1])
                y_gt_tmp = np.zeros(self.voc_size[2])
                y_gt_tmp[adm[2]] = 1
                y_gt.append(y_gt_tmp) # binary label

                # prediction prod
                target_output = torch.sigmoid(target_output).detach().cpu().numpy()[0]
                y_pred_prob.append(target_output)

                # prediction med set
                y_pred_tmp = target_output.copy()
                y_pred_tmp[y_pred_tmp >= 0.5] = 1
                y_pred_tmp[y_pred_tmp < 0.5] = 0
                y_pred.append(y_pred_tmp)

                # prediction label
                y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
                y_pred_label.append(sorted(y_pred_label_tmp))

                visit_cnt += 1 
                med_cnt += len(y_pred_label_tmp)
            smm_record.append(y_pred_label)

            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = calculate_metrics(
                np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
            )
            ja.append(adm_ja)
            prauc.append(adm_prauc)
            avg_p.append(adm_avg_p)
            avg_r.append(adm_avg_r)
            avg_f1.append(adm_avg_f1)
            llprint("\rtest step: {} / {}".format(step, len(data)))

        # ddi rate
        ddi_rate = ddi_rate_score(smm_record, self.ddi_adj)

        llprint(
            "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
                ddi_rate,
                np.mean(ja),
                np.mean(prauc),
                np.mean(avg_p),
                np.mean(avg_r),
                np.mean(avg_f1),
                med_cnt / visit_cnt,
            )
        )

        return (
            ddi_rate,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            med_cnt / visit_cnt,
        )


    def main(self, test=False, case=False):
        model = self.get_model()
        optimizer = self.get_opt(model)
        criterion = None 
        history = defaultdict(list)
        best_epoch, best_ja = 0, 0
        
        ddi_losses, ddi_values = [],[]

        for epoch in range(self.epochs):
            start = time.time()
            ddi_mean = self.train(model, self.data_train, criterion, optimizer, epoch)
            ddi_losses.append(ddi_mean)
            end = time.time()
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = self.eval(
                model, self.data_test)
            print(
                "all time: {}, test time: {}".format(
                    time.time() - start, time.time() - end
                )
            )
            ddi_values.append(ddi_rate)

            
            history["ja"].append(ja)
            history["ddi_rate"].append(ddi_rate)
            history["avg_p"].append(avg_p)
            history["avg_r"].append(avg_r)
            history["avg_f1"].append(avg_f1)
            history["prauc"].append(prauc)
            history["med"].append(avg_med)

            if epoch >= 5: 
                print(
                    "ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}".format(
                        np.mean(history["ddi_rate"][-5:]),
                        np.mean(history["med"][-5:]),
                        np.mean(history["ja"][-5:]),
                        np.mean(history["avg_f1"][-5:]),
                        np.mean(history["prauc"][-5:]),
                    )
                )

            torch.save(
                model.state_dict(),
                open(os.path.join(self.log, self.model_name + '-' + self.task,
                        "Epoch_{}_TARGET_{:.2}_JA_{:.4}_DDI_{:.4}.model".format(
                            epoch, self.target_ddi, ja, ddi_rate),),"wb",),)

            if epoch != 0 and best_ja < ja:
                best_epoch = epoch
                best_ja = ja

            print("best_epoch: {}".format(best_epoch))
            
        with open(os.path.join(
                self.log,
                    self.model_name + '-' + self.task, "ddi_loss_{}.txt".format(self.model_name)
            ), 'w') as Fout:
            for dloss, dvalue in zip(ddi_losses, ddi_values):
                Fout.write(f'{dloss}\t{dvalue}\n')            
            
        dill.dump(
            history,
            open(
                os.path.join(
                    self.log, self.model_name + '-' + self.task, "history2_{}.pkl".format(self.model_name)
                ),
                "wb",
            ),
        )
