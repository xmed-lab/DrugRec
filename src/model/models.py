#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/5 14:05
# @Author  : Anonymous
# @Site    : 
# @File    : models.py
# @Software: PyCharm

# #Desc: dropout 0.6  
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import build_map, get_indices, get_nonzero_values
from torch.nn.utils.rnn import pad_sequence

class SimilarityConstraintLoss(nn.Module):
    def __init__(self, threshold=0.5):
        super(SimilarityConstraintLoss, self).__init__()
        self.threshold = threshold

    def forward(self, predictions):
        above_threshold = (predictions >= self.threshold)
        num_above_threshold = torch.sum(above_threshold, dim=1, keepdim=True)
        equal_probs = torch.ones_like(predictions) / num_above_threshold

        similarity_probs = torch.where(above_threshold, equal_probs, torch.zeros_like(predictions))
        similarity_probs /= torch.sum(similarity_probs, dim=1, keepdim=True)

        # cross
        masked_predictions = torch.where(above_threshold, predictions, torch.zeros_like(predictions))

        # sim
        similarity_loss = torch.mean(torch.sum(torch.pow(similarity_probs - masked_predictions, 2), dim=1))

        return similarity_loss



class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.2):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention layer
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward layer
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    
    
class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TransformerEncoder, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=input_size, nhead=2, dim_feedforward=hidden_size) 
        self.transformer = nn.TransformerEncoder(self.encoder, num_layers=num_layers)


    def forward(self, x, src_mask):
        """
        :param x: [Smile_num, seq_len, dim]: seq_len=max(sub_num)
        :param src_mask: [Smile_num, seq_len]: seq_len=max(sub_num): bool, mask为true
        :return: [Smile_num, embedding]
        """
        x = x.transpose(0, 1) # seq_len, batch, dim
        x = self.transformer(x, src_key_padding_mask=src_mask)
        # print(src_mask.sum(dim=1)) 

        x = x.transpose(0, 1) # b,seq, dim
        mask = ~src_mask
        avg = torch.sum(mask, dim=1).unsqueeze(dim=1) # smile_num, 1
        x = torch.bmm(mask.unsqueeze(1).float(), x).squeeze() # Smile_num, dim; 
        x = x / avg
        return x

    

class SubGraphNetwork(nn.Module):
    def __init__(self, sub_embed, emb_dim, smile_num):
        
        super(SubGraphNetwork, self).__init__()
        self.sub_emb = sub_embed # sub_num = [unknown + voc]

        self.transformer = TransformerEncoder(emb_dim, emb_dim, 2) 
        self.smile_emb_id = nn.Embedding(smile_num, emb_dim)
        self.recency_emb = nn.Embedding(30, emb_dim) 
        self.degree_emb = nn.Embedding(30, emb_dim)
        self.dropout = nn.Dropout(p=0.7, inplace=False) 

        self.init_weights()

        self.ff = nn.Linear(emb_dim, emb_dim, bias=False)
        self.ff2 = nn.Linear(emb_dim, emb_dim, bias=False)



    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        nn.init.xavier_uniform_(self.smile_emb_id.weight)
        nn.init.xavier_uniform_(self.recency_emb.weight)
        nn.init.xavier_uniform_(self.degree_emb.weight)


    def get_embedding(self, smile_sub_matrix, smile_sub_recency, smile_sub_degree):
        # smile-sub indice
        sub_ids, lens = get_indices(smile_sub_matrix) # [],
        # print(len(sub_ids))
        smile_sub_seq = [v for v in torch.split(sub_ids, lens)] # [[],[]]
        padded_sequences = pad_sequence(smile_sub_seq, batch_first=True, padding_value=0).long() # [smile, max_subs+1]
        sub_embs = self.sub_emb(padded_sequences) # smile_num, max_sub_num+1, dim
        mask = (padded_sequences == 0).bool()

        # structure embedding
        smile_sub_recency = get_nonzero_values(smile_sub_recency)
        smile_sub_recency = [torch.sort(v)[1] for v in torch.split(smile_sub_recency, lens)] 
        padded_sequences = pad_sequence(smile_sub_recency, batch_first=True, padding_value=0).long()
        recency_embs = self.recency_emb(padded_sequences) # smile_num, sub_num+1, dim

        smile_sub_degree = get_nonzero_values(smile_sub_degree)
        smile_sub_degree = [torch.sort(v)[1] for v in torch.split(smile_sub_degree, lens)]
        padded_sequences = pad_sequence(smile_sub_degree, batch_first=True, padding_value=0).long()
        degree_embs = self.degree_emb(padded_sequences) # smile_num, sub_num+1, dim

        # intergrate
        emb = self.dropout(self.ff(recency_embs + degree_embs)) + sub_embs 


        return emb, mask


    def graph_transformer(self, input, src_mask):
        """
        input: [sub,sub,sub], [pos, pos, pos];
        :return:
        """
        smile_rep = self.transformer(input, src_mask)
        return smile_rep


    def sum(self, smile_embs, split_point):
        """
        split_point: list[3,4,2]
        """
        sum_vectors = [torch.sum(v, 0) for v in torch.split(smile_embs, split_point)] 
        return torch.stack(sum_vectors) # molecular , B

    def forward(self, inputs, query=None):
        """
        sub_dicts:,
        smile_sub_matrix:,
        adjacencies:,
        molecular_sizes:
        """
        smile_sub_matrix, recency_matrix, degree_matrix, drug_smile_matrix = inputs
        emb, mask = self.get_embedding(smile_sub_matrix, recency_matrix, degree_matrix) 
        smile_rep = self.graph_transformer(emb, mask)
        smile_rep = smile_rep + self.smile_emb_id.weight
        drug_attn = F.softmax(torch.matmul(query, smile_rep.t()), dim=-1) 
        drug_rep = drug_attn.matmul(smile_rep)
        drug_rep = F.normalize(drug_rep, p=2, dim=1)

        return drug_rep



class SubAttention(nn.Module):
    """whole/before history->repeat/explore module"""
    def __init__(self, query_size, key_size, hidden_size):
        super(SubAttention, self).__init__()
        self.linear_key = nn.Linear(key_size, hidden_size, bias=False)
        self.linear_query = nn.Linear(query_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.hidden_size = hidden_size

    def matching(self, query, key, mask=None):
        '''
        :param query: [batch_size, query_seq_len, query_size]
        :param key: [batch_size, key_seq_len, key_size]
        :param mask: [batch_size, query_seq_len, key_seq_len]
        :return: [batch_size, query_seq_len, key_seq_len]
        '''
        wq = self.linear_query(query)
        wq = wq.unsqueeze(-2)
        uh = self.linear_key(key)
        uh = uh.unsqueeze(-3) 
        wuc = wq + uh
        wquh = torch.tanh(wuc)
        attn = self.v(wquh).squeeze(-1)
        if mask is not None:
            attn = attn.masked_fill(~mask, -float('inf'))
        return attn # B, 1, sub

    def score(self, query, key, softmax_dim=-1, mask=None):
        attn = self.matching(query, key, mask) # B,1,sub
        norm_attn = F.softmax(attn, dim=softmax_dim)
        if mask is not None:
            norm_attn = norm_attn.masked_fill(~mask, 0)
        return attn, norm_attn

    def forward(self, query, key, value, mask=None):
        '''
        :param query: [batch_size, query_seq_len, query_size]
        :param key: [batch_size, key_seq_len, key_size]
        :param value: [batch_size, value_seq_len=key_seq_len, value_size]
        :param mask: [batch_size, query_seq_len, key_seq_len]
        :return: [batch_size, query_seq_len, value_size]
        '''
        attn, norm_attn = self.score(query, key, mask=mask) # attn score
        h = torch.bmm(norm_attn.view(-1, norm_attn.size(-2), norm_attn.size(-1)), value.view(-1, value.size(-2), value.size(-1)))

        return h.view(list(value.size())[:-2]+[norm_attn.size(-2), -1]), attn, norm_attn



class DrugRec(nn.Module):
    def __init__(self, vocab_size, emb_dim, drug_smile, smile_sub, ddi_matrix, structure_matrix, device):
        super(DrugRec, self).__init__()
        print(vocab_size[-2], vocab_size[-1])
        self.emb_dim = emb_dim
        self.device = device
        self.diagnose_emb = nn.Embedding(vocab_size[0], emb_dim)
        self.procedure_emb = nn.Embedding(vocab_size[1], emb_dim)

        self.dropout = nn.Dropout(p=0.7, inplace=False)
        self.diagnose_encoder = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.procedure_encoder = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.inter_encoder = nn.GRU(emb_dim, emb_dim, batch_first=True)


        self.drug_smile = drug_smile # matrix
        self.smile_sub = smile_sub # matrix
        self.drug_sub_martix = (self.drug_smile @ self.smile_sub).bool().float()
        self.sub_embed = nn.Embedding(vocab_size[-1], emb_dim)  

        self.query = nn.Sequential(nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim))



        # # graph embedding
        self.graph_network = SubGraphNetwork(self.sub_embed, emb_dim,self.drug_smile.shape[1]).to(self.device)
        self.stru = structure_matrix
        # self.drug_emb = self.graph_network([self.smile_sub, structure_matrix[0], structure_matrix[1], self.drug_smile]) # drug emb_dim
        self.drug_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.drug_layernorm = nn.LayerNorm(vocab_size[2])

        # distribution
        self.W_s_r = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_mu_r = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_sigma_r = nn.Linear(emb_dim, emb_dim, bias=False)

        self.W_s_e = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_mu_e = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_sigma_e = nn.Linear(emb_dim, emb_dim, bias=False)

        # repeat / explore
        self.sub_num_all = vocab_size[-1]
        self.repeat_attn = SubAttention(emb_dim, emb_dim, emb_dim)
        self.explore_attn = SubAttention(emb_dim, emb_dim, emb_dim)
        self.explore = nn.Linear(emb_dim, self.sub_num_all)

        self.inter_attn = SubAttention(emb_dim, emb_dim, emb_dim)
        self.inter = nn.Linear(emb_dim, 2)
        self.his_attn = nn.Linear(emb_dim, 1)
        self.his_softmax = nn.Softmax(dim=1)

        # ddi
        self.ddi_matrix = ddi_matrix

        self.drug_emb_id = nn.Embedding(self.drug_smile.shape[0], self.emb_dim)


        self.init_weights()


    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        # nn.init.xavier_uniform_(self.query.weight)
        # nn.init.xavier_uniform_(self.diagnose_emb.weight)
        # nn.init.xavier_uniform_(self.procedure_emb.weight)
        # nn.init.xavier_uniform_(self.sub_embed.weight)
        # nn.init.xavier_uniform_(self.drug_emb_id.weight)
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
        self.apply(_init_weights)
        

    def generateEpsilon(self):
        return torch.randn(size=(self.emb_dim, )).to(self.device)

    def distribution(self, health_change, W_s, W_mu, W_sigma):
        """分布形式"""
        health_change = F.relu(W_s(health_change))
        mu_i, sigma_i = W_mu(health_change), torch.exp(W_sigma(health_change))
        epsilon = self.generateEpsilon()
        health_change = mu_i + epsilon * sigma_i
        return health_change


    def repeat_forward(self, his_query, sub_emb_before, drug_sub_matrix, source_map):
        """
        :param his_query: [seq, dim]
        :param sub_emb_before: [1, before_sub_num, dim]; 
        :param drug_sub_matrix: [drug_num, sub_num]
        :param source_map: [1, before_sub_num, voc_sub_all] 
        :return:
        """
        before_query, cur_query = his_query[-2:-1, :], his_query[-1:, :]
        vary_health = (cur_query - before_query).unsqueeze(dim=1) # [1,1,dim]
        vary_health = self.distribution(vary_health, self.W_s_r, self.W_mu_r, self.W_sigma_r) 
        sub_emb_before = sub_emb_before # 1, sub_num, dim
        vary_health = F.dropout(vary_health, p=0.7, training=self.training, inplace=False)
        sub_emb_before = F.dropout(sub_emb_before, p=0.7, training=self.training, inplace=False)

        _, p_repeat = self.repeat_attn.score(vary_health, sub_emb_before) # 1, 1, sub_num
        p_repeat = torch.bmm(p_repeat, source_map).squeeze(1) # 1,1,sub * 1,sub,voc-> 1,voc_sub
        
        
        drug_repeat_emb = torch.sigmoid(torch.mm(p_repeat, drug_sub_matrix.T)) # 1, drug_voc

        return drug_repeat_emb

    def explore_forward(self, his_query, sub_emb_before, drug_sub_matrix, source_map, explore_mask=None):
        before_query, cur_query = his_query[-2:-1, :], his_query[-1:, :]
        vary_health = (cur_query - before_query).unsqueeze(dim=1)  # [1,1,dim]
        vary_health = self.distribution(vary_health, self.W_s_r, self.W_mu_r, self.W_sigma_r)

        sub_emb_before =  sub_emb_before # 1, sub_num, dim
        vary_health = F.dropout(vary_health, p=0.7, training=self.training, inplace=False)
        sub_emb_before = F.dropout(sub_emb_before, p=0.7, training=self.training, inplace=False)

        explore_feature, attn, norm_attn = self.explore_attn(vary_health, sub_emb_before, sub_emb_before) # [1,1,dim]; [1,1,sub]
        p_explore = self.explore(explore_feature.squeeze(1)) # [1,1,dim]->[1,voc_sub]
        if explore_mask is not None:
            p_explore = p_explore.masked_fill(explore_mask.bool(), float(
                '-inf'))  # not sure we need to mask this out, depends on experiment results

        p_explore = F.softmax(p_explore, dim=-1) # 1,voc_sub

        drug_explore_emb = torch.sigmoid(torch.mm(p_explore, drug_sub_matrix.T)) # 1, drug_voc
        return drug_explore_emb

    def inter_forward(self, his_query, sub_emb_before):
        before_query, cur_query = his_query[-2:-1, :], his_query[-1:, :] 
        vary_health = (cur_query - before_query).unsqueeze(dim=1)  # [1,1,dim]
        # vary_health = torch.cat((cur_query,before_query),dim=-1).unsqueeze(dim=1) # [1,1,dim]，
        # vary_health = self.distribution(vary_health, self.W_s_r, self.W_mu_r, self.W_sigma_r) 

        # vary_health = self.his_softmax(self.his_attn(his_query).view(1, -1)) @ his_query # [1,seq] * [seq,dim], 
        sub_emb_before =  sub_emb_before # 1, sub_num, dim
        vary_health = F.dropout(vary_health, p=0.7, training=self.training, inplace=False)
        sub_emb_before = F.dropout(sub_emb_before, p=0.7, training=self.training, inplace=False)

        mode_feature, attn, norm_attn = self.inter_attn(vary_health, sub_emb_before, sub_emb_before) # 1，1，dim
        p_mode = F.softmax(self.inter(mode_feature.squeeze(1)), dim=-1) # [1,2]
        return p_mode

    def forward(self, input):
        """
        :param input: patient health [[[],[],[]], [[],[],[]]]
        :param drug_his: patient drug history

        :return: prob, ddi rate
        """

        def sum_embedding(embedding): 
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        # patient health representation
        diag_seq = []
        proc_seq = []

        # preprocess
        for adm in input:
            diag = sum_embedding(
                self.dropout(
                    self.diagnose_emb(
                        torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )  # (1,1,dim)
            proc = sum_embedding(
                self.dropout(
                    self.procedure_emb(
                        torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )
            diag_seq.append(diag)
            proc_seq.append(proc)
        diag_seq = torch.cat(diag_seq, dim=1)  # (1,seq,dim)
        proc_seq = torch.cat(proc_seq, dim=1)  # (1,seq,dim)

        o1, h1 = self.diagnose_encoder(diag_seq) # o: all，h: last
        o2, h2 = self.procedure_encoder(proc_seq)

        patient_repr = torch.cat([o1, o2], dim=-1).squeeze(
            dim=0)  # (seq, dim*2)
        
        his_query = self.query(patient_repr)
        query = his_query[-1:, :]

        self.drug_emb = self.graph_network([self.smile_sub, self.stru[0], self.stru[1], self.drug_smile], self.drug_emb_id.weight) 
        self.drug_emb = self.drug_emb + self.drug_emb_id.weight
        
        

        # m1: input to all drugs
        drug_match = torch.sigmoid(torch.mm(query, self.drug_emb.T)) 

        drug_match = self.drug_layernorm(drug_match + self.drug_output(drug_match)) # 1, drug;

        if len(input) !=1:
            # drug- substructure emb[drug, sub] [sub, emb]
            his_med = input[-2][-1] # last
            # his_med = list(itertools.chain.from_iterable([i[-1] if isinstance(i[-1], tuple) else i[-1] for i in input[:-1]])) # all
            drug_sub_relation = self.drug_sub_martix[  # [drug, all_sub]
                torch.LongTensor(his_med).to(self.device)] 
            nonzero_indices = torch.nonzero(drug_sub_relation).to(self.device)
            sub_indices = nonzero_indices[:, 1].unique().long()
            source_map = build_map(sub_indices.unsqueeze(dim=0), max=self.sub_num_all) # 1, sub, voc_sub;

            med = self.sub_embed(sub_indices.unsqueeze(dim=0)) # 1, sub_num, emb;
            # m2: repeat forward
            repeat_prob = self.repeat_forward(his_query, med, self.drug_sub_martix, source_map)

            # m3: explore forward
            explore_mask = torch.bmm(sub_indices.unsqueeze(dim=0).float().unsqueeze(1), source_map).squeeze(1) # 1, sum_voc
            # explore_mask = None
            explore_prob = self.explore_forward(his_query, med, self.drug_sub_martix, source_map, explore_mask=explore_mask) 

            # intergrate m2, m3- > m2; m1, m2 ->m1
            p_mode = self.inter_forward(his_query, med)
            satis_match = p_mode[:, 0].unsqueeze(-1) * repeat_prob + p_mode[:, 1].unsqueeze(-1) * explore_prob
        else:
            satis_match = torch.ones_like(drug_match).to(self.device)

        result = torch.mul(satis_match, drug_match) # 1, drug
                

        # calculate ddl
        neg_pred_prob = torch.sigmoid(result) 
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
        ddi_rate = 0.0005 * neg_pred_prob.mul(self.ddi_matrix).sum()

        return result, ddi_rate

