from scipy.stats import entropy
from scipy.stats import wasserstein_distance
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def ours_loss(student_model, student_features, teacher_features, dist_type, top_k=3):
    # B, N, D
    B = student_features['block_0'].shape[0]

    if dist_type == 'wd': # Wasserstein Distance
        layer_dist_s = [] # layer 간 distance 저장
        layer_dist_t = []
        for i in range(11):
            print('block_{}'.format(i))
            sf_1 = student_features['block_{}'.format(i)].cpu().detach().numpy()
            sf_2 = student_features['block_{}'.format(i+1)].cpu().detach().numpy()
            # dist_s = sum([wasserstein_distance(sf_1[batch].flatten(), sf_2[batch].flatten()) for batch in range(B)])
            dist_s = torch.cdist(torch.tensor(sf_1.reshape(B, -1)), torch.tensor(sf_2.reshape(B, -1)), p=2).diag().sum()
            tf_1 = teacher_features['block_{}'.format(i)].cpu().detach().numpy()
            tf_2 = teacher_features['block_{}'.format(i+1)].cpu().detach().numpy()
            # dist_t = sum([wasserstein_distance(tf_1[batch].flatten(), tf_2[batch].flatten()) for batch in range(B)])
            dist_t = torch.cdist(torch.tensor(tf_1.reshape(B, -1)), torch.tensor(tf_2.reshape(B, -1)), p=2).diag().sum()
            layer_dist_s.append(dist_s)
            layer_dist_t.append(dist_t)
        
        dist, tok_k_idx = torch.topk(torch.tensor(layer_dist_t), top_k)
        loss_mse = nn.MSELoss(reduction='sum')
        loss_lr = 0

        for idx in tok_k_idx:
            idx = idx.item()
            loss_lr += loss_mse(torch.tensor(layer_dist_s[idx]), torch.tensor(layer_dist_t[idx]))/B
            return loss_lr
    if dist_type == 'kl': # Kullback-Leibler Divergence
        layer_dist_s = []
        layer_dist_t = []
        for i in range(11):
            sf_1 = student_features['block_{}'.format(i)].cpu().detach().numpy()
            sf_2 = student_features['block_{}'.format(i+1)].cpu().detach().numpy()
            sf_1 = sf_1 / np.sum(sf_1, axis=(1,2), keepdims=True)
            sf_2 = sf_2 / np.sum(sf_2, axis=(1,2), keepdims=True)
            dist_s = (sf_1 * np.log(sf_1/sf_2)).sum(axis=(1,2))

            tf_1 = teacher_features['block_{}'.format(i)].cpu().detach().numpy()
            tf_2 = teacher_features['block_{}'.format(i+1)].cpu().detach().numpy()
            tf_1 = tf_1 / np.sum(tf_1, axis=(1,2), keepdims=True)
            tf_2 = tf_2 / np.sum(tf_2, axis=(1,2), keepdims=True)
            dist_t = (tf_1 * np.log(tf_1/tf_2)).sum(axis=(1,2))
            epsilon = 1e-8  # 매우 작은 값 추가
            sf_1 = np.clip(sf_1, epsilon, None)  # 0 이하 값 방지
            sf_2 = np.clip(sf_2, epsilon, None)
            tf_1 = np.clip(tf_1, epsilon, None)
            tf_2 = np.clip(tf_2, epsilon, None)
            layer_dist_s.append(dist_s)
            layer_dist_t.append(dist_t)
        
        sum_dist_t = [dist.sum() for dist in layer_dist_t]
        dist, tok_k_idx = torch.topk(torch.tensor(sum_dist_t), top_k)
        loss_mse = nn.MSELoss(reduction='sum')
        loss_lr = 0

        for idx in tok_k_idx:
            idx = idx.item()
            loss_lr += loss_mse(torch.tensor(layer_dist_s[idx]), torch.tensor(layer_dist_t[idx]))/B

        return loss_lr
    
    if dist_type == 'test':  
        layer_dist_s = []
        layer_dist_t = []

        # w1 = 2.0
        # for i in range(11):
        for i in range(10):
            sf_1 = student_features[f'block_{i+1}'].clone().detach()
            sf_2 = student_features[f'block_{i+2}'].clone().detach()

            # 🔹 `torch.cdist()` 사용하여 거리 계산
            # cost_mat = torch.cdist(sf_1.reshape(B, -1), sf_2.reshape(B, -1), p=2)
            cost_mat = F.cosine_similarity(sf_1, sf_2, dim=1) # B, 196
            student_dist = torch.mean(cost_mat, dim=1)
            stduent_dist = torch.mean(student_dist, dim=1)
            # dist_s = w1 * cost_mat.diag().sum()/B
            # dist_s = cost_mat.diag().sum()/B
            tf_1 = teacher_features[f'block_{i+1}'].clone().detach()
            tf_2 = teacher_features[f'block_{i+2}'].clone().detach()

            cost_mat = F.cosine_similarity(tf_1, tf_2, dim=1) # B, 196
            teacher_dist = torch.mean(cost_mat, dim=1)
            teacher_dist = torch.mean(teacher_dist, dim=1)

            # cost_mat = torch.cdist(tf_1.reshape(B, -1), tf_2.reshape(B, -1), p=2)
            # dist_t = cost_mat.diag().sum()/B

            layer_dist_s.append(dist_s)
            layer_dist_t.append(dist_t)
            
            # w1 -= 0.1

        block_0_s = student_features['block_0'][:, 1:, :] # CLS token 제외
        block_0_t = teacher_features['block_0'][:, 2:, :] # CLS, DIST token 제외

        '''ViTKD: Mimicking'''
        if student_model.align2 is not None:
            for i in range(1):
                if i == 0:
                    xc = student_model.align2[i].to(block_0_s.device)(block_0_s).unsqueeze(1)
                else:
                    xc = torch.cat((xc, student_model.align2[i].to(block_0_s.device)(block_0_s[:,i]).unsqueeze(1)),dim=1)
        else:
            xc = block_0_s

        dist, top_k_idx = torch.topk(torch.tensor(layer_dist_t), top_k)
        loss_mse = nn.MSELoss(reduction='sum')
        loss_mimick = loss_mse(xc, block_0_t) / B * 0.001
        loss_l1 = nn.L1Loss(reduction='sum')
        loss_lr = 0

        for idx in top_k_idx:
            idx = idx.item()
            loss_lr += loss_l1(layer_dist_s[idx], layer_dist_t[idx])

        return loss_lr * 0.01 + loss_mimick 

    if dist_type == 'fitnet':   
        block_0_s = student_features['block_3'][:, 1:, :]
        block_0_t = teacher_features['block_5'][:, 2:, :]

        '''ViTKD: Mimicking'''
        if student_model.align2 is not None:
            for i in range(1):
                if i == 0:
                    xc = student_model.align2[i].to(block_0_s.device)(block_0_s[:,i]).unsqueeze(1)
                else:
                    xc = torch.cat((xc, student_model.align2[i].to(block_0_s.device)(block_0_s[:,i]).unsqueeze(1)),dim=1)
        else:
            xc = block_0_s

        loss_mse = nn.MSELoss(reduction='sum')

        loss_lr = loss_mse(xc, block_0_t) / B * 0.001

        return loss_lr
