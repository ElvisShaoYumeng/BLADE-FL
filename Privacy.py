# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 20:37:06 2019

@author: WEIKANG
"""

import numpy as np
import copy
import torch
import math
from Calculate import f_zero

def get_1_norm(params_a):
    sum = 0
    for i in params_a.keys():
        if len(params_a[i]) == 1:
            sum += pow(np.linalg.norm(params_a[i].cpu().numpy(), ord=2),2)
        else:
            a = copy.deepcopy(params_a[i].cpu().numpy())
            for j in a:
                x = copy.deepcopy(j.flatten())
                sum += pow(np.linalg.norm(x, ord=2),2)
    norm = np.sqrt(sum)
    return norm

def get_2_norm(params_a, params_b):
    sum = 0
    for i in params_a.keys():
        if len(params_a[i]) == 1:
            sum += pow(np.linalg.norm(params_a[i].cpu().numpy()-\
                params_b[i].cpu().numpy(), ord=2),2)
        else:
            a = copy.deepcopy(params_a[i].cpu().numpy())
            b = copy.deepcopy(params_b[i].cpu().numpy())
            for j in range(len(a)):
                x=copy.deepcopy(a[j].flatten())         
                y=copy.deepcopy(b[j].flatten())
                sum += pow(np.linalg.norm(x-y, ord=2),2)            
    norm = np.sqrt(sum)
    return norm

def inner_product(params_a, params_b):
    sum = 0
    for i in params_a.keys():
        sum += np.sum(np.multiply(params_a[i].cpu().numpy(),\
                params_b[i].cpu().numpy()))     
    return sum

def avg_grads(g):
    grad_avg = copy.deepcopy(g[0])
    for k in grad_avg.keys():
        for i in range(1, len(g)):
            grad_avg[k] += g[i][k]
        grad_avg[k] = torch.div(grad_avg[k], len(g))
    return grad_avg

def calculate_grads(args, w_before, w_new):
    grads = copy.deepcopy(w_before)
    for k in grads.keys():
        grads[k] =(w_before[k]-w_new[k]) * 1.0 / args.lr
    return grads

def para_estimate(args, list_loss, loss_locals, w_glob_before, w_locals_before,\
                  w_locals, w_glob):
    Lipz_c = []
    Lipz_s = []
    beta = []
    delta = []
    norm_grads_locals = []
    Grads_locals = copy.deepcopy(w_locals)
    for idx in range(args.num_Chosenusers):
        ### Calculate▽F_i(w(t))=[w(t)-w_i(t)]/lr ###
        Grads_locals[idx] = copy.deepcopy(calculate_grads(args, w_glob, w_locals[idx]))
    ### Calculate▽F(w(t)) ###    
    Grads_glob =  copy.deepcopy(avg_grads(Grads_locals))    
    for idx in range(args.num_Chosenusers):
        ### Calculate ||w(t-1)-w(t)|| ###
        diff_weights_glob = copy.deepcopy(get_2_norm(w_glob_before, w_glob))
        ### Calculate ||▽F_i(w(t-1))-▽F_i(w(t))|| ###
        diff_grads = copy.deepcopy(get_2_norm(calculate_grads(args, w_glob_before, \
                w_locals_before[idx]), calculate_grads(args, w_glob, w_locals[idx])))
        ### Calculate ||w(t)-w_i(t)|| ###
        diff_weights_locals = copy.deepcopy(get_2_norm(w_glob, w_locals[idx]))
        ### Calculate ||▽F(w(t))-▽F_i(w(t))|| ###
        Grads_variance = copy.deepcopy(get_2_norm(Grads_glob, Grads_locals[idx]))
        ### Calculate ||▽F(w(t))|| ###
        norm_grads_glob = copy.deepcopy(get_1_norm(Grads_glob))
        ### Calculate ||▽F_i(w(t))|| ###
        norm_grads_locals.append(copy.deepcopy(get_1_norm(Grads_locals[idx])))
        ### Calculate Lipz_s=||▽F_i(w(t-1))-▽F_i(w(t))||/||w(t-1)-w(t)|| ###
        Lipz_s.append(copy.deepcopy(diff_grads/diff_weights_glob))
        ### Calculate Lipz_c=||F_i(w(t))-F_i(w_i(t))||/||w(t)-w_i(t)|| ###
        Lipz_c.append(copy.deepcopy(abs(list_loss[idx]-loss_locals[idx])/diff_weights_locals))
        ### Calculate delta= ||▽F(w(t))-▽F_i(w(t))||###
        delta.append(copy.deepcopy(Grads_variance))
    beta = copy.deepcopy(np.sqrt(sum(c*c for c in norm_grads_locals)/args.num_Chosenusers)/norm_grads_glob)
    return Lipz_s, Lipz_c, delta, beta, Grads_glob, Grads_locals, norm_grads_glob, norm_grads_locals


def Privacy_account(args, threshold_epochs, noise_list, iter):
    q_s = args.num_Chosenusers/args.num_users
    delta_s = 2*args.clipthr/args.num_items_train
    if args.dp_mechanism != 'CRD':
        noise_scale = delta_s*np.sqrt(2*q_s*threshold_epochs*np.log(1/args.delta))/args.privacy_budget
    elif args.dp_mechanism == 'CRD':
        noise_sum = 0
        for i in range(len(noise_list)):
            noise_sum += pow(1/noise_list[i],2)
        if pow(args.privacy_budget/delta_s,2)/(2*q_s*np.log(1/args.delta))>noise_sum:
            noise_scale = np.sqrt((threshold_epochs-iter)/(pow(args.privacy_budget/delta_s,2)/(2*q_s*np.log(1/args.delta))-noise_sum))
        else:
            noise_scale = noise_list[-1]
    return noise_scale


def Adjust_T(args, loss_avg_list, threshold_epochs_list, iter):
    if loss_avg_list[iter-1]-loss_avg_list[iter-2]>=0:
        threshold_epochs = copy.deepcopy(math.floor( math.ceil(args.dec_cons*threshold_epochs_list[-1])))
        threshold_epochs_list.append(threshold_epochs)
        # print('\nThreshold epochs:', threshold_epochs_list)
    else:
        threshold_epochs = threshold_epochs_list[-1]
    return threshold_epochs

def Noise_TB_decay(args, noise_list, loss_avg_list, dec_cons, iter, method_selected):
    if loss_avg_list[-1]-loss_avg_list[-2]>=0:   
        if method_selected == 'UD':
            noise_scale = copy.deepcopy(noise_list[-1]*dec_cons)
        elif method_selected == 'TBD': 
            noise_scale = copy.deepcopy(noise_list[0]/(1+dec_cons*iter))
        elif method_selected == 'ED':
            noise_scale = copy.deepcopy(noise_list[0]*np.exp(-dec_cons*iter)) 
    else:
        noise_scale = copy.deepcopy(noise_list[-1])
    q_s = args.num_Chosenusers/args.num_users
    eps_tot = 0
    for i in range(len(noise_list)-1):
        eps_tot += 1/pow(noise_list[i], 2)
    eps_tot *= 2*q_s*pow(args.num_users,2)*np.log(1/args.delta)
    return noise_scale, eps_tot

