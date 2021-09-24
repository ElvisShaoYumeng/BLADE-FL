#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import time
import matplotlib
import sys

matplotlib.use('Agg')
import os
import copy
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from scipy import optimize
import random
import cmath

from Calculate import get_2_norm, get_2_diff, calculate_grads, avg_grads
from sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from options import args_parser
from Update import LocalUpdate
from FedNets import MLP1, CNNMnist, CNN_test
from averaging import average_weights
from Privacy import Privacy_account, Adjust_T
from Noise_add import noise_add, users_sampling, clipping

if __name__ == '__main__':
    # return the available GPU
    av_GPU = torch.cuda.is_available()
    if av_GPU == False:
        exit('No available GPU')
    # parse args
    args = args_parser()
    # define paths
    path_project = os.path.abspath('..')

    summary = SummaryWriter('local')
    ### computation allocation ###
    args.dp_mechanism = 'CRD'  ### CRD or non-CRD###
    args.dec_cons = 0.8  ## discount constant
    # args.privacy_budget = 100
    args.delta = 0.1

    args.local_frequence = 1
    args.bc_difficulty = 500

    # args.cauchy=0.1
    # args.T=3

    args.gpu = -1  # -1 (CPU only) or GPU = 0
    args.lr = 0.01  # 0.001 for cifar dataset
    args.model = 'mlp'  # 'mlp' or 'cnn'
    args.dataset = 'mnist'  # 'mnist'
    args.num_users = 50  ### numb of users ###
    # args.num_Chosenusers = 30

    args.num_items_train = 256  # numb of local data size #
    args.num_items_test = 128
    args.local_bs = 32  ### Local Batch size (1200 = full dataset ###
    ### size of a user for mnist, 2000 for cifar) ###
    args.total_time = 100
    args.bl_antifrequence = int(args.bc_difficulty / args.num_users)
    args.T_max = int((args.local_frequence * args.total_time) // (args.local_frequence * args.bl_antifrequence + 1))
    # T_max = B *delta_t / (B * C + 1)
    args.set_epoch = range(1, args.T_max + 1)
    # args.set_epochs = range(5,105,5)
    args.set_num_Chosenusers = [args.num_users]
    args.set_lazy = int(args.num_users * 0)  ### no lazy
    args.num_experiments = 20
    args.clipthr = 10

    args.iid = False
    args.degree_noniid = 1

    # load dataset and split users
    dict_users = {}
    dict_users_test = {}
    dataset_train = []
    dataset_test = []
    dict_users_train = {}
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))

    # sample users
    if args.iid:
        dict_users = mnist_iid(args, dataset_train, args.num_users, args.num_items_train)
        # dict_users_test = mnist_iid(dataset_test, args.num_users, args.num_items_test)
        dict_sever = mnist_iid(args, dataset_test, args.num_users, args.num_items_test)
    else:
        dict_users = mnist_noniid(args, dataset_train, args.num_users, args.num_items_train)
        dict_sever = mnist_noniid(args, dataset_test, args.num_users, args.num_items_test)

    img_size = dataset_train[0][0].shape

    final_train_loss = [[0 for i in range(len(args.set_epoch))] for j in range(len(args.set_num_Chosenusers))]
    final_train_accuracy = [[0 for i in range(len(args.set_epoch))] for j in range(len(args.set_num_Chosenusers))]
    final_test_loss = [[0 for i in range(len(args.set_epoch))] for j in range(len(args.set_num_Chosenusers))]
    final_test_accuracy = [[0 for i in range(len(args.set_epoch))] for j in range(len(args.set_num_Chosenusers))]

    final_Lipschitz_chixi = [[0 for i in range(len(args.set_epoch))] for j in range(len(args.set_num_Chosenusers))]
    final_smooth_L = [[0 for i in range(len(args.set_epoch))] for j in range(len(args.set_num_Chosenusers))]
    final_gap_delta = [[0 for i in range(len(args.set_epoch))] for j in range(len(args.set_num_Chosenusers))]
    final_lazy_theta = [[0 for i in range(len(args.set_epoch))] for j in range(len(args.set_num_Chosenusers))]

    for s in range(len(args.set_num_Chosenusers)):
        for j in range(len(args.set_epoch)):
            args.num_Chosenusers = copy.deepcopy(args.set_num_Chosenusers[s])
            args.epochs = copy.deepcopy(args.set_epoch[j])  # numb of global iters
            args.tau = args.local_frequence * (args.total_time - args.bl_antifrequence * args.epochs)
            args.tau_avg = args.tau // args.epochs
            args.local_ep = int(args.tau_avg)  # numb of local iters
            print("dataset:", args.dataset, " num_users:", args.num_users, " num_chosen_users:", args.num_Chosenusers,
                  " epochs:", args.epochs, \
                  "local_ep:", args.local_ep, "local train size", args.num_items_train, "batch size:", args.local_bs)
            loss_test, loss_train = [], []
            acc_test, acc_train = [], []
            smooth_L, Lipschitz_chixi, gap_delta, lazy_theta = [], [], [], []
            for m in range(args.num_experiments):
                # build model
                net_glob = None
                # net_local = None
                if args.model == 'cnn' and args.dataset == 'mnist':
                    if args.gpu != -1:
                        torch.cuda.set_device(args.gpu)
                        # net_glob = CNNMnist(args=args).cuda()
                        net_glob = CNN_test(args=args).cuda()
                    # net_local = CNN_test(args=args).cuda()
                    else:
                        net_glob = CNNMnist(args=args)
                    # net_local = CNNMnist(args=args)
                elif args.model == 'mlp':
                    len_in = 1
                    for x in img_size:
                        len_in *= x
                    if args.gpu != -1:
                        torch.cuda.set_device(args.gpu)
                        net_glob = MLP1(dim_in=len_in, dim_hidden=32, dim_out=args.num_classes).cuda()
                    #  net_local = MLP1(dim_in=len_in, dim_hidden=256, dim_out=args.num_classes).cuda()
                    else:
                        net_glob = MLP1(dim_in=len_in, dim_hidden=32, dim_out=args.num_classes)
                        # net_local = MLP1(dim_in=len_in, dim_hidden=256, dim_out=args.num_classes)
                else:
                    exit('Error: unrecognized model')
                print("Nerual Net:", net_glob)

                net_glob.train()  # Train() does not change the weight values
                # copy weights
                w_glob = net_glob.state_dict()

                # net_local.train()  #Train() does not change the weight values
                # copy weights
                # w_local = net_local.state_dict()

                w_size = 0
                w_size_all = 0
                for k in w_glob.keys():
                    size = w_glob[k].size()
                    if (len(size) == 1):
                        nelements = size[0]
                    else:
                        nelements = size[0] * size[1]
                    w_size += nelements * 4
                    w_size_all += nelements
                    # print("Size ", k, ": ",nelements*4)
                print("Weight Size:", w_size, " bytes")
                print("Weight & Grad Size:", w_size * 2, " bytes")
                print("Each user Training size:", 784 * 8 / 8 * args.local_bs, " bytes")
                print("Total Training size:", 784 * 8 / 8 * 60000, " bytes")
                # training
                threshold_epochs = copy.deepcopy(args.epochs)
                threshold_epochs_list, noise_list = [], []
                loss_avg_list, acc_avg_list, list_loss, loss_avg = [], [], [], []
                eps_tot_list, eps_tot = [], 0
                ###  FedAvg Aglorithm  ###
                ### Compute noise scale ###
                noise_scale = 0.01
                for iter in range(args.epochs):
                    print('\n', '*' * 20, f'Epoch: {iter}', '*' * 20)
                    if args.num_Chosenusers < args.num_users:
                        chosenUsers = random.sample(range(1, args.num_users), args.num_Chosenusers)
                        chosenUsers.sort()
                    else:
                        chosenUsers = range(args.num_users)
                    print("\nChosen users:", chosenUsers)
                    w_locals, w_locals_1ep, loss_locals, acc_locals, w_locals_2ep = [], [], [], [], []
                    w_difference, difference_loss = [], []
                    w_lazy_diff_list = []
                    w_glob_pre = w_glob
                    # local_chixi, local_L, local_delta = [], [], []
                    ### local train  ###
                    for idx in range(len(chosenUsers)):
                        # m = max(int(args.frac * args.num_users), 1)
                        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                        if idx < (len(chosenUsers) - args.set_lazy):
                            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[chosenUsers[idx]],
                                                tb=summary)
                            w_1st_ep, w_2st_ep, w, loss, acc = local.update_weights(net=copy.deepcopy(net_glob))
                            ### get updated local weights ###
                            w_locals.append(copy.deepcopy(w))
                            ### record 1st-ep and 2nd-ep local weights ###
                            w_locals_1ep.append(copy.deepcopy(w_1st_ep))
                            w_locals_2ep.append(copy.deepcopy(w_2st_ep))
                            ### get local loss ###
                            loss_locals.append(copy.deepcopy(loss))
                            # print("User:", chosenUsers[idx], " Acc:", acc, " Loss:", loss)
                            acc_locals.append(copy.deepcopy(acc))
                            ### for lazy user ###

                        else:
                            ###  copy  ###
                            k = random.randint(0, (idx - 1))
                            lazy_locals = copy.deepcopy(w_locals[k])
                            lazy_locals_1ep = copy.deepcopy(w_locals_1ep[k])
                            lazy_locals_2ep = copy.deepcopy(w_locals_2ep[k])
                            w_locals.append(copy.deepcopy(lazy_locals))
                            w_locals_1ep.append(copy.deepcopy(lazy_locals_1ep))
                            w_locals_2ep.append(copy.deepcopy(lazy_locals_2ep))
                            lazy_loss = copy.deepcopy(loss_locals[k])
                            lazy_acc = copy.deepcopy(acc_locals[k])
                            loss_locals.append(copy.deepcopy(lazy_loss))
                            acc_locals.append(copy.deepcopy(lazy_acc))

                            ### perturb 'w_local' ###
                            w_locals[len(chosenUsers) - args.set_lazy:len(chosenUsers)] = noise_add(args, noise_scale, \
                                 w_locals[len(chosenUsers) - args.set_lazy:len(chosenUsers)])  # noise variance is 0.01#
                            w_locals_1ep[len(chosenUsers) - args.set_lazy:len(chosenUsers)] = noise_add(args,noise_scale, \
                                 w_locals_1ep[len(chosenUsers) - args.set_lazy:len(chosenUsers)])
                            w_locals_2ep[len(chosenUsers) - args.set_lazy:len(chosenUsers)] = noise_add(args,noise_scale, \
                                 w_locals_2ep[len(chosenUsers) - args.set_lazy:len(chosenUsers)])
                            ### theta para estimate ###
                            if iter == (args.epochs - 1):
                                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[chosenUsers[idx]],
                                                    tb=summary)
                                w_1st_ep, w_2st_ep, w, loss, acc = local.update_weights(net=copy.deepcopy(net_glob))
                                w_lazy_diff_list.append(get_2_norm(w, w_locals[k]))

                    ''''### Clipping ###
                    for idx in range(len(chosenUsers)):
                        w_locals[idx] = copy.deepcopy(clipping(args, w_locals[idx]))
                        # print(get_2_norm(w_locals[idx], w_glob))'''

                    ### diffrent_loss ###
                    '''
                    for idx in range(len(chosenUsers)):
                        #w_locals[idx] = copy.deepcopy(clipping(args, w_locals[idx]))
                        # print(get_2_norm(w_locals[idx], w_glob))
                        difference_loss.append(get_2_norm(w_locals[idx], w_glob))
                    '''

                    ### update global weights ###
                    # w_locals = users_sampling(args, w_locals, chosenUsers)
                    w_glob = average_weights(w_locals)

                    ###  update 1ep_weights  ###
                    w_1ep = average_weights(w_locals_1ep)

                    # copy weight to net_glob
                    net_glob.load_state_dict(w_glob)
                    # global test
                    list_acc, list_loss = [], []
                    grad_list, grad_local_list = [], []

                    chixi_list, delta_list, = [], []

                    w_avg, w_last_avg = [], [],
                    grad_local = []
                    grad_glob = []
                    para_loss = []

                    net_glob.eval()
                    for c in range(args.num_users):
                        net_local = LocalUpdate(args=args, dataset=dataset_test, idxs=dict_sever[c], tb=summary)
                        acc, loss = net_local.test(net=net_glob)
                        # acc, loss = net_local.test_gen(net=net_glob, idxs=dict_users[c], dataset=dataset_test)
                        list_acc.append(copy.deepcopy(acc))
                        list_loss.append(copy.deepcopy(loss))
                    for c in range(args.num_users):
                        net_local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[c], tb=summary)
                        acc, loss = net_local.test(net=net_glob)
                        # acc, loss = net_local.test_gen(net=net_glob, idxs=dict_users[c], dataset=dataset_test)
                        para_loss.append(copy.deepcopy(loss))
                        ###  for lazy user  ###
                        # else:
                    # print("\nEpoch: {}, Global test loss {}, Global test acc: {:.2f}%".\
                    #      format(iter, sum(list_loss) / len(list_loss),100. * sum(list_acc) / len(list_acc)))
                    # print loss

                    grad_locals_1ep, grad_locals_glob, grad_list, delta_list = [], [], [], []
                    for idx in range(len(chosenUsers)):
                        ###-calculate gradients-###
                        grad_locals_glob.append(calculate_grads(args, w_glob_pre, w_locals_1ep[idx]))
                        grad_locals_1ep.append(calculate_grads(args, w_locals_1ep[idx], w_locals_2ep[idx]))

                        grad_list.append(get_2_norm(grad_locals_glob[idx], grad_locals_1ep[idx]) / \
                                         get_2_norm(w_glob_pre, w_locals_1ep[idx]))

                    grad_glob = avg_grads(grad_locals_glob)
                    for idx in range(len(chosenUsers)):
                        delta_list.append(get_2_norm(grad_locals_glob[idx], grad_glob))
                    ###-lipschitz continuous-###
                    '''rho = sum(rho_locals) / len(rho_locals)
                    rho_list.append(rho)
                    ###-lipschitz smooth-###
                    beta = sum(beta_locals) / len(beta_locals)
                    beta_list.append(beta)
                    ###-gradient divergence-###
                    delta = sum(delta_locals) / len(delta_locals)
                    delta_list.append(delta)
                    omega_list.append(1 / get_2_norm(w_glob, opt_w))
                    phi_list.append(min(omega_list) * (1 - 0.5 * beta * args.lr))
                    norm_w_list.append(get_1_norm(w_glob))'''
                    ###  grad_local  ###
                    '''for idx in range(len(chosenUsers)):
                        grad_local = get_2_diff(w_locals_1ep[idx], w_locals[idx])
                        #grad_local_list.append(grad_local)
                        #grad_local.append(get_2_norm(w_locals_1ep[idx], w_locals[idx]))

                    ###  grad_global  ###
                    grad_glob = average_weights(grad_local)
                    grad_glob_list = []
                    for idx in range(len(chosenUsers)):
                        grad_glob_list.append(grad_glob)
                        '''
                    '''
                    w_avg.append(get_1_sum(w_locals[idx]))
                    w_last_avg.append(get_1_sum(w_locals_1ep[idx]))
                    for idx in range(len(w_avg)):
                        grad_glob.append(w_last_avg[idx] - w_avg[idx])
                        '''
                    ###  different_w  ###

                    for idx in range(len(chosenUsers)):
                        # diff_w = w_locals_1ep[idx] - w_glob
                        w_difference.append(get_2_norm(w_locals[chosenUsers[idx]], w_glob))

                    ###  loss_difference  ###
                    for idx in range(len(chosenUsers)):
                        diff_loss = loss_locals[idx] - para_loss[idx]
                        difference_loss.append(np.linalg.norm(diff_loss))

                    ###  update lazy diff weights  ###
                    if iter == (args.epochs - 1) and args.set_lazy != 0:
                        w_lazy_diff = sum(w_lazy_diff_list) / len(w_lazy_diff_list)
                        # noise_scale = w_lazy_diff * 0.4

                    ###  grad_difference  ###
                    '''diff_grad = []
                    for idx in range(len(chosenUsers)):
                        diff_grad[idx] = np.array(grad_local[idx]) - np.array(grad_glob)
                        grad_list.append(np.linalg.norm(diff_grad) /( args.lr * w_difference[idx]))

                    ### delta  ###
                    for idx in range(len(chosenUsers)):
                        grad_delta=grad_local[idx] - grad_glob
                        delta_list.append(np.linalg.norm(grad_delta))'''

                    ###  chixi_list  ###
                    for idx in range(len(chosenUsers)):
                        chixi_list.append(difference_loss[idx] / w_difference[idx])

                    chixi_avg = sum(chixi_list) / len(chixi_list)
                    L_avg = sum(grad_list) / len(grad_list)
                    delta_avg = sum(delta_list) / len(delta_list)

                    loss_avg = sum(loss_locals) / len(loss_locals)
                    acc_avg = sum(acc_locals) / len(acc_locals)
                    loss_avg_list.append(loss_avg)
                    acc_avg_list.append(acc_avg)

                    '''chixi_avg = 0
                    if iter >= 1:
                        chixi_avg = sum(local_chixi) / len(local_chixi)
                        chixi_avg_list.append(chixi_avg)'''

                    print("\nTrain loss: {}, Train acc: {}". \
                          format(loss_avg_list[-1], acc_avg_list[-1]))
                    print("\nTest loss: {}, Test acc: {}". \
                          format(sum(list_loss) / len(list_loss), sum(list_acc) / len(list_acc)))
                    '''if iter >= 1:
                        print("\nTrain chixi: {}". \
                            format(chixi_avg_list[-1]))'''

                    # noise_list.append(noise_scale)
                    # print('\nNoise Scale:', noise_list)

                    if args.dp_mechanism == 'CRD' and iter >= 1:
                        threshold_epochs_list.append(threshold_epochs)
                        threshold_epochs = Adjust_T(args, loss_avg_list, threshold_epochs_list, iter)
                        # noise_scale = copy.deepcopy(Privacy_account(args, threshold_epochs, noise_list, iter))
                        print('\nThreshold epochs:', threshold_epochs_list)
                        '''
                    if iter >= args.epochs:

                        dec_cons = 0.8
                        #noise_scale,  eps_tot= Noise_TB_decay(args, noise_list, loss_avg_list, dec_cons, iter, 'UD')

                        q_s = args.num_Chosenusers/args.num_users
                        eps_tot = 0
                        for i in range(len(noise_list)):
                            eps_tot = copy.deepcopy(eps_tot + 1/pow(noise_list[i], 2))
                        eps_tot = copy.deepcopy(np.sqrt(eps_tot*2*q_s*np.log(1/args.delta)/pow(args.num_users,2)))

                        eps_tot_list.append(eps_tot)
                        print('\nEpslion:', eps_tot_list)
                        print('\nNoise Scale:', noise_list)
                        if eps_tot >= args.privacy_budget:
                            threshold_epochs = 0 


                    if iter >= threshold_epochs:
                        break
                        '''

                Lipschitz_chixi.append(chixi_avg)
                smooth_L.append(L_avg)
                gap_delta.append(delta_avg)
                if args.set_lazy != 0:
                    lazy_theta.append(w_lazy_diff)

                loss_train.append(loss_avg)
                acc_train.append(acc_avg)
                loss_test.append(sum(list_loss) / len(list_loss))
                acc_test.append(sum(list_acc) / len(list_acc))

            # plot loss curve
            final_train_loss[s][j] = copy.deepcopy(sum(loss_train) / len(loss_train))
            final_train_accuracy[s][j] = copy.deepcopy(sum(acc_train) / len(acc_train))
            final_test_loss[s][j] = copy.deepcopy(sum(loss_test) / len(loss_test))
            final_test_accuracy[s][j] = copy.deepcopy(sum(acc_test) / len(acc_test))

            final_Lipschitz_chixi[s][j] = copy.deepcopy(sum(Lipschitz_chixi) / len(Lipschitz_chixi))
            final_smooth_L[s][j] = copy.deepcopy(sum(smooth_L) / len(smooth_L))
            final_gap_delta[s][j] = copy.deepcopy(sum(gap_delta) / len(gap_delta))
            if args.set_lazy != 0:
                final_lazy_theta[s][j] = copy.deepcopy(sum(lazy_theta) / len(lazy_theta))

        print('\nFinal train loss:', final_train_loss)
        print('\nFinal train accuracy:', final_train_accuracy)
        print('\nFinal test loss:', final_test_loss)
        print('\nFinal test accuracy:', final_test_accuracy)

        print('\nFinal Lipschitz chixi:', final_Lipschitz_chixi)
        print('\nFinal smooth L:', final_smooth_L)
        print('\nFinal delta:', final_gap_delta)
        if args.set_lazy != 0:
            print('\nFinal theta:', final_lazy_theta)

    timeslot = int(time.time())
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeslot))
    with open('./SimulationData/new_fed_{}UEs_{}_{}_{}_{}_{}_C{}_lr{}_iid{}_{}.csv'. \
                      format(args.num_users, args.dataset, \
                             args.model, args.total_time, args.local_frequence, args.bl_antifrequence, args.epochs,
                             args.lr, args.iid, timeslot), 'w', encoding='utf-8') as f:
        f.write('Test_loss:')
        f.write(str(final_train_loss))
        f.write('\nTest_accuracy:')
        f.write(str(final_train_accuracy))
        f.write('\nTrain_loss:')
        f.write(str(final_test_loss))
        f.write('\nTrain_accuracy:')
        f.write(str(final_test_accuracy))
        f.write('\nLipschitz chixi:')
        f.write(str(final_Lipschitz_chixi))
        f.write('\nsmooth L:')
        f.write(str(final_smooth_L))
        f.write('\ndelta:')
        f.write(str(final_gap_delta))
        if args.set_lazy != 0:
            f.write('\ntheta:')
            f.write(str(final_lazy_theta))
        f.write('\nsigma:')
        f.write(str(noise_scale))