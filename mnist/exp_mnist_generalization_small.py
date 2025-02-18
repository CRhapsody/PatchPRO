

# 1. load repaired model
# 2. from buggy dataset and its original dataset, we get the true label and error label of buggy dataset respectively
# 3. then we assign the error label and true label to the every patch as their strategies
# 4. load the trainset and testset as testing set, 
# 5. first get the max logit and runner-up logit, then get the bitmap according the strategy of patch
# 6. then we take this input and bitmap to the repaired model

import logging
import math
from selectors import EpollSelector
import sys
from argparse import Namespace
import torchvision
from ast import literal_eval
from pathlib import Path
from typing import Optional, Tuple, List
from timeit import default_timer as timer
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch.optim as optim

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils import data
from art.prop import AndProp
from art.bisecter import Bisecter
from art import exp, utils
from art.repair_moudle import Netsum
from mnist.mnist_utils import MnistNet_CNN_small, MnistNet_FNN_big, MnistNet_FNN_small, MnistProp, Mnist_patch_model_small


device = torch.device(f'cuda:0')
MNIST_DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'MNIST' / 'processed'
MNIST_NET_DIR = Path(__file__).resolve().parent.parent / 'model' /'mnist'
# MNIST_NET_DIR = Path(__file__).resolve().parent.parent / 'pgd' /'model' 
RES_DIR = Path(__file__).resolve().parent.parent / 'results' / 'mnist' / 'repair' / 'generalization' / 'small'
RES_DIR.mkdir(parents=True, exist_ok=True)

# RES_DIR_LABEL = Path(__file__).resolve().parent.parent / 'results' / 'mnist' / 'repair' / 'generalization' / 'label'
# RES_DIR_LABEL.mkdir(parents=True, exist_ok=True)

REPAIR_MODEL_DIR = Path(__file__).resolve().parent.parent / 'model' / 'mnist_patch_format_small'
# REPAIR_MODEL_DIR = Path(__file__).resolve().parent.parent / 'model' / 'mnist_label_format'
REPAIR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
# LABEL_MODEL_DIR = Path(__file__).resolve().parent.parent / 'model' / 'mnist_label_format' 
# LABEL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

COMP_DIR = Path(__file__).resolve().parent.parent / 'results' / 'mnist' / 'repair' / 'generalization' / 'compare'
COMP_DIR.mkdir(parents=True, exist_ok=True)

class MnistArgParser(exp.ExpArgParser):
    """ Parsing and storing all ACAS experiment configuration arguments. """

    def __init__(self, log_path: Optional[str], *args, **kwargs):
        super().__init__(log_path, *args, **kwargs)

        self.add_argument('--net', type=str, choices=['FNN_small', 'FNN_big', 'CNN_small'],
                          default='CNN_small', help='network architecture')

        # use repair or nor
        self.add_argument('--no_repair', type=bool, default=True, 
                        help='not repair use incremental')
        self.add_argument('--repair_number', type=int, default=50,
                          help='the number of repair datas')
        self.add_argument('--repair_batchsize', type=int, default=1,
                            help='the batchsize of repair datas')
        self.add_argument('--patch_size', type=str, default= 'big', 
                          choices=['big', 'small'], help='the size of patch network')
        self.add_argument('--label_repaired', type=bool, default=False,
                            help='if True, we use the label repaired method')
        # the combinational form of support and patch net
        # self.add_argument('--reassure_support_and_patch_combine',type=bool, default=False,
        #                 help='use REASSURE method to combine support and patch network')

        self.add_argument('--repair_radius',type=float, default=0.2, 
                          help='the radius of repairing datas or features')

        # training
        self.add_argument('--divided_repair', type=int, default=1, help='batch size for training')
        self.add_argument('--accuracy_loss', type=str, choices=['L1', 'SmoothL1', 'MSE', 'CE'], default='CE',
                          help='canonical loss function for concrete points training')
        self.add_argument('--sample_amount', type=int, default=5000,
                          help='specifically for data points sampling from spec')
        self.add_argument('--reset_params', type=literal_eval, default=False,
                          help='start with random weights or provided trained weights when available')
        self.add_argument('--train_datasize', type=int, default=10000, 
                          help='dataset size for training')
        self.add_argument('--test_datasize', type=int, default=2000,
                          help='dataset size for test')


        self.set_defaults(exp_fn='test_goal_safety', use_scheduler=True)
        return

    def setup_rest(self, args: Namespace):
        super().setup_rest(args)

        def ce_loss(outs: Tensor, labels: Tensor):
            softmax = nn.Softmax(dim=1)
            ce = nn.CrossEntropyLoss()
            return ce(softmax(outs), labels)
            # return ce(outs, labels)

        args.accuracy_loss = {
            'L1': nn.L1Loss(),
            'MSE': nn.MSELoss(),
            'CE': ce_loss
        }[args.accuracy_loss]
        return

class MnistPoints(exp.ConcIns):
    """ Storing the concrete data points for one ACAS network sampled.
        Loads to CPU/GPU automatically.
    """
    @classmethod
    def load(cls, train: bool, device, 
            net,
            repairnumber = None,
            trainnumber = None, testnumber = None, radius = 0,
            is_test_accuracy = False, 
            is_attack_testset_repaired = False, 
            is_attack_repaired = False,
            is_origin_data = False):
        #_attack_data_full
        suffix = 'train' if train else 'test'
        if train:
            fname = f'train_norm00.pt'  # note that it is using original data
            combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
            inputs, labels = combine 
            inputs = inputs[:trainnumber]
            labels = labels[:trainnumber]
        else:
            if is_test_accuracy:
                fname = f'test_norm00.pt'
                combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
                inputs, labels = combine
                inputs = inputs[:testnumber]
                labels = labels[:testnumber]
            elif is_origin_data:
                fname = f'origin_data_{net}_{radius}.pt'
                combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
                inputs, labels = combine
                inputs = inputs[:repairnumber]
                labels = labels[:repairnumber]
            
            elif is_attack_testset_repaired:
                fname = f'test_attack_data_full_{net}_{radius}.pt'
                combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
                inputs, labels = combine
                inputs = inputs[:testnumber]
                labels = labels[:testnumber]
            elif is_attack_repaired:
                fname = f'train_attack_data_full_{net}_{radius}.pt'
                combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
                inputs, labels = combine
                inputs = inputs[:repairnumber]
                labels = labels[:repairnumber]
        
        assert len(inputs) == len(labels)
        return cls(inputs, labels)

def eval_test(net, testset: MnistPoints, bitmap: Tensor = None, repair_hash_table = None,
              generalization = False) -> float:
    """ Evaluate accuracy on test set.
    :param categories: e.g., acas.AcasOut
    """
    with torch.no_grad():
        if bitmap is None:
            outs = net(testset.inputs)
        else:
            outs = net(testset.inputs, in_bitmap = bitmap)
        predicted = outs.argmax(dim=1)
        correct = (predicted == testset.labels).sum().item()
        # ratio = correct / len(testset)
        ratio = correct / len(testset.inputs)
    return ratio

def test_repaired(args: Namespace) -> float:
    # construct the original net and patch net
    # original_net = eval(f'MnistNet_{args.net}(dom = {args.dom})').to(device)
    if args.net == 'CNN_small':
        original_net = MnistNet_CNN_small(dom=args.dom)
        for_repair_net = MnistNet_CNN_small(dom=args.dom)
    elif args.net == 'FNN_big':
        original_net = MnistNet_FNN_big(dom=args.dom)
        for_repair_net = MnistNet_FNN_big(dom=args.dom)
    elif args.net == 'FNN_small':
        original_net = MnistNet_FNN_small(dom=args.dom)
        for_repair_net = MnistNet_FNN_small(dom=args.dom)
    original_net.to(device)
    original_net.load_state_dict(torch.load(Path(MNIST_NET_DIR, f'mnist_{args.net}.pth')))
    patch_lists = []
    for i in range(args.repair_number):
        patch_net = Mnist_patch_model_small(dom=args.dom,
            name = f'{args.patch_size} patch network {i}').to(device)
        patch_lists.append(patch_net)
    logging.info(f'--small patch network: {patch_net}')

    # load the repaired model
    # for_repair_net.load_state_dict(torch.load(Path(REPAIR_MODEL_DIR, f'Mnist-{args.net}-repair_number{args.repair_number}-rapair_radius{args.repair_radius}-{args.patch_size}.pt')))
    for_repair_net.to(device)
    state = torch.load(Path(REPAIR_MODEL_DIR, f'Mnist-{args.net}-repair_number{args.repair_number}-rapair_radius{args.repair_radius}-{args.patch_size}.pt'))
    repaired_net = Netsum(args.dom, for_repair_net, patch_lists, device=device,
                          generalization=True)
    repaired_net.load_state_dict(state)
    # load the dataset
    originalset = MnistPoints.load(train=False, device=device, net=args.net, repairnumber=args.repair_number, radius=args.repair_radius,is_origin_data=True)
    repairset = MnistPoints.load(train=False, device=device, net=args.net, repairnumber=args.repair_number, radius=args.repair_radius,is_attack_repaired=True)

    # from buggy dataset and its original dataset, we get the true label and error label of buggy dataset respectively
    # then we assign the error label and true label to the every patch as their strategies
    assign_table = torch.zeros(args.repair_number, 2).to(device)
    with torch.no_grad():
        buggy_outs = original_net(repairset.inputs)
        buggy_predicted = buggy_outs.argmax(dim=1)

    for i in range(args.repair_number):
        assign_table[i][0] = originalset.labels[i]
        assign_table[i][1] = buggy_predicted[i]

    repaired_net.set_repair_direction_dict(assign_table)

 





    def get_bitmap(in_lb: Tensor, in_ub: Tensor, in_bitmap: Tensor, batch_inputs: Tensor):
        '''
        in_lb: n_prop * input
        in_ub: n_prop * input
        batch_inputs: batch * input
        '''
        with torch.no_grad():
            # if len(batch_inputs.shape) == 3:
            batch_inputs_clone = batch_inputs.clone().unsqueeze_(1)
            # elif len(batch_inputs.shape) == 4:
            #     batch_inputs_clone = batch_inputs.clone()
            # distingush the photo and the property
            if len(in_lb.shape) == 2:
                batch_inputs_clone = batch_inputs_clone.expand(batch_inputs.shape[0], in_lb.shape[0], in_lb.shape[1])
                is_in = (batch_inputs_clone >= in_lb) & (batch_inputs_clone <= in_ub)
                is_in = is_in.all(dim=-1) # every input is in the region of property, batch * n_prop
            elif len(in_lb.shape) == 4:
                # if in_lb.shape[0] > 600:
                #     is_in_list = []
                #     # for i in range(batch_inputs_clone.shape[0]):
                #     # for every 500 inputs, we compare them with the property
                #     for i in range(math.ceil(batch_inputs_clone.shape[0]/500)):
                #         batch_inputs_compare_datai = batch_inputs_clone[i*500:(i+1)*500].clone().expand(in_lb.shape[0], in_lb.shape[1], in_lb.shape[2], in_lb.shape[3])
                #         is_in_datai = (batch_inputs_compare_datai >= in_lb) & (batch_inputs_compare_datai <= in_ub)
                #         is_in_datai = is_in_datai.all(dim=(-1)).all(dim=(-1)).all(dim=(-1)) # every input is in the region of property, batch * n_prop
                #         is_in_list.append(is_in_datai)

                #         # batch_inputs_compare_datai = batch_inputs_clone[i].clone().expand(in_lb.shape[0], in_lb.shape[1], in_lb.shape[2], in_lb.shape[3])
                #         # is_in_datai = (batch_inputs_compare_datai >= in_lb) & (batch_inputs_compare_datai <= in_ub)
                #         # is_in_datai = is_in_datai.all(dim=(-1)).all(dim=(-1)).all(dim=(-1)) # every input is in the region of property, batch * n_prop
                #         # is_in_list.append(is_in_datai)
                #     is_in = torch.stack(is_in_list, dim=0)
                # else:
                    batch_inputs_clone = batch_inputs_clone.expand(batch_inputs.shape[0], in_lb.shape[0], in_lb.shape[1], in_lb.shape[2], in_lb.shape[3])
                    is_in = (batch_inputs_clone >= in_lb) & (batch_inputs_clone <= in_ub)
                    is_in = is_in.all(dim=(-1)).all(dim=(-1)).all(dim=(-1)) # every input is in the region of property, batch * n_prop
            # convert to bitmap
            bitmap = torch.zeros((batch_inputs.shape[0], in_bitmap.shape[1]), device = device).to(torch.uint8)
            # is in is a batch * in_bitmap.shape[0] tensor, in_bitmap.shape[1] is the number of properties
            # the every row of is_in is the bitmap of the input which row of in_bitmap is allowed
            # for i in range(is_in.shape[0]):
            #     for j in range(is_in.shape[1]):
            #         if is_in[i][j]:
            #             bitmap[i] = in_bitmap[j]
            #             break
            #         else:
            #             continue
            bitmap_i, inbitmap_j =  is_in.nonzero(as_tuple=True)
            if bitmap_i.shape[0] != 0:
                bitmap[bitmap_i, :] = in_bitmap[inbitmap_j, :]
            else:
                pass
            return bitmap


    
    logging.info(f'--load the trainset and testset as testing set')
    # trainset = MnistPoints.load(train=True, device=device, net=args.net, trainnumber=args.train_datasize)
    testset = MnistPoints.load(train=False, device=device, net=args.net, repairnumber=args.repair_number, testnumber=args.test_datasize,is_test_accuracy=True)    # trainset = MnistPoints.load(train=True, device=device, net=args.net, trainnumber=args.train_datasize)

    # evaluate the repaired net on testset
    logging.info(f'--evaluate the original net on testset')
    ratio = eval_test(original_net, testset)
    logging.info(f'--For testset, out of {len(testset)} items, ratio {ratio}')

    logging.info(f'--evaluate the repaired net on testset and get the bitmap')
    ratio = eval_test(repaired_net, testset)
    logging.info(f'--For testset, out of {len(testset)} items, ratio {ratio}')






    # compare the autoattack defense with other models 


    from torchattacks import PGD,AutoAttack
    torch.cuda.empty_cache()


    logging.info(f'--test the defense against autoattack')
    testloader = data.DataLoader(testset, batch_size=1024, shuffle=False)
    correct2_sum = 0
    wrong_sum = 0
    correct2_sum_ori = 0
    # count = 0
    # adv_train_net.train()
    for images, labels in testloader:
        # count+=1
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            ori_images_pred = original_net(images)
            _,index = ori_images_pred.topk(2, dim=1)
            # filt the wrong prediction
            corre = index[...,0] == labels
            wrong_sum += len(labels) - corre.sum().item()
            index = index[corre]
            images = images[corre]
            labels = labels[corre]
            correct2_sum_ori += corre.sum().item()

        repaired_net.get_bitmap(sample_top2=index)
        at2 = AutoAttack(repaired_net, norm='Linf', eps=args.repair_radius, version='standard', verbose=False,steps=100,
                         bitmap=repaired_net.bitmap)
        adv_images2 = at2(images, labels)
        # adv_images2_bitmap = get_bitmap(in_lb, in_ub, repaired_net.bitmap, adv_images2)
        outs2 = repaired_net(adv_images2, bitmap = repaired_net.bitmap)
        predicted2 = outs2.argmax(dim=1)
        correct2 = (predicted2 == labels).sum().item()
        correct2_sum += correct2
        logging.info(f'correct2 {correct2}')

    logging.info(f'--For testset, out of {len(testset)} items, repaired net ratio {correct2_sum/len(testset)}')



    with open(Path(COMP_DIR, f'compare_generalization.txt'), 'a') as f:
        f.write(f'For net: {args.net}, radius: {args.repair_radius}, ' +
                f'repair_number: {args.repair_number}, ' +
                # f'adv training net ratio: {correct1_sum/len(testset)}, ' +
                f'{args.patch_size}:{correct2_sum/len(testset)}\n') 


def _run_test(args: Namespace):
    """ Run for different networks with specific configuration. """
    logging.info('===== start repair ======')
    # for nid in nids:
    logging.info(f'For pgd attack net')
    # if args.label_repaired:
    #     test_label_repaired(args)
    # else:
    test_repaired(args)
    # res.append(outs)

    # avg_res = torch.tensor(res).mean(dim=0)
    # logging.info(f'=== Avg <epochs, train_time, certified, accuracy> for pgd attack networks:')
    # logging.info(avg_res)
    return




def test_goal_test(parser: MnistArgParser):
    """ Q1: Show that we can train previously unsafe networks to safe. """
    defaults = {
        # 'start_abs_cnt': 5000,
        # 'max_abs_cnt': 
        'batch_size': 50,  # to make it faster

    }
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    logging.info(utils.fmt_args(args))
    # nids = acas.AcasNetID.goal_safety_ids(args.dom)
    if args.no_repair:
        print('why not repair?')
    else:
        _run_test(args)
    return

def test(lr:float = 0.005, net:str = 'CNN_small',repair_radius:float = 0.1, repair_number = 200, refine_top_k = 300,
         train_datasize = 200, test_datasize = 2000, 
         accuracy_loss:str = 'CE',patch_size = 'big',label_repaired = False):
    test_defaults = {
        'net': net,
        'patch_size': patch_size,
        # 'no_pts': False,
        # 'no_refine': True,
        # 'debug': False,
        # 'divided_repair': math.ceil(repair_number/100),
        'exp_fn': 'test_goal_test',
        # 'refine_top_k': refine_top_k,
        'repair_batch_size': repair_number,
        'start_abs_cnt': 500,
        'max_abs_cnt': 1000,
        'no_repair': False,
        'repair_number': repair_number,
        'train_datasize':train_datasize,
        'test_datasize': test_datasize,
        'repair_radius': repair_radius,
        'label_repaired': label_repaired,
        # 'lr': lr,
        # 'accuracy_loss': accuracy_loss,
        # 'tiny_width': repair_radius*0.0001,
        # 'min_epochs': 15,
        # 'max_epochs': 100,

        
    }
    # if label_repaired:
    #     parser = MnistArgParser(RES_DIR_LABEL, description='MNIST Correct by Construction, label repaired')
    # else:
    parser = MnistArgParser(RES_DIR, description='MNIST Correct by Construction')
    parser.set_defaults(**test_defaults)
    args, _ = parser.parse_known_args()

    exp_fn = globals()[args.exp_fn]
    start = timer()
    exp_fn(parser)

    logging.info(f'Total Cost Time: {timer() - start}s.\n\n\n')
    pass



if __name__ == '__main__':
    # for net in ['FNN_small', 'FNN_big', 'CNN_small']:
    # # for net in ['FNN_small']:
    #     for patch_size in ['small']:
    #     for label_repaired in [False]:
    #         for patch_size in ['big']:
    #             # for radius in [0.3]:
    #             for radius in [0.3]: #,0.1,0.3
    #                 # for repair_number,test_number in zip([1000],[10000]):
    #                 for repair_number in [50]:
    #                 # for repair_number in [100, 200, 500, 1000]:
    #                     test(net=net, repair_radius=radius, repair_number = repair_number, 
    #         train_datasize = 10000, test_datasize = 10000, 
    #         accuracy_loss='CE',patch_size=patch_size,label_repaired = label_repaired)
    for net in ['FNN_small', 'FNN_big', 'CNN_small']:
    # for net in ['FNN_small']:
        for patch_size in ['small']:
        # for label_repaired in [False]:
        # for label_repaired in [True]:
            # for patch_size in ['big']:
                # for radius in [0.3]:
            for radius in [0.05, 0.1, 0.3]: #,0.1,0.3
                    # for repair_number,test_number in zip([1000],[10000]):
                    

                    # for repair_number in [1000]:
                    # for repair_number in [100]:
                    for repair_number in [50, 100, 200, 500, 1000]:
                        test(net=net, repair_radius=radius, repair_number = repair_number, 
            train_datasize = 10000, test_datasize = 10000, 
            accuracy_loss='CE',patch_size=patch_size)