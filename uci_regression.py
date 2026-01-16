### --- Dependencies --- ###
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import argparse
import time
import os
import configparser
import utils.datasets
import utils.models 
import json
import utils.regression_util as utility
# import posteriors.cuqls as cuqls

import LinearSampling.Posteriors

torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description='Regression Experiment')
parser.add_argument('--dataset', required=True,type=str,help='Dataset')
parser.add_argument('--n_experiment',default=10,type=int,help='number of experiments')
parser.add_argument('--activation',default='tanh',type=str,help='Non-linear activation for MLP')
parser.add_argument('--verbose', action='store_true',help='verbose flag for results')
parser.add_argument('--extra_verbose', action='store_true',help='verbose flag for training')
parser.add_argument('--progress_bar', action='store_true',help='progress bar flag for all methods')
args = parser.parse_args()

# # Parse datasets
# r = open(args.dataset,"r")
# datasets = []
# for d in r:
#     datasets.append(d.strip())

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"\n Using {device} device")

# # Iterate through datasets:
# for dataset in datasets:

# Get hyperparameters from config file
config = configparser.ConfigParser()
config.read('utils/regression.ini')
df = utils.datasets.read_regression(args.dataset)
n_experiment = config.getint(args.dataset,'n_experiment')
input_start = config.getint(args.dataset,'input_start')
input_dim = config.getint(args.dataset,'input_dim')
target_dim = config.getint(args.dataset,'target_dim')
hidden_sizes = json.loads(config.get(args.dataset,'hidden_sizes'))
epochs = config.getint(args.dataset,'epochs')
lr = config.getfloat(args.dataset,'lr')
de_epochs = config.getint(args.dataset,'de_epochs')
de_lr = config.getfloat(args.dataset,'de_lr')
weight_decay = config.getfloat(args.dataset,'weight_decay')
nuqls_S = config.getint(args.dataset,'nuqls_S')
nuqls_epoch = config.getint(args.dataset,'nuqls_epoch')
nuqls_lr = config.getfloat(args.dataset,'nuqls_lr')

# Fixed parameters
train_ratio = 0.7
normalize = True
batch_size = 200 if (args.dataset == 'protein' or args.dataset == 'song') else 1000
mse_loss = nn.MSELoss(reduction='mean')
nll = torch.nn.GaussianNLLLoss()
S = 10

# Give dataframe summary
print("--- Loading dataset {} --- \n".format(args.dataset))
print("Number of data points = {}".format(len(df)))
print("Number of coloumns = {}".format(len(df.columns)))
print("Number of features = {}".format(input_dim-input_start))

## Num of points and dimension of data
num_points = len(df)
dimension = len(df.columns)-1

train_size = int(num_points*train_ratio)
validation_size = int(num_points*((1-train_ratio)/2))
test_size = int(num_points - train_size - validation_size)
dataset_numpy = df.values

# Normalize the dataset
if normalize:
    mx = dataset_numpy[:,input_start:input_dim].mean(0)
    my = dataset_numpy[:,target_dim].mean(0)
    sx = dataset_numpy[:,input_start:input_dim].std(0)
    sx = np.where(sx==0,1,sx)
    sy = dataset_numpy[:,target_dim].std(0)
    sy = np.where(sy==0,1,sy)

# Setup metrics
methods = ['MAP','DNN-GLM','LL-GLM']
train_methods = ['MAP','DNN-GLM','LL-GLM']
test_res = {}
train_res = {}
for m in methods:
    if m in train_methods:
        train_res[m] = {'loss': [],
                        'time': []}
    test_res[m] = {'rmse': [],
                'nll': [],
                'ece': [],
                'time': []}

# Iterate through number of experiments
for ei in tqdm(range(n_experiment)):
    print("\n--- experiment {} ---".format(ei))
    np.random.shuffle(dataset_numpy) # Randomness
    training_set, validation_set, test_set = dataset_numpy[:train_size,:], dataset_numpy[train_size:train_size+validation_size], dataset_numpy[train_size+validation_size:,:]

    train_dataset = utils.datasets.RegressionDataset(training_set, 
                                    input_start=input_start, input_dim=input_dim, target_dim=target_dim,
                                    mX=mx, sX=sx, my=my, sy=sy)
    validation_dataset = utils.datasets.RegressionDataset(validation_set, 
                                    input_start=input_start, input_dim=input_dim, target_dim=target_dim,
                                    mX=mx, sX=sx, my=my, sy=sy)
    test_dataset = utils.datasets.RegressionDataset(test_set, 
                                    input_start=input_start, input_dim=input_dim, target_dim=target_dim,
                                    mX=mx, sX=sx, my=my, sy=sy)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)
    _, test_y = next(iter(test_loader))

    calibration_test_loader = DataLoader(test_dataset,1)
    calibration_test_loader_val = DataLoader(validation_dataset,len(validation_dataset))
    _, val_y = next(iter(calibration_test_loader_val))

    for m in methods:
        print(f'METHOD:: {m}')
        t1 = time.time()
        if m == 'MAP':
            map_net = utils.models.mlp(input_size=input_dim-input_start, hidden_sizes=hidden_sizes, output_size=1, activation=args.activation, flatten=False, bias=True).to(device=device, dtype=torch.float64)
            map_net.apply(utility.weights_init)
            map_p = sum(p.numel() for p in map_net.parameters() if p.requires_grad)
            print(f'parameters of network = {map_p}')
            
            if args.dataset=='kin8nm' or args.dataset=='wine' or args.dataset=='naval' or args.dataset=='protein' or args.dataset=='song':
                optimizer = torch.optim.SGD(map_net.parameters(),lr=lr, weight_decay=weight_decay, momentum=0.9)
                scheduler = None
            else:
                optimizer = torch.optim.Adam(map_net.parameters(), lr=lr, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs*10, power=0.5)

            # Run the training loop
            for epoch in tqdm(range(epochs)):
                map_train_loss  = utility.train(train_loader, map_net, optimizer=optimizer, loss_function=mse_loss, scheduler=scheduler)
                map_test_loss = utility.test(test_loader, map_net, my=0, sy=1, loss_function=mse_loss)
                if args.extra_verbose and epoch % 10 == 0:
                    print("Epoch {} of {}".format(epoch,epochs))
                    print("Training loss = {:.4f}".format(map_train_loss))
                    print("Test loss = {:.4f}".format(map_test_loss))
                    print("\n -------------------------------------")
            train_res[m]['loss'].append(map_train_loss)

            print(f'MAP: TEST RMSE = {np.sqrt(map_test_loss):.4}')

            map_pred = []
            for x,_ in test_loader:
                x = x.to(device)
                map_pred.append(map_net(x))
            map_test_pred = torch.cat(map_pred)

            map_pred = []
            val_loader = DataLoader(validation_dataset, batch_size=test_size, shuffle=False)
            for x,_ in val_loader:
                x = x.to(device)
                map_pred.append(map_net(x))
            map_val_pred = torch.cat(map_pred)

        elif m == 'DNN-GLM':
            dnn_glm = LinearSampling.Posteriors.Posterior(network=map_net, 
                                                        glm_type='DNN',
                                                        task='regression', 
                                                        precision = 'double')

            res = dnn_glm.train(train=train_dataset, 
                                bs=batch_size,
                                gamma = 0.01,
                                S = nuqls_S,
                                epochs=nuqls_epoch,
                                lr=nuqls_lr,
                                mu=0.9,
                                verbose=args.verbose,
                                extra_verbose=args.extra_verbose,
                                plot_loss_dir=('metrics/uci_regression/' if args.verbose else None))
            
            train_res[m]['loss'].append(res['mean_sq_loss'])
            dnn_glm.HyperparameterTuning(validation=validation_dataset,
                                        bs=batch_size,
                                        left=0.01,
                                        right=10000,
                                        its=100,
                                        verbose=args.extra_verbose
            )

            mean_pred, var_pred = dnn_glm.UncertaintyPrediction(test=test_dataset,
                                                            bs=batch_size)

        elif m == 'LL-GLM':
            ll_glm = LinearSampling.Posteriors.Posterior(network=map_net, 
                                                        glm_type='LL',
                                                        task='regression', 
                                                        precision = 'double')

            res = ll_glm.train(train=train_dataset, 
                                bs=batch_size,
                                gamma = 0.01,
                                S = nuqls_S,
                                epochs=nuqls_epoch,
                                lr=nuqls_lr,
                                mu=0.9,
                                verbose=args.verbose,
                                extra_verbose=args.extra_verbose,
                                plot_loss_dir=('metrics/uci_regression/' if args.extra_verbose else None))
            train_res[m]['loss'].append(res['mean_sq_loss'])
            ll_glm.HyperparameterTuning(validation=validation_dataset,
                                        bs=batch_size,
                                        left=0.01,
                                        right=10000,
                                        its=100,
                                        verbose=args.extra_verbose
            )

            mean_pred, var_pred = ll_glm.UncertaintyPrediction(test=test_dataset,
                                                            bs=batch_size)

        print(f"\n--- Method {m} ---")
        t2 = time.time()
        if m in train_res:
            print("\nTrain Results:")
            print(f"Train Loss: {train_res[m]['loss'][ei]:.3f}")
            # if m == 'MAP':
            train_res[m]['time'].append(t2-t1)
            t = train_res[m]['time'][ei]
            print(f'Time(s): {t:.3f}')

        if m != 'MAP':
            test_res[m]['time'].append(t2-t1)

            test_res[m]['rmse'].append(torch.sqrt(mse_loss(mean_pred.detach().cpu().reshape(-1,1),test_y.reshape(-1,1))).detach().cpu().item())

            test_res[m]['nll'].append(nll(mean_pred.detach().cpu().reshape(-1,1),test_y.reshape(-1,1),var_pred.detach().cpu().reshape(-1,1)).detach().cpu().item())

            observed_conf, predicted_conf = utility.calibration_curve_r(test_y,mean_pred,var_pred,11)
            test_res[m]['ece'].append(torch.mean(torch.square(observed_conf - predicted_conf)).detach().cpu().item())

            print("\nTest Prediction:")
            t = test_res[m]['time'][ei]
            print(f'Time(s): {t:.3f}')
            print(f"RMSE.: {test_res[m]['rmse'][ei]:.3f}; NLL: {test_res[m]['nll'][ei]:.3f}; ECE: {test_res[m]['ece'][ei]:.1%}")
        print('\n')

## Record results
res_dir = "./results/uci_regression"

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

results = open(f"{res_dir}/{args.dataset}.txt",'w')
results.write("Training, Val, Test points = {}, {}, {}\n".format(train_size, validation_size, test_size))
results.write(f"Number of hidden units, parameters = {hidden_sizes}, {map_p}\n")
results.write(f'n_experiment = {n_experiment}\n')

results.write("\n--- MAP Training Details --- \n")
results.write("training: epochs, de_epochs, lr, weight decay, batch size = {}, {}, {}, {}, {}\n".format(
    epochs, de_epochs, lr, weight_decay, batch_size
))

results.write("\n --- NUQLS Details --- \n")
results.write(f"epochs: {nuqls_epoch}; S: {nuqls_S}; lr: {nuqls_lr}\n")

for m in methods:
    if n_experiment > 1:
        results.write(f"\n--- Method {m} ---\n")
        if m in train_res:
            results.write("\n - Train Results: - \n")
            for k in train_res[m].keys():
                results.write(f"{k}: {np.mean(train_res[m][k]):.3f} +- {np.std(train_res[m][k]):.3f} \n")
        if m != 'MAP':
            results.write("\n - Test Prediction: - \n")
            for k in test_res[m].keys():
                # if k == 'time':
                #     t = time.strftime("%H:%M:%S", time.gmtime(np.mean(test_res[m][k])))
                #     results.write(f"{k}: {t}\n")
                # else:
                results.write(f"{k}: {np.mean(test_res[m][k]):.3f} +- {np.std(test_res[m][k]):.3f} \n")
    else:
        results.write(f"\n--- Method {m} ---\n")
        if m in train_res:
            results.write("\n - Train Results: - \n")
            for k in train_res[m].keys():
                results.write(f"{k}: {train_res[m][k][0]:.3f}\n")
        if m != 'MAP':
            results.write("\n - Test Prediction: - \n")
            for k in test_res[m].keys():
                # if k == 'time':
                #     t = time.strftime("%H:%M:%S", time.gmtime(test_res[m][k][0]))
                #     results.write(f"{k}: {t}\n")
                # else:
                results.write(f"{k}: {test_res[m][k][0]:.3f}\n")
results.close()

print('results created')

