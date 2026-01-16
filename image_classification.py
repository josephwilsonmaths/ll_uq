import torch
import tqdm
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

import LinearSampling
import os
import posteriors.util as pu
import posteriors.swag as swag
import posteriors.mc as mc
import utils.metrics as metrics
from posteriors.de import DeepEnsemble
import time
import datetime
import argparse
import warnings
import utils.training
import utils.classification_dataset
import utils.networks
import utils.hyperparameters
import utils.optimizers
from laplace import Laplace

warnings.filterwarnings('ignore')

precision = 'double'

if precision == 'half':
    dtype = torch.float16
elif precision == 'single':
    dtype = torch.float32
elif precision == 'double':
    dtype = torch.float64
else:
    print('invalid precision value. Valid options are [half, single, double].')

torch.set_default_dtype(dtype)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"\n Using {device} device")
if device == 'cuda':
    print(f"CUDA version: {torch.version.cuda}")

parser = argparse.ArgumentParser(description='Classification Experiment')
parser.add_argument('--dataset', default='mnist', type=str, help='dataset')
parser.add_argument('--model', default='lenet', type=str, help='model: lenet, resnet9, resnet50')
parser.add_argument('--methods', default='./utils/methods.txt', type=str, help='pathfile to method text file')
parser.add_argument('--subsample', action='store_true', help='Use less datapoints for train and test.')
parser.add_argument('--save_var', action='store_true', help='save variances if on (memory consumption)')
parser.add_argument('--pre_load', action='store_true', help='use pre-trained weights (must have saved state_dict() for correct model + dataset)')
parser.add_argument('--glm_pre_load', action='store_true', help='use pre-trained weights (must have saved state_dict() for correct model + dataset)')
parser.add_argument('--verbose', action='store_true',help='verbose flag for all methods')
parser.add_argument('--extra_verbose', action='store_true',help='extra verbose flag for some methods')
parser.add_argument('--progress', action='store_false')
args = parser.parse_args()

# Get dataset
dataset = utils.classification_dataset.load_dataset(name=args.dataset, subsample=args.subsample)

#--- Get hyperparameters from config file
config = utils.hyperparameters.get_config('utils/classification.ini', args.model, args.dataset)
# ---

print('Creating dataloader')
train_dataloader = dataset.trainloader(batch_size=config['bs'])
test_dataloader = dataset.testloader(batch_size=config['bs'])
ood_test_dataloader = dataset.oodtestloader(batch_size=config['bs'])
loss_fn = nn.CrossEntropyLoss()

# Setup metrics
methods = ['MAP']
with open(args.methods,'r') as method_file:
    for method_line in method_file:
        methods.append(method_line.strip())
print(methods)

train_methods = ['MAP','DNN-GLM','LL-GLM','DE']
test_res = {}
train_res = {}
for m in methods:
    if m in train_methods:
        train_res[m] = {'nll': [],
                        'acc': []}
    test_res[m] = {'nll': [],
                  'acc': [],
                  'ece': [],
                  'oodauc': [],
                  'aucroc': [],
                  'varroc': [],
                  'varroc_id': [],
                  'varroc_ood': [],
                  'varroc_rot': [],
                  'time': [],
                  'mem': []}
                  
prob_var_dict = {}
prediction_dict = {}
tolerance_acc_dict = {}

# Setup directories
res_dir = f"./results/image_classification/{args.dataset}_{args.model}/"
ct = datetime.datetime.now()
time_str = f"{ct.day}_{ct.month}_{ct.hour}_{ct.minute}"
if not args.subsample:
    res_dir = res_dir + f"_{time_str}/"
else:
    res_dir = res_dir + f"_s_{time_str}/"

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

if args.verbose:
    print(f'Using {args.dataset} dataset, num train points = {len(dataset.training_data)}')

for ei in tqdm.trange(config['n_experiment']):
    print("\n--- experiment {} ---".format(ei))
    
    id_mean, id_var, ood_mean, ood_var = None, None, None, None

    if device == 'cuda':
        print('yes')
        torch.cuda.reset_peak_memory_stats()

    for m in methods:
        print(f'method: {m}, max memory usage: {1e-9*torch.cuda.max_memory_allocated():.2f}GB, current memory usage: {1e-9*torch.cuda.memory_allocated():.2f}GB')
        if id_mean is not None:
            del(id_mean); del(id_var); del(ood_mean); del(ood_var)
        t1 = time.time()
        if m == 'MAP':
            if args.dataset == 'imagenet':
                map_net = resnet50(ResNet50_Weights).to(device)
            else:
                map_net = utils.networks.get_model(args.model, dataset.n_output, dataset.n_channels).to(device)
            num_weights = sum(p.numel() for p in map_net.parameters() if p.requires_grad)
            if args.verbose:
                print(f"Network parameter count: {num_weights}")

            # Multiple GPUs
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                map_net = nn.DataParallel(map_net)

            if args.pre_load:
                print('Using pre-loaded weights!')
                if args.dataset == 'imagenet':
                    map_net.eval()
                    train_res[m]['nll'].append(0.0)
                    train_res[m]['acc'].append(1.0)
                else:
                    model_dict = torch.load(f'saved_models/{args.model}_trained_{args.dataset}.pt', weights_only=True, map_location=device)
                    if 'params' in model_dict.keys():
                        map_net.load_state_dict(model_dict['params'])
                        map_net.eval()
                        train_res[m]['nll'].append(model_dict['nll'])
                        train_res[m]['acc'].append(model_dict['acc'])
                    else:
                        map_net.load_state_dict(model_dict)
                        map_net.eval()
                        train_res[m]['nll'].append(0.0)
                        train_res[m]['acc'].append(1.0)
            else:
                optimizer, scheduler = utils.optimizers.get_optim_sched(map_net, config['optim'], config['sched'], config['lr'], config['wd'], config['epochs'])
                train_loss, train_acc, _, _ = utils.training.training(train_loader=train_dataloader,test_loader=test_dataloader,
                                                                            model=map_net, loss_fn=loss_fn, optimizer=optimizer,
                                                                            scheduler=scheduler,epochs=config['epochs'],
                                                                            verbose=args.verbose, progress_bar=args.progress)
                train_res[m]['nll'].append(train_loss)
                train_res[m]['acc'].append(train_acc)
                model_dict = {'params': map_net.state_dict(),
                              'nll': train_loss,
                              'acc': train_acc}
                torch.save(model_dict, f'saved_models/{args.model}_trained_{args.dataset}.pt')

            if m == 'MAP' and args.dataset == 'imagenet':
              continue
            else:
              id_map_logits = pu.test_sampler(map_net, dataset.test_data, bs=config['bs'], probit=False)
              ood_map_logits = pu.test_sampler(map_net, dataset.ood_test_data, bs=config['bs'], probit=False)
              id_mean = torch.nn.functional.softmax(id_map_logits,dim=1).cpu(); id_var=None
              ood_mean = torch.nn.functional.softmax(ood_map_logits,dim=1).cpu(); ood_var=None
              id_predictions = id_mean; ood_predictions = ood_mean

        elif m == 'DNN-GLM':
            dnn_glm = LinearSampling.Posteriors.Posterior(network=map_net,
                                                                glm_type='DNN',
                                                                task='classification',
                                                                precision=precision)
            network_mean = True
            if not args.glm_pre_load:
                loss_dict = dnn_glm.train(train=dataset.training_data, 
                                    bs=config['dnn_bs'], 
                                    S=config['dnn_S'],
                                    gamma=config['dnn_gamma'], 
                                    lr=config['dnn_lr'], 
                                    epochs=config['dnn_epoch'], 
                                    mu=0.9,
                                    verbose=args.verbose,
                                    extra_verbose=args.extra_verbose,
                                    save_weights=f'saved_models/dnn_glm_{args.model}_{args.dataset}.pt',
                                    plot_loss_dir=res_dir,
                                    average=('moving' if args.dataset == 'imagenet' else 'running')
                                    )
                train_res[m]['nll'].append(loss_dict['mean_ce_loss'])
                train_res[m]['acc'].append(loss_dict['mean_accuracy'])
                pre_load = None
            else:
                train_res[m]['nll'].append(0.0)
                train_res[m]['acc'].append(1.0)
                pre_load = f'saved_models/dnn_glm_{args.model}_{args.dataset}.pt'
                dnn_glm.pre_load(pre_load)

            id_mean, id_var = dnn_glm.UncertaintyPrediction(test=dataset.test_data, bs=config['dnn_bs'], network_mean=network_mean)
            ood_mean, ood_var = dnn_glm.UncertaintyPrediction(test=dataset.ood_test_data, bs=config['dnn_bs'], network_mean=network_mean)

            var_function = lambda dataset_input : dnn_glm.UncertaintyPrediction(test=dataset_input, bs=config['dnn_bs'], network_mean=network_mean)[1] 

        elif m == 'LL-GLM':
            ll_glm = LinearSampling.Posteriors.Posterior(network=map_net,
                                                                glm_type='LL',
                                                                task='classification',
                                                                precision=precision)
            network_mean = True
            if not args.glm_pre_load:
                loss_dict = ll_glm.train(train=dataset.training_data, 
                                    bs=config['ll_bs'], 
                                    S=config['ll_S'],
                                    gamma=config['ll_gamma'], 
                                    lr=config['ll_lr'], 
                                    epochs=config['ll_epoch'], 
                                    mu=0.9,
                                    verbose=args.verbose,
                                    extra_verbose=args.extra_verbose,
                                    save_weights=f'saved_models/ll_{args.model}_{args.dataset}.pt',
                                    plot_loss_dir=res_dir,
                                    average=('moving' if args.dataset == 'imagenet' else 'running')
                                    )
                train_res[m]['nll'].append(loss_dict['mean_ce_loss'])
                train_res[m]['acc'].append(loss_dict['mean_accuracy'])
                pre_load = None
            else:
                train_res[m]['nll'].append(0.0)
                train_res[m]['acc'].append(1.0)
                pre_load = f'saved_models/ll_glm_{args.model}_{args.dataset}.pt'
                ll_glm.pre_load(pre_load)

            id_mean, id_var = ll_glm.UncertaintyPrediction(test=dataset.test_data, bs=config['ll_bs'], network_mean=network_mean)
            ood_mean, ood_var = ll_glm.UncertaintyPrediction(test=dataset.ood_test_data, bs=config['ll_bs'], network_mean=network_mean)
            var_function = lambda dataset_input : ll_glm.UncertaintyPrediction(test=dataset_input, bs=config['ll_bs'], network_mean=network_mean)[1] 

        elif m == 'DE':
            de_posterior = DeepEnsemble(network=map_net, task='classification', M = 10)
            train_nll, train_acc = de_posterior.train(loader=train_dataloader, 
                                                    lr=config['lr'], 
                                                    wd=config['wd'],
                                                    epochs=config['epochs'], 
                                                    optim_name=config['optim'], 
                                                    sched_name=config['sched'], 
                                                    verbose=args.verbose,
                                                    extra_verbose=args.extra_verbose)
            train_res[m]['nll'].append(train_nll)
            train_res[m]['acc'].append(train_acc)

            id_mean, id_var = de_posterior.UncertaintyPrediction(test_dataloader)
            ood_mean, ood_var = de_posterior.UncertaintyPrediction(ood_test_dataloader)

            var_function = lambda dataset : de_posterior.UncertaintyPrediction(DataLoader(dataset, config['bs']))[1]


        elif m == 'LLA':
            ## LLA definitions
            def predict(dataloader, la, link='probit'):
                py = []
                for x, _ in dataloader:
                    py.append(la(x.to(device), pred_type="glm", link_approx=link))
                return torch.cat(py).cpu()
                
            if args.dataset == 'imagenet':
                hessian_structure = 'diag'
            else:
                hessian_structure = 'kron'

            la = Laplace(map_net, "classification",
                        subset_of_weights="last_layer",
                        hessian_structure=hessian_structure)
            la.fit(train_dataloader, progress_bar=args.verbose)
            la.optimize_prior_precision(
                method="marglik",
                pred_type='glm',
                link_approx='probit',
                progress_bar=args.progress
            )
            
            print(f'MEM BEFORE LLA: {1e-9*torch.cuda.max_memory_allocated()}')

            if args.dataset == 'imagenet':
                T = 10
            else:
                T = 100

            id_mean, id_var, _ = pu.lla_sampler(dataset=dataset.test_data, 
                                                              model = lambda x : la.predictive_samples(x=x,pred_type='glm',n_samples=T), 
                                                              bs = 50)  # id_predictions -> S x N x C

            ood_mean, ood_var, _ = pu.lla_sampler(dataset=dataset.ood_test_data, 
                                                                    model = lambda x : la.predictive_samples(x=x,pred_type='glm',n_samples=T), 
                                                                    bs = 50)  # ood_predictions -> S x N x C
            
            var_function = lambda dataset : pu.lla_sampler(dataset=dataset, 
                                                                    model = lambda x : la.predictive_samples(x=x,pred_type='glm',n_samples=T), 
                                                                    bs = 50)[1]
            


        elif m == 'SWAG':
            if args.dataset == 'cifar100' or args.dataset == 'svhn':
                swag_lr = config['lr']
                swag_wd = config['wd']
            else:
                swag_lr = config['lr']*1e2
                swag_wd = 0
            swag_net = swag.SWAG(map_net,epochs = config['epochs'],lr = swag_lr, cov_mat = True,
                                max_num_models=config['S'], wd=swag_wd)
            swag_net.train_swag(train_dataloader=train_dataloader,progress_bar=args.verbose)

            T = (5 if args.subsample else 100) # Set a lower T for subsample experiments to save time/memory
            id_mean, id_var, _ = pu.swag_sampler(dataset=dataset.test_data,model=swag_net,T=T,n_output=dataset.n_output,bs=config['bs']) # id_predictions -> S x N x C
            ood_mean, ood_var, _ = pu.swag_sampler(dataset=dataset.ood_test_data,model=swag_net,T=T,n_output=dataset.n_output,bs=config['bs']) # ood_predictions -> S x N x C
            
            var_function = lambda input_dataset : pu.swag_sampler(dataset=input_dataset,model=swag_net,T=T,n_output=dataset.n_output,bs=config['bs'])[1]

        elif m == 'MC':
            p = 0.1
            T = 10
            dropout_posterior = mc.MCDropout(network=map_net,
                                 p=p)

            id_mean, id_var = dropout_posterior.mean_variance(test_dataloader, samples=T, verbose=args.verbose, extra_verbose=args.extra_verbose)
            ood_mean, ood_var = dropout_posterior.mean_variance(ood_test_dataloader, samples=T, verbose=args.verbose, extra_verbose=args.extra_verbose)
            
            var_function = lambda dataset : dropout_posterior.mean_variance(DataLoader(dataset, batch_size=config['bs']), samples=T, verbose=args.verbose)[1]

        # Record metrics
        t2 = time.time()
        test_res[m]['time'].append(t2-t1)
        test_res[m]['mem'].append(1e-9*torch.cuda.max_memory_allocated())

        if m == 'MAP' and args.dataset == 'imagenet':
          continue
        else:
          # [ce, acc, ece, oodauc, aucroc, vmsp_dict]
          metrics_m = metrics.compute_metrics(test_loader=test_dataloader, 
                                              id_mean=id_mean, id_var=id_var, 
                                              ood_mean=ood_mean, ood_var=ood_var, 
                                              variance=(m != 'MAP'), sum=False)
          # metrics_m = metrics.compute_metrics(test_dataloader, id_predictions.cpu(), ood_predictions.cpu(), samples=(m != 'MAP'))
  
          test_res[m]['nll'].append(metrics_m[0])
          test_res[m]['acc'].append(metrics_m[1])
          test_res[m]['ece'].append(metrics_m[2])
          test_res[m]['oodauc'].append(metrics_m[3])
          test_res[m]['aucroc'].append(metrics_m[4])
          if m != 'MAP':
              test_res[m]['varroc'].append(metrics_m[5])
              test_res[m]['varroc_id'].append(metrics_m[6])
              test_res[m]['varroc_ood'].append(metrics_m[7])
              prob_var_dict[m] = metrics_m[8]
              acc_conf, acc_var = metrics.tolerance_acc(mean=id_mean, variance=id_var, test_loader=test_dataloader)
              tolerance_acc_dict[m] = {'conf': acc_conf,
                                          'var': acc_var}
              if args.dataset == 'mnist' or args.dataset == 'fmnist':
                  varroc_rot = metrics.rotated_dataset(id_var, var_function, args.dataset)
              else:
                  varroc_rot = 0.00
              test_res[m]['varroc_rot'].append(varroc_rot)
  
          elif m == 'MAP':
              acc_conf = metrics.tolerance_acc(mean=id_mean, variance=id_var, test_loader=test_dataloader, deterministic=True)
              tolerance_acc_dict[m] = {'conf': acc_conf}
  
          metrics.print_results(m, test_res, ei, (train_res if m in train_methods else None))

    # Save predictions, variances for plotting
    if args.save_var:
        prob_var_dict = metrics.add_baseline(prob_var_dict,dataset.test_data,dataset.ood_test_data)
        torch.save(prob_var_dict,res_dir + f"prob_var_dict_{ei}.pt")

        metrics.plot_vmsp(prob_dict=prob_var_dict,
                          title=f'{args.dataset} {args.model}',
                          save_fig=res_dir + f"vmsp_plot.pdf")
        
        torch.save(tolerance_acc_dict, res_dir + f"tolerance_acc_dict_{ei}.pt")
        
    metrics.tolerance_plot(tolerance_acc_dict, save_fig=res_dir + f"tolerance_plot.pdf")

## Record results
res_text = res_dir + f"result.txt"
results = open(res_text,'w')

percentage_metrics = ['acc','ece','oodauc','aucroc','varroc']

results.write("Training Details:\n")
results.write(f"MAP/DE: epochs: {config['epochs']}; S: {config['S']}; lr: {config['lr']}; wd: {config['wd']}; bs: {config['bs']}; n_experiment: {config['n_experiment']}\n")
results.write(f"DNN-GLM: epochs: {config['dnn_epoch']}; S: {config['dnn_S']}; lr: {config['dnn_lr']}; bs: {config['dnn_bs']}; gamma: {config['dnn_gamma']}\n")
results.write(f"LL-GLM: epochs: {config['ll_epoch']}; S: {config['ll_S']}; lr: {config['ll_lr']}; bs: {config['ll_bs']}; gamma: {config['ll_gamma']}\n")

if config["n_experiment"] > 1:
    results.write("\nTrain Results:\n")
    for m in train_res.keys():
        results.write(f"{m}: ")
        for k in train_res[m].keys():
            results.write(f"{k}: {np.mean(train_res[m][k]):.4} +- {np.std(train_res[m][k]):.4}; ")
        results.write('\n')
    results.write("\nTest Prediction:\n")
    for m in test_res.keys():
        results.write(f"{m}: ")
        for k in test_res[m].keys():
            if k == 'varroc' and m != 'DNN-GLM':
                continue
            results.write(f"{k}: {np.mean(test_res[m][k]):.4} +- {np.std(test_res[m][k]):.4}; ")
        results.write('\n')
else:
    results.write("\nTrain Results:\n")
    for m in train_res.keys():
        results.write(f"{m}: ")
        for k in train_res[m].keys():
            results.write(f"{k}: {train_res[m][k][0]:.4}; ")
        results.write('\n')
    results.write("\nTest Prediction:\n")
    for m in test_res.keys():
        results.write(f"{m}: ")
        for k in test_res[m].keys():
            if (k == 'varroc' or k == 'varroc_id' or k == 'varroc_ood' or k == 'varroc_rot') and m == 'MAP':
                continue
            results.write(f"{k}: {test_res[m][k][0]:.4}; ")
        results.write('\n')

results.close()