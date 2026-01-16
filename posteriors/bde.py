import torch
from torch.func import functional_call
import tqdm
import sys
import utils.optimizers

def l_layer_params(net):
    theta_star_k = []
    sd = net.state_dict()
    sdk = sd.keys()
    for i,p in enumerate(sdk):
        if i < len(sdk) - 2:
            theta_star_k.append(sd[p].flatten(0))
        else:
            theta_star_k.append(torch.zeros(sd[p].flatten(0).shape))
    theta_star_k = torch.cat(theta_star_k)
    return theta_star_k

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)

def train(dataloader, model, optimizer, loss_function, scheduler=None):
    model.train()
    train_loss = 0
    for i, (X,y) in enumerate(dataloader):
        # Get and prepare inputs
        y = y.reshape(-1,1)
        
        # Perform forward pass
        pred = model(X)
        
        # Compute loss
        loss = loss_function(pred, y)
        
        # Perform backward pass
        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    train_loss = train_loss / (i+1)
    return train_loss

def train_bde(train_loader,net,delta,theta_k,loss_fn,Lambda,optim,sched):
    train_loss = 0
    for i, (X,y) in enumerate(train_loader):
        # Get and prepare inputs
        y = y.reshape(-1,1)
        # Compute prediction error
        pred = net(X)
        

        # Add delta function to outputs
        pred = pred + delta(X)

        # Calculate loss
        loss = loss_fn(y, pred)

        # Regularisation
        theta_t = torch.nn.utils.parameters_to_vector(net.parameters())
        diff = theta_t - theta_k
        reg = diff @ (Lambda * diff)
        loss = 0.5 * loss + 0.5 * reg

        # Backpropagation
        loss.backward()

        optim.step()
        if sched is not None:
            sched.step()
        optim.zero_grad()

        train_loss += loss.item()

    train_loss = train_loss / (i+1)
    return train_loss

def bde_weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        # torch.nn.init.normal_(m.weight,mean=0,std=1)
        torch.nn.init.normal_(m.bias,mean=0,std=1)

def _dub(x,y):
        return {yi:xi - y[yi] for (xi, yi) in zip(x, y)}

def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[i : i + n].view(tensor.shape))
        i += n
    return tuple(outList)

def init_weights_he(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
        torch.nn.init.constant_(m.bias, 0)

class bde_net(torch.nn.Module):
    def __init__(self, net):
        super().__init__()

        self.net = net
        
        # Get theta_k_star
        self.net.apply(init_weights_he)
        self.l_layer_params(self.net)

        # Get theta_k
        self.net.apply(init_weights_he)
        self.params = {k: v.clone().detach() for k, v in self.net.named_parameters()}

        self.device = next(self.net.parameters()).device

    def l_layer_params(self, net):
        theta_star_k = []
        sd = net.state_dict()
        sdk = sd.keys()
        for i,p in enumerate(sdk):
            if i < len(sdk) - 2:
                theta_star_k.append(sd[p].flatten(0))
            else:
                theta_star_k.append(torch.zeros(sd[p].flatten(0).shape))
        theta_star_k = torch.cat(theta_star_k).clone().detach()
        self.theta_star_k = theta_star_k

    def fnet(self, params, x):
        return functional_call(self.net, params, x)

    def jvp_func(self, theta_s,params,x):
        dparams = _dub(unflatten_like(theta_s, params.values()),params)
        _, proj = torch.func.jvp(lambda param: self.fnet(param, x),
                                (params,), (dparams,))
        proj = proj.detach()
        return proj    
    
    def forward(self, x):
        return self.net(x.to(self.device)) + self.jvp_func(self.theta_star_k, self.params, x.to(self.device))
    
class BayesianDeepEnsemble(object):
    def __init__(self, network, M, num_classes, target = 'multiclass'):
        self.network = network
        self.device = next(network.parameters()).device
        self.M = M
        self.num_classes = num_classes
        self.target = target

        self.network_list = [bde_net(self.network) for _ in range(self.M)]

    def train(self, loader, lr, wd, epochs, optim_name, sched_name, verbose=False, extra_verbose=False):
        if verbose:
            pbar = tqdm.trange(self.M)
        else:
            pbar = range(self.M)

        opt_list = []
        sched_list = []

        for i in range(self.M):
            opt, sched = utils.optimizers.get_optim_sched(self.network_list[i], optim_name, sched_name, lr, wd, epochs)
            opt_list.append(opt)
            sched_list.append(sched)

        total_loss, total_acc = 0, 0

        for idx in pbar:

            if extra_verbose:
                pbar_inner = tqdm.trange(epochs)
            else:
                pbar_inner = range(epochs)

            for _ in pbar_inner:

                if self.target == 'multiclass':
                    train_loss, train_acc = train_loop_multiclass(loader, self.network_list[idx], torch.nn.MSELoss(), opt, sched, self.num_classes)
                elif self.target == 'binary':
                    train_loss, train_acc = train_loop_binary(loader, self.network_list[idx], torch.nn.MSELoss(), opt, sched, self.num_classes)
                

                if extra_verbose:
                    metrics = {'Brier Loss': train_loss,
                    'Acc': train_acc}
                    pbar_inner.set_postfix(metrics)

            total_loss += train_loss
            total_acc += train_acc

            if verbose:
                metrics = {'Brier Loss': train_loss,
                   'Acc': train_acc}
                pbar.set_postfix(metrics)

        total_loss /= self.M
        train_acc /= self.M

        return total_loss, train_acc
    
    def test(self, loader):
        ensemble_pred = []
        for idx in range(self.M):
            network_pred = []
            for x,_ in loader:
                network_pred.append(self.network_list[idx](x.to(self.device)).detach())
            ensemble_pred.append(torch.cat(network_pred, dim=0))
        ensemble_pred = torch.stack(ensemble_pred)

        return ensemble_pred.detach() # M x N x C
    
    def UncertaintyPrediction(self, loader):
        predictions = self.test(loader) # M x N x C
        return predictions.softmax(-1).mean(0), predictions.softmax(-1).var(0)
    

def train_loop_multiclass(dataloader, net, loss_fn, optimizer, scheduler, num_classes):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)    
    train_loss, correct = 0, 0

    for batch, (x, y) in enumerate(dataloader):
        x,y = x.to(net.device), y.to(net.device)
        # Compute prediction and loss
        pred = net(x)
        loss = loss_fn(torch.nn.functional.softmax(pred, -1), torch.nn.functional.one_hot(y, num_classes=num_classes).to(dtype=torch.float64))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Evaluate metrics
        train_loss += loss.item()
        
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    if scheduler is not None:
        scheduler.step()

    train_loss /= num_batches
    correct /= size

    return train_loss, correct

def train_loop_binary(dataloader, net, loss_fn, optimizer, scheduler, num_classes):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)    
    train_loss, correct = 0, 0

    for batch, (x, y) in enumerate(dataloader):
        x,y = x.to(net.device), y.to(net.device)
        # Compute prediction and loss
        pred = net(x)
        loss = loss_fn(torch.nn.functional.sigmoid(pred), y.reshape(-1,1).to(dtype=torch.float64))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Evaluate metrics
        train_loss += loss.item()
        
        correct += (pred.sigmoid().round().squeeze(1) == y).type(torch.float).sum().item()

    if scheduler is not None:
        scheduler.step()

    train_loss /= num_batches
    correct /= size

    return train_loss, correct


def test_loop(dataloader, net, loss_fn, num_classes):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for x, y in dataloader:
            x,y = x.to(net.device), y.to(net.device)
            pred = net(x)
            test_loss += loss_fn(torch.nn.functional.softmax(pred, -1), torch.nn.functional.one_hot(y, num_classes=num_classes).to(dtype=torch.float64)).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    
    return test_loss, correct



    



