import torch
import scipy
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

## Calibration function (ECE)
def calibration_curve_r(target,mean,variance,c):
    target = target.detach().cpu(); mean = mean.detach().cpu(); variance = variance.detach().cpu()
    predicted_conf = torch.linspace(0,1,c)
    observed_conf = torch.empty((c))
    for i,ci in enumerate(predicted_conf):
        z = scipy.stats.norm.ppf((1+ci)/2)
        ci_l = mean.reshape(-1) - z*torch.sqrt(variance.reshape(-1))
        ci_r = mean.reshape(-1) + z*torch.sqrt(variance.reshape(-1)) 
        observed_conf[i] = torch.logical_and(ci_l < target.reshape(-1), target.reshape(-1) < ci_r).type(torch.float).mean()
    return observed_conf,predicted_conf

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        # nn.init.normal_(m.weight,mean=0,std=1)
        torch.nn.init.normal_(m.bias,mean=0,std=1)


def train(dataloader, model, optimizer, loss_function, scheduler=None):
    model.train()
    train_loss = 0
    for i, (X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)
        # Get and prepare inputs
        y = y.reshape(-1,1)
        
        # Perform forward pass
        pred = model(X)
        
        # Compute loss
        loss = loss_function(pred, y)
        
        # Perform backward pass
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        

        train_loss += loss.item()

    train_loss = train_loss / (i+1)
    return train_loss

def test(dataloader, model, my, sy, loss_function):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X,y = X.to(device), y.to(device)
            y = y.reshape((y.shape[0],1))
            pred = model(X)
            pred = pred * sy + my
            test_loss += loss_function(pred, y).item()
    test_loss /= (i+1)
    return test_loss

def to_np(x):
    return x.cpu().detach().numpy()