import tempfile
import copy
import torch
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from lightning_uq_box.uq_methods import SNGPClassification
from lightning_uq_box.viz_utils import (
    plot_training_metrics,
)
from functools import partial
import os
import utils.training
import tqdm

class SNGP(object):
    def __init__(self, network, num_classes, input_size, T, loss_fn, lr, epochs, wd, optim, set_dtype = torch.float64, sched=None, res_dir=None):

        child_list = list(copy.deepcopy(network).children())
        if len(child_list) > 1:
            child_list = child_list
        elif len(child_list) == 1:
            child_list = child_list[0]
        first_layers = child_list[:-1]
        self.network = torch.nn.Sequential(*first_layers)

        self.network.apply(utils.training.init_weights)

        # self.network = network
        self.num_classes = num_classes
        self.input_size = input_size
        self.T = T
        self.loss_fn = loss_fn
        self.lr = lr
        self.epochs = epochs
        self.wd = wd
        self.optim = optim
        self.sched = sched
        self.res_dir = res_dir
        self.dtype = set_dtype

        if self.optim == 'adam':
            self.optimizer = partial(torch.optim.Adam, lr=lr, weight_decay=wd)  
        elif self.optim == 'sgd':
            self.optimizer = partial(torch.optim.SGD, lr=lr, momentum=0.9, weight_decay=wd)    
        else:
            print("Invalid optimizer choice. Valid choices: [adam, sgd]")

        if self.sched == 'cosine':
            self.scheduler = partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max = epochs)
        else:
            self.scheduler = None

        self.res_dir = res_dir
        
        # self.my_temp_dir = tempfile.mkdtemp(dir=res_dir)

        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)

        self.logger = CSVLogger(self.res_dir)

        self.sngp = SNGPClassification(
            feature_extractor=self.network,
            loss_fn=self.loss_fn,
            num_targets=self.num_classes,
            input_size = self.input_size,
            optimizer = self.optimizer,
            lr_scheduler=self.scheduler
        )

        self.trainer = Trainer(
            max_epochs=self.epochs,  # number of epochs we want to train
            logger=self.logger,  # log training metrics for later evaluation
            log_every_n_steps=1,
            enable_checkpointing=False,
            enable_progress_bar=True,
            default_root_dir=self.res_dir,
            enable_model_summary=False
        )

    def train(self, train_data, test_data, batch_size, plot_loss=False):
        lightningdataset = LightningDataset(train=train_data,
                        test=test_data,
                        train_bs=batch_size,
                        test_bs=batch_size,
        )

        self.trainer.fit(self.sngp, lightningdataset)
        if plot_loss:
            fig = plot_training_metrics(
                os.path.join(self.res_dir, "lightning_logs"), ["train_loss", "trainAcc"]
            )
            fig.savefig(os.path.join(self.res_dir, "lightning_logs.pdf"), format='pdf')

        self.sngp.recompute_covariance_matrix()
        
    def eval(self, x):
        preds = self.sngp.predict_step(x)
        return (torch.randn(self.T,x.shape[0],self.num_classes)*preds['pred_uct'].reshape(1,-1,1) + preds['pred']).detach()
    
    def test(self, loader, verbose=False):
        predictions = []
        if verbose:
            pbar = tqdm.tqdm(loader)
        else:
            pbar = loader
        for x,_ in pbar:
            predictions.append(self.eval(x=x).detach()) # output is (S x Nb x C)
        predictions_sngp = torch.cat(predictions, dim=1)  # output is (S x N x C)
        return predictions_sngp.detach()
    
    def UncertaintyPrediction(self, loader, verbose=False):
        predictions = self.test(loader, verbose) # S x N x C
        return predictions.softmax(-1).mean(0), predictions.softmax(-1).var(0)

def collate_fn_tensordataset(batch, set_dtype=torch.float64):
    """Collate function for tensor dataset to our framework."""
    inputs = torch.stack([item[0].to(set_dtype) for item in batch])
    targets = torch.stack([torch.tensor(item[1]) for item in batch])
    return {"input": inputs, "target": targets}

class LightningDataset(LightningDataModule):

    def __init__(
        self,
        train,
        test,
        train_bs,
        test_bs,
        set_dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.train = train
        self.test = test
        self.train_bs = train_bs
        self.test_bs = test_bs
        self.set_dtype = set_dtype

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        return DataLoader(self.train,self.train_bs, shuffle=True, collate_fn= lambda batch: collate_fn_tensordataset(batch, set_dtype=self.set_dtype))

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(self.test,self.test_bs, shuffle=True, collate_fn= lambda batch: collate_fn_tensordataset(batch, set_dtype=self.set_dtype))