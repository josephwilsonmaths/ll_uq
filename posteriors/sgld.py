from posteriors.sgld_lightning import SGLDClassification
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
import tqdm

from lightning_uq_box.viz_utils import (
    plot_training_metrics,
)
import tempfile
from lightning import LightningDataModule
import os

import torch
from torch.utils.data import DataLoader


class sgld(object):
    def __init__(self, net, loss_fn, lr, lr_final, max_itr, wd, nf, S, epochs, res_dir=None):
        self.net = net
        self.device = next(self.net.parameters()).device
        self.loss_fn = loss_fn
        self.lr = lr
        self.lr_final = lr_final
        self.max_itr = max_itr
        self.wd = wd
        self.nf = nf
        self.epochs = epochs
        self.task = 'multiclass' if isinstance(loss_fn, torch.nn.CrossEntropyLoss) else ('binary' if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss) else None)
        self.sgldmodel = SGLDClassification(model=self.net,
                                    loss_fn=self.loss_fn,
                                    lr=self.lr,
                                    lr_final=self.lr_final,
                                    max_itr = self.max_itr,
                                    weight_decay=self.wd,
                                    noise_factor=self.nf,
                                    n_sgld_samples=S,
                                    task=self.task)
        self.res_dir = res_dir
        
        # self.my_temp_dir = tempfile.mkdtemp(dir=res_dir)

        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)

        self.logger = CSVLogger(self.res_dir)

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
        self.lightningdata = LightningDataset(train=train_data,
                                        test=test_data,
                                        train_bs=batch_size,
                                        test_bs=batch_size,
                                        task=self.task
                    )
        
        self.trainer.fit(self.sgldmodel, self.lightningdata)

        if plot_loss:
            fig = plot_training_metrics(
                os.path.join(self.res_dir, "lightning_logs"), ["train_loss", "trainAcc"]
            )
            fig.savefig(os.path.join(self.res_dir, "lightning_logs.pdf"), format='pdf')
        
    def eval(self, x):
        return self.sgldmodel.predict_step(x.to(self.device), device=self.device)['logits']
    
    def test(self, loader, verbose=False):
        predictions = []
        if verbose:
            pbar = tqdm.tqdm(loader)
        else:
            pbar = loader
        for x,_ in pbar:
            predictions.append(self.eval(x=x).detach()) # output is (Nb x S x C)
        predictions_sgld = torch.cat(predictions, dim=0).permute(2,0,1)  # output is (S x N x C)
        return predictions_sgld.detach()
    
    def UncertaintyPrediction(self, loader, verbose=False):
        predictions = self.test(loader, verbose) # S x N x C
        return predictions.softmax(-1).mean(0), predictions.softmax(-1).var(0)


def collate_fn_tensordataset(batch, task):
    """Collate function for tensor dataset to our framework."""
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([torch.tensor(item[1]) for item in batch])
    if task == 'binary':
        targets = targets.reshape(-1,1).to(dtype=torch.float64)
    return {"input": inputs, "target": targets}

class LightningDataset(LightningDataModule):
    def __init__(
        self,
        train,
        test,
        train_bs,
        test_bs,
        task
    ) -> None:
        """Initialize a new Toy Data Module instance.
        """
        super().__init__()

        self.train = train
        self.test = test
        self.train_bs = train_bs
        self.test_bs = test_bs
        self.task = task

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        return DataLoader(self.train,self.train_bs, collate_fn=lambda batch: collate_fn_tensordataset(batch, task=self.task))

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(self.test,self.test_bs, collate_fn=lambda batch: collate_fn_tensordataset(batch, task=self.task))