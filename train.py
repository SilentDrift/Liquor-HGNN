import argparse
import os
import random
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from torch_geometric.data import LightningDataset
from torch_geometric.loader import DataLoader

from models.liquor_gnn import LiquorGNNModelPL
from dataset import BattleDIMDataset


def _set_deterministic(seed: int) -> None:
    """Force deterministic behaviour across the training stack."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(data_module, args):
    _set_deterministic(args.seed)

    sample = data_module.train_dataset.get(0)
    model = LiquorGNNModelPL(sample.metadata(), args)
    model.create_mapping(data_module.train_dataset)

    auto_lr = True if args.lr == "auto" else False

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=args.epochs,
        accelerator="cpu" if args.no_cuda else "gpu",
        auto_scale_batch_size=None,                 # run batch size scaling, result overrides hparams.batch_size
        auto_lr_find=auto_lr                           # run learning rate finder, results override hparams.learning_rate
    )

    # call tune to find the batch_size and to optimize lr
    trainer.tune(model, data_module)
    data_module.kwargs["batch_size"] = model.batch_size         # otherwise the tune function did not change the batch_size properly
    trainer.fit(model, data_module)
    torch.save(model.state_dict(), model.prefix + "/LiquorGNN.statedict")
    del data_module

    val_ds = BattleDIMDataset("./", mode="val")
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    trainer.validate(model, val_dl)
    del val_ds, val_dl

    test_ds = BattleDIMDataset("./", mode="test")
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    trainer.test(model, test_dl)


def getArgs(argv=None):
    parser = argparse.ArgumentParser(description="LiquorNAM")
    parser.add_argument("--layer", default="GATConv", help="options: 'GATConv', 'SAGEConv', 'GraphConv', 'LEConv'")
    parser.add_argument("--epochs", type=int, default=50, metavar="N")
    parser.add_argument('--hidden-channels', type=int, default=64, metavar="N")
    parser.add_argument('--batch-size', type=int, default=512, metavar="N")
    parser.add_argument("--lr", default="auto", metavar="lr", help="initial learning rate for optimizer e.g.: 1e-4 | 'auto'")
    parser.add_argument('--num-layer', type=int, default=10, metavar="N")
    parser.add_argument('--seed', type=int, default=0, metavar="N")
    parser.add_argument("--no-cuda", action="store_true", default=False)

    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        args.__dict__["no_cuda"] = True

    return args


if __name__ == "__main__":
    args = getArgs(sys.argv[1:])
    pl.seed_everything(args.seed)
    _set_deterministic(args.seed)

    train_ds = BattleDIMDataset("./", mode="train")

    data_module = LightningDataset(
        train_dataset=train_ds,
        batch_size=args.batch_size,
        num_workers=0,
    )

    train(data_module, args)
