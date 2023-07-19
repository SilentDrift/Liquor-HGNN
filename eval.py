import datetime
import json
import sys
import os
import torch
import torch_geometric as pyg
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import argparse
from models.liquor_gnn import LiquorGNNModelPL
from dataset import BattleDIMDataset


def getArgs(argv=None):
    parser = argparse.ArgumentParser(description="LiquorNAM")
    parser.add_argument("weights", metavar="Path",
                        help="path to pretrained weights, e.g. models/pretrained/GATConv.statedict")
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

    starttime = datetime.datetime.now()
    starttime = starttime.strftime("%H:%M:%S")

    train_ds = BattleDIMDataset("./", mode="train")

    model = LiquorGNNModelPL(train_ds.get(0).metadata(), args)
    model.create_mapping(train_ds)
    model.load_state_dict(torch.load(args.weights))

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=0,
        gpus=0 if args.no_cuda else 1,
    )

    val_ds = BattleDIMDataset("./", mode="val")
    test_ds = BattleDIMDataset("./", mode="test")

    val_dl = DataLoader(val_ds, batch_size=512, shuffle=False)
    trainer.validate(model, val_dl)

    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    trainer.test(model, test_dl)