import comet_ml
import sys
import os
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CometLogger
sys.path.append('../../')
from Model import Transformer as ts
from Model import CNN as cn
from Model import CHB_Loader as cl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser
import torch


torch.set_float32_matmul_precision("medium")
project_path = os.path.abspath(os.path.relpath('../../', os.getcwd()))
log_path = os.path.join(project_path, 'BilinearNetwork/Data/Logs')

parser = ArgumentParser()
def main(args):
    comet_logger = False
    if args.logger == 1:
        print("Using Comet Logger")
        comet_logger = CometLogger(
            api_key='LXYMHm1xV9Y09OfGVdmB0YQUy',
            workspace="ziyitsang",  # Optional
            save_dir=log_path,  # Optional
            project_name="chb-dependent",  # Optional
            experiment_name="independent_test-2",  # Optional

        )

    # ----
    if args.datamodule=="Independent":
        datamodule = cl.CHBIndependentDM(
        root_dir=os.path.join(project_path, 'BilinearNetwork/Data/PreprocessedData/CHB-MIT/Concanate'),
        batch_size=args.batch_size,)
    elif args.datamodule=="SanityCheck":
        datamodule = cl.SanityCheckDM(
            root_dir=os.path.join(project_path, 'BilinearNetwork/Data/PreprocessedData/CHB-MIT/Concanate'),
            batch_size=args.batch_size, )

    trainer = Trainer(logger=comet_logger,
                      default_root_dir=log_path, precision="32-true", max_epochs=args.epochs, benchmark=True,
                      log_every_n_steps=5)
    # ----
    if args.net=="transformer":
        Net = ts.Transformer(n_heads=args.n_heads, n_layers=args.n_layers, lr=args.lr, d_in=args.d_in)
    elif args.net=="CNN":
        Net = cn.ConvNet(lr=args.lr)
    else:
        raise ValueError("Invalid Net Type")

    trainer.fit(model=Net, datamodule=datamodule,)

if __name__ == '__main__':
    parser.add_argument('--net', type=str, default="transformer")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patient_id', type=int, default=1)
    parser.add_argument('--logger', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--datamodule', type=str, default='Independent')
    parser=ts.Transformer.add_model_specific_args(parser)
    # args = parser.parse_args()
    print(args)
    main(args)