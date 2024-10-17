import datetime
import gc
import random

# import comet_ml
import sys
import os
import pandas as pd
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CometLogger
import random
sys.path.append('../../')
from Model import Transformer as ts
from Model import CNN as cn
from Model import CHB_Loader as cl
from Model import Convlstm as clm
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser
import torch
from Utils.Functions import get_total_patient_file, get_Net, write_metric_to_csv, append_metric, processing_metric, \
    get_DM, DetectionLatency, DetectionLatency_overall, Analyze_write_delay_to_csv, FDR_for_patient_ictal_file, \
    Analyze_write_delay_FDR_to_csv, processing_metric_withloss, append_metric_withloss, write_metric_to_csv_wloss

torch.set_float32_matmul_precision("medium")
project_path = os.path.abspath(os.path.relpath('../../', os.getcwd()))
log_path = os.path.join(project_path, 'BilinearNetwork/Data/Logs')
chb_root_path=os.path.join(project_path,'Dataset/CHB-MIT')
constrain_path=os.path.join(project_path,'BilinearNetwork/Data/Constraint/Detection')
seizure_table=pd.read_csv(os.path.join(constrain_path,'seizure_ictal_summary.csv'))
parser = ArgumentParser()

def get_data_path(args):
    return os.path.join(project_path, 'BilinearNetwork/Data/PreprocessedData/CHB-MIT/{}'.format(args.task))
def get_save_filename(args,file_exist_check_path):
    size_ref={0:"small",1:"large"}
    file_name="{}-{}".format(args.net,size_ref[args.size])
    if(os.path.exists(os.path.join(file_exist_check_path,file_name))):
        file_name=file_name+random.randint(0,50)
    return file_name

def train_one_patient(logger):
    data_path = get_data_path(args)

    if args.filename==None:
        args.filename=get_save_filename(args=args,file_exist_check_path=f'BilinearNetwork/Data/Result/{args.task}/fit/')
        print('filename in arg is None')
    else:
        print("filename in arg is not None, given:{}".format(args.filename))

    acc, f1, auc, ap ,latencys,loss = [], [], [], [],[],[]
    acc_pre, f1_pre, auc_pre, ap_pre,loss_pre = [], [], [], [],[]

    for i in range(get_total_patient_file(data_dir=data_path, patient_code=args.patient_id,task=args.task)):
        print(f"Training on patient {args.patient_id} leave_out id {i}")
        datamodule = get_DM(data_dir=data_path,args=args, leave_out_id=i)
        Net = get_Net(args)
        trainer = Trainer(logger=logger,
                          default_root_dir=log_path, precision="32-true", max_epochs=args.epochs, benchmark=True,
                          log_every_n_steps=5, callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')])
        result_pre = trainer.test(model=Net, datamodule=datamodule)[0]
        append_metric_withloss(acc_pre, f1_pre, auc_pre, ap_pre, loss_pre,result_pre)

        trainer.fit(model=Net, datamodule=datamodule, )

        result_post = trainer.test(model=Net, datamodule=datamodule)[0]
        del datamodule
        gc.collect()
        append_metric_withloss(acc, f1, auc, ap, loss,result_post)
        # latency=DetectionLatency_overall(chb_root_path=chb_root_path, model=Net, seizure_table=seizure_table,
        #                          patient_id=args.patient_id, leave_out_id=i,args=args)

        # FDR=FDR_for_patient_ictal_file(chb_root_path, constrain_path, model=Net, patient_id=args.patient_id)
        # print("Latency:{} FDR:{}".format(latency,FDR))

        # latencys.append(latency)
        # FDRs.append(FDR)


    metric_pre = processing_metric_withloss(acc_pre, f1_pre, auc_pre, ap_pre,loss_pre)
    write_metric_to_csv_wloss(file_path=os.path.join(project_path, f'BilinearNetwork/Data/Result/{args.task}/Non_fit/{args.filename}.xlsx'),
                        patient_id=args.patient_id, metric=metric_pre)
    metric_post = processing_metric_withloss(acc, f1, auc, ap,loss)
    write_metric_to_csv_wloss(file_path=os.path.join(project_path, f'BilinearNetwork/Data/Result/{args.task}/fit/{args.filename}.xlsx'),
                        patient_id=args.patient_id, metric=metric_post)
    Analyze_write_delay_to_csv(file_path=os.path.join(project_path, f'BilinearNetwork/Data/Result/{args.task}/detect_seizure/{args.filename}.xlsx'),
                        patient_id=args.patient_id, delays=latencys)


    # Analyze_write_delay_FDR_to_csv(file_path=os.path.join(project_path, f'BilinearNetwork/Data/Result/{args.task}/detect_seizure/{args.filename}.xlsx'),
    #                     patient_id=args.patient_id, delays=latencys,FDRs=FDRs)


    print(f"acc: Avg.{np.mean(acc):.2f}, Std.{np.std(acc):.2f}")
    print(f"f1: Avg.{np.mean(f1):.2f}, Std.{np.std(f1):.2f}")
    print(f"auc: Avg.{np.mean(auc):.2f}, Std.{np.std(auc):.2f}")
    print(f"ap: Avg.{np.mean(ap):.2f}, Std.{np.std(ap):.2f}")


def TrainOnePatient_oneSeizure():
    data_path = get_data_path(args)
    datamodule = get_DM(data_dir=data_path,args=args,leave_out_id=args.leave_out_id)
    Net = get_Net(args)

    trainer = Trainer(logger=False,
                      default_root_dir=log_path, precision="32-true", max_epochs=args.epochs, benchmark=True,
                      log_every_n_steps=5, callbacks=[EarlyStopping(monitor='val_loss', patience=2, mode='min')])
    trainer.test(model=Net, datamodule=datamodule)
    if args.sanity_test == 0:
        trainer.fit(model=Net, datamodule=datamodule,)
        trainer.test(model=Net, datamodule=datamodule)
    return None


def main(args):
    comet_logger = False
    if args.logger == 1:
        name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("Using Comet Logger")
        comet_logger = CometLogger(
            api_key='LXYMHm1xV9Y09OfGVdmB0YQUy',
            workspace="ziyitsang",  # Optional
            save_dir=log_path,  # Optional
            project_name="chb-dependent",  # Optional
            experiment_name=name,  # Optional

        )

    include_patient=[]
    if args.patient_id == 0 or args.Continue:
        if args.task=="prediction":
            include_patient = [1, 2, 3, 5, 9, 10, 14, 18, 19, 20, 21, 22, 23]
        elif "detection" in args.task:
            include_patient = [1, 2, 3, 4,5, 6,7,8,9, 10, 11,14, 17,18, 19, 20, 21, 22, 23]
        if args.Continue:
            start_index = include_patient.index(args.patient_id) if args.patient_id in include_patient else 0
            include_patient = include_patient[start_index:]

        for patient_id in include_patient:
            args.patient_id = patient_id
            train_one_patient(comet_logger)
    else:
        if args.leave_out_id == -1:
            train_one_patient(comet_logger)
        else:
            TrainOnePatient_oneSeizure()


# arg:
# train all patients: --patient_id=0
# train one patient, all seizures: --patient_id=1 --leave_out_id=-1
# train one patient, one seizure: --patient_id=1 --leave_out_id=k (k=0,1,2,3,4...)
# train one patient, one seizure, sanity test: --patient_id=1 --leave_out_id=k --sanity_test=1

if __name__ == '__main__':
    parser.add_argument('--net', type=str, default="transformer")
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patient_id', type=int, default=1)
    parser.add_argument('--logger', type=int, default=1)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--sanity_test', type=int, default=0)
    parser.add_argument('--leave_out_id', type=int, default=-1)
    parser.add_argument('--size', type=int, default=1)
    parser.add_argument('--task', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('-Continue',action='store_true')

    # parser=ts.TransformerBlock.add_model_specific_args(parser)
    # parser=clm.ConvLSTMCompose.add_model_specific_args(parser)
    # args = parser.parse_args()
    print(args)
    main(args)
