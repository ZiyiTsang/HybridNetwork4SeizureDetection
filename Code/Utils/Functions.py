import gc
import glob
import os
import sys
from collections import deque

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

sys.path.append('../')
from Model import Transformer as ts
from Model import CNN as cn
from Model import CommonBlock as cb
from Model import Convlstm as clm
from Model import CHB_Loader as cl

import Utils.Preprocess as ut
import pandas as pd
import numpy as np


def get_total_patient_file(data_dir, patient_code,task) -> int:
    """
    Get the total number of patient files of specific patient code in the data directory.
    """
    count=0
    patient_predix = f"chb{patient_code:02d}"
    if task=="prediction":
        count = len(glob.glob(os.path.join(data_dir, f"{patient_predix}*_preictal.npy")))
    elif "detection" in task:
        count = len(glob.glob(os.path.join(data_dir, f"{patient_predix}-normal*.npz")))
    assert count > 0, f"No file found in {data_dir} with patient code {patient_code}"
    return count


# if __name__ == "__main__":
#     data_dir = "E:\Research\BilinearNetwork\Data\PreprocessedData\CHB-MIT\Prediction"
#     patient_code = 2
#     print(get_total_patient_file(data_dir, patient_code))
#     # Output: 20
def get_Net(args):
    """
    Get the network model based on the arguments.
    """

    if args.net == "transformer":
        if args.size == 1:
            encoder = ts.TransformerBlock(d_in=65*22,n_heads=8,num_layers=8)
        else:
            encoder = ts.TransformerBlock(d_in=65*22,n_heads=6,num_layers=3)
    elif args.net == "CNN":
        if args.size == 1:
            encoder = cn.ConvNetBlock_large()
        else:
            encoder = cn.ConvNetBlock_small()
    elif args.net == "LSTM":
        if args.size == 1:
            encoder = clm.ConvLSTMCompose(22, 96, (3, 3), 2, True, True, False)
        else:
            encoder = clm.ConvLSTMCompose(22, 48, (3, 3), 1, True, True, False)
    elif args.net == "transformerCNN":
        # encoder1 = ts.TransformerBlock(d_in=65*22,n_heads=6,num_layers=3)
        encoder1 = ts.TransformerBlock(d_in=65 * 22, n_heads=8, num_layers=8)
        encoder2 = cn.ConvNetBlock_large()
        encoder=cb.CombineEncoder(encoder1,encoder2)
    elif args.net == "transformerLSTM":
        encoder1 = ts.TransformerBlock(d_in=65*22,n_heads=6,num_layers=3)
        encoder2 = clm.ConvLSTMCompose(22, 96, (3, 3), 2, True, True, False)
        encoder=cb.CombineEncoder(encoder1,encoder2)
    elif args.net == "bCNN":
        encoder1 = clm.ConvLSTMCompose(22, 96, (3, 3), 2, True, True, False)
        encoder2 = clm.ConvLSTMCompose(22, 48, (3, 3), 1, True, True, False)
        encoder = cb.CombineEncoder(encoder1, encoder2)
    elif args.net == "bTransformer":
        encoder1 = ts.TransformerBlock(d_in=65*22,n_heads=6,num_layers=3)
        encoder2 = ts.TransformerBlock(d_in=65*22,n_heads=8,num_layers=8)
        encoder = cb.CombineEncoder(encoder1, encoder2)
    elif args.net == "bLSTM":
        encoder1 = clm.ConvLSTMCompose(22, 96, (3, 3), 2, True, True, False)
        encoder2 = clm.ConvLSTMCompose(22, 48, (3, 3), 1, True, True, False)
        encoder = cb.CombineEncoder(encoder1, encoder2)
    else:
        raise ValueError("Invalid Net Type")

    Net = cb.CommonNet(encoder=encoder, lr=args.lr)

    return Net

def get_DM(data_dir,args,leave_out_id=None):
    if args.task == "prediction":
        DM = cl.CHBDependentDMT(
            data_dir=data_dir,
            patient_id=args.patient_id, leave_out_id=leave_out_id,
            batch_size=args.batch_size)
    elif "detection"in args.task:
        DM = cl.CHBDependentDETTECTION(
            data_dir=data_dir,
            patient_id=args.patient_id, leave_out_id=leave_out_id,
            batch_size=args.batch_size)
    else:
        raise ValueError("Invalid Task Type")
    return DM


def write_metric_to_csv(file_path, patient_id, metric):
    """
    Write the metric to the file. if the file does not exist, create the file.
    """
    ac = metric['acc']
    f1 = metric['f1']
    auc = metric['auc']
    ap = metric['ap']
    #     check if the file and record exist
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        if df[df['patient_id'] == patient_id].shape[0] > 0:
            df.loc[df['patient_id'] == patient_id, 'acc'] = ac
            df.loc[df['patient_id'] == patient_id, 'f1'] = f1
            df.loc[df['patient_id'] == patient_id, 'auc'] = auc
            df.loc[df['patient_id'] == patient_id, 'ap'] = ap
        else:
            # df=df.append({'patient_id':patient_id,'acc':ac,'f1':f1,'auc':auc,'ap':ap},ignore_index=True)
            df = pd.concat(
                [df, pd.DataFrame({'patient_id': [patient_id], 'acc': [ac], 'f1': [f1], 'auc': [auc], 'ap': [ap]})],
                ignore_index=True)
    else:
        df = pd.DataFrame({'patient_id': [patient_id], 'acc': [ac], 'f1': [f1], 'auc': [auc], 'ap': [ap]})
    df.to_excel(file_path, index=False)

def write_metric_to_csv_wloss(file_path, patient_id, metric):
    """
    Write the metric to the file. if the file does not exist, create the file.
    """
    ac = metric['acc']
    f1 = metric['f1']
    auc = metric['auc']
    ap = metric['ap']
    loss=metric['loss']
    #     check if the file and record exist
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        if df[df['patient_id'] == patient_id].shape[0] > 0:
            df.loc[df['patient_id'] == patient_id, 'acc'] = ac
            df.loc[df['patient_id'] == patient_id, 'f1'] = f1
            df.loc[df['patient_id'] == patient_id, 'auc'] = auc
            df.loc[df['patient_id'] == patient_id, 'ap'] = ap
            df.loc[df['patient_id'] == patient_id, 'loss'] = loss
        else:
            # df=df.append({'patient_id':patient_id,'acc':ac,'f1':f1,'auc':auc,'ap':ap},ignore_index=True)
            df = pd.concat(
                [df, pd.DataFrame({'patient_id': [patient_id], 'acc': [ac], 'f1': [f1], 'auc': [auc], 'ap': [ap],'loss':[loss]})],
                ignore_index=True)
    else:
        df = pd.DataFrame({'patient_id': [patient_id], 'acc': [ac], 'f1': [f1], 'auc': [auc], 'ap': [ap],'loss':[loss]})
    df.to_excel(file_path, index=False)


def batch_zscore_normalize(batch_array):
    normalized_batch_array = np.zeros_like(batch_array)
    for i in range(batch_array.shape[0]):
        sample = batch_array[i]
        mean_val = np.mean(sample)
        std_dev = np.std(sample)
        normalized_sample = (sample - mean_val) / std_dev
        normalized_batch_array[i] = normalized_sample
    return normalized_batch_array


def append_metric(acc, f1, auc, ap, result_list):
    acc.append(result_list['test_acc_epoch'])
    f1.append(result_list['test_f1_epoch'])
    auc.append(result_list['test_auc_epoch'])
    ap.append(result_list['test_ap_epoch'])

def append_metric_withloss(acc, f1, auc, ap, loss,result_list):
    acc.append(result_list['test_acc_epoch'])
    f1.append(result_list['test_f1_epoch'])
    auc.append(result_list['test_auc_epoch'])
    ap.append(result_list['test_ap_epoch'])
    loss.append(result_list['test_loss_epoch'])


def processing_metric(acc, f1, auc, ap):
    acc, f1, auc, ap = np.array(acc), np.array(f1), np.array(auc), np.array(ap)
    acc, f1, auc, ap = acc[~np.isnan(acc)], f1[~np.isnan(f1)], auc[~np.isnan(auc)], ap[~np.isnan(ap)]
    metric = {'acc': '{:.2f}({:.2f})'.format(np.mean(acc), np.std(acc)),
              'f1': '{:.2f}({:.2f})'.format(np.mean(f1), np.std(f1)),
              'auc': '{:.2f}({:.2f})'.format(np.mean(auc), np.std(auc)),
              'ap': '{:.2f}({:.2f})'.format(np.mean(ap), np.std(ap))}
    return metric
def processing_metric_withloss(acc, f1, auc, ap,loss):
    acc, f1, auc, ap ,loss= np.array(acc), np.array(f1), np.array(auc), np.array(ap),np.array(loss)
    acc, f1, auc, ap, loss = acc[~np.isnan(acc)], f1[~np.isnan(f1)], auc[~np.isnan(auc)], ap[~np.isnan(ap)],loss[~np.isnan(loss)]
    metric = {'acc': '{:.2f}({:.2f})'.format(np.mean(acc), np.std(acc)),
              'f1': '{:.2f}({:.2f})'.format(np.mean(f1), np.std(f1)),
              'auc': '{:.2f}({:.2f})'.format(np.mean(auc), np.std(auc)),
              'ap': '{:.2f}({:.2f})'.format(np.mean(ap), np.std(ap)),
              'loss': '{:.2f}({:.2f})'.format(np.mean(loss), np.std(loss)),
              }

    return metric


def DetectionLatency(model, file_path, seizure_start, seizure_stop, windows_size,threshold=0.5, queue_size=10, threshold_count=7):
    raw_T = ut.PreprocessTool(file_path).do_preprocess()

    def get_latency(_data_stft):
        data_ds = TensorDataset(torch.Tensor(_data_stft))
        del _data_stft
        data_loader = DataLoader(data_ds, batch_size=1)
        one_sample_delay = 1 / raw_T.raw.info['sfreq']
        model.eval()
        y_pred_queue = deque(maxlen=queue_size)
        for i, data in enumerate(tqdm(data_loader)):
            data = data[0]
            y_pred = model(data).view(-1)
            y_pred_queue.append(y_pred.item())
            if sum(1 for y in y_pred_queue if y > threshold) >= threshold_count:
                current_delay = i * one_sample_delay
                model.train()
                print(current_delay)
                return current_delay
        model.train()
        return -1

    whole_start, whole_stop = seizure_start - windows_size, seizure_stop + windows_size
    partial_start, partial_stop = whole_start, whole_stop-((whole_stop-whole_start)/4*3)
    if(partial_stop-partial_start>20) :partial_stop=partial_start+20
    if(partial_stop-partial_start<10):partial_stop=whole_stop-((whole_stop-whole_start)/4*1)
    if(whole_stop-whole_start>30) :whole_stop=whole_start+30
    print(whole_start,whole_stop)
    print(partial_start,partial_stop)

    data_stft = raw_T.overlap_events_slice_all(start=partial_start, stop=partial_stop,duration=windows_size).cut_epochs().get_epochs_stft()
    latency = get_latency(data_stft)
    if (latency == -1):
        del data_stft
        gc.collect()
        raw_T.clear()
        data_stft = raw_T.overlap_events_slice_all(start=whole_start, stop=whole_stop,duration=windows_size).cut_epochs().get_epochs_stft()
        latency = get_latency(data_stft)
    return latency


def DetectionLatency_overall(chb_root_path, model, seizure_table, patient_id, leave_out_id,args):
    if args.task == "detection2s":
        print("Detection2s")
        windows_size = 2
    elif args.task == "detection" or args.task == "prediction":
        print("5s")
        windows_size = 5
    else:
        raise ValueError("Invalid Task Type, prediction or detection")
    patient_name = "chb" + str(patient_id).zfill(2)
    seizure_table_df = seizure_table[seizure_table['File Name'].str.startswith(patient_name)]
    row = seizure_table_df.iloc[leave_out_id, :]
    file_name = row['File Name']
    file_path = os.path.join(chb_root_path, "{}/{}".format(patient_name, file_name))
    start_time = row['Start Time']
    end_time = row['End Time']
    latency = DetectionLatency(model=model, file_path=file_path, seizure_start=start_time, seizure_stop=end_time,windows_size=windows_size)
    return latency


def analyze_delay(arr):
    valid_numbers = [num for num in arr if num != -1]
    valid_numbers = [x for x in valid_numbers if isinstance(x, (int, float))]

    mean_value = np.mean(valid_numbers) if valid_numbers else float('nan')
    std_value = np.std(valid_numbers) if valid_numbers else float('nan')

    count_negative_ones = arr.count(-1)
    count_valid_numbers = len(valid_numbers)

    return {
        "mean": mean_value,
        "std": std_value,
        "un_detect": count_negative_ones,
        "detect": count_valid_numbers
    }


def Analyze_write_delay_to_csv(file_path, patient_id, delays):
    """
    Write the metric to the file. if the file does not exist, create the file.
    """
    result = analyze_delay(delays)
    delay_mean, delay_std, un_detect_num, detect_num = result['mean'], result['std'], result['un_detect'], result[
        'detect']
    delay_str = "{}({})".format(delay_mean, delay_std)
    detect_num = "{}/{}".format(detect_num, detect_num + un_detect_num)
    #     check if the file and record exist
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        if df[df['patient_id'] == patient_id].shape[0] > 0:
            df.loc[df['patient_id'] == patient_id, 'patient_id'] = patient_id
            df.loc[df['patient_id'] == patient_id, 'delay'] = delay_str
            df.loc[df['patient_id'] == patient_id, 'detected'] = detect_num
        else:
            df = pd.concat(
                [df, pd.DataFrame({'patient_id': [patient_id], 'delay': [delay_str], 'detected': [detect_num]})],
                ignore_index=True)
    else:
        df = pd.DataFrame({'patient_id': [patient_id], 'delay': [delay_str], 'detected': [detect_num]})
    df.to_excel(file_path, index=False)

def Analyze_write_delay_FDR_to_csv(file_path, patient_id, delays,FDRs):
    """
    Write the metric to the file. if the file does not exist, create the file.
    """
    result = analyze_delay(delays)
    delay_mean, delay_std, un_detect_num, detect_num = result['mean'], result['std'], result['un_detect'], result[
        'detect']
    FDR_mean,FDR_std=np.mean(FDRs),np.std(FDRs)
    delay_str = "{}({})".format(delay_mean, delay_std)
    detect_num = "{}/{}".format(detect_num, detect_num + un_detect_num)
    FDR_str="{}({})".format(FDR_mean, FDR_std)
    #     check if the file and record exist
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        if df[df['patient_id'] == patient_id].shape[0] > 0:
            df.loc[df['patient_id'] == patient_id, 'patient_id'] = patient_id
            df.loc[df['patient_id'] == patient_id, 'delay'] = delay_str
            df.loc[df['patient_id'] == patient_id, 'detected'] = detect_num
            df.loc[df['patient_id'] == patient_id, 'FDR'] = FDR_str
        else:
            df = pd.concat(
                [df, pd.DataFrame({'patient_id': [patient_id], 'delay': [delay_str], 'detected': [detect_num],'FDR':[FDR_str]})],
                ignore_index=True)
    else:
        df = pd.DataFrame({'patient_id': [patient_id], 'delay': [delay_str], 'detected': [detect_num],'FDR':[FDR_str]})
    df.to_excel(file_path, index=False)


def get_interictal_files_name(chb_root_path, constrain_path, patient_id):
    def find_all_files(chb_root_path):
        all_files = []
        for root, dirs, files in os.walk(chb_root_path):
            for file in files:
                if file.endswith('.edf'):
                    all_files.append(file)
        return all_files

    exclude_file = pd.read_csv(os.path.join(constrain_path, 'exclude_File.csv'))
    exclude_patient = pd.read_csv(os.path.join(constrain_path, 'exclude_Patient.csv'))
    small_constrant = list(exclude_file['File Name'])
    exclude_patient = list(exclude_patient['0'])
    large_table = find_all_files(chb_root_path)
    select_files = large_table.copy()
    for file_name in large_table:
        prelix = file_name.split('_')[0]
        if prelix in exclude_patient:
            select_files.remove(file_name)
            continue
        if file_name in small_constrant:
            select_files.remove(file_name)
            continue
    select_file_final = []
    for i in range(len(select_files) - 1):
        current_num = int(select_files[i].split('_')[1].split('.')[0])
        prelix = select_files[i].split('_')[0]
        next_num = int(select_files[i + 1].split('_')[1].split('.')[0])
        if abs(next_num - current_num) > 1:
            select_file_final.append(select_files[i])
            previous_file_name = prelix + '_' + str(current_num - 1).zfill(2) + '.edf'
            previous_previous_file_name = prelix + '_' + str(current_num - 2).zfill(2) + '.edf'
            if previous_file_name in select_files:
                select_file_final.append(previous_file_name)
            if previous_previous_file_name in select_files:
                select_file_final.append(previous_previous_file_name)
    select_file_final.extend(
        ['chb19_07.edf', 'chb19_10.edf', 'chb19_15.edf'])
    select_file_final.sort()
    files = pd.Series(select_file_final, name='file')
    patient_name = "chb" + str(patient_id).zfill(2)
    files = files[files.str.startswith(patient_name)].drop_duplicates()
    return files[:8]


def FDR_for_one_ictal_file(model, file_path, threshold=0.5, threshold_count=7):
    raw_T = ut.PreprocessTool(file_path).do_preprocess(truncate_time=True)
    duration = raw_T.raw.n_times / raw_T.raw.info['sfreq']
    _data_stft = raw_T.create_group_slicing_event(group_interval=20, num_events_per_group=10,
                                                  duration=5).cut_epochs().get_epochs_stft()
    data_ds = TensorDataset(torch.Tensor(_data_stft))
    del _data_stft
    data_loader = DataLoader(data_ds, batch_size=10)
    alarm_count = 0

    for i, data in enumerate(data_loader):
        data = data[0]
        y_pred = model(data).view(-1)
        if sum(1 for y in y_pred if y > threshold) >= threshold_count:
            alarm_count += 1

    FDR_per_second = alarm_count / duration
    return FDR_per_second


def FDR_for_patient_ictal_file(chb_root_path, constrain_path, model, patient_id, **kwargs):
    print("Start evaluate the FDR for algorithm")
    model.eval()
    files_plan = get_interictal_files_name(chb_root_path=chb_root_path, constrain_path=constrain_path,
                                           patient_id=patient_id)
    patient_name = "chb" + str(patient_id).zfill(2)
    FDR_per_seconds = []
    for ictal_file_name in files_plan:
        file_path = os.path.join(chb_root_path, "{}/{}".format(patient_name, ictal_file_name))
        one_FDR = FDR_for_one_ictal_file(model, file_path, **kwargs)
        FDR_per_seconds.append(one_FDR)
    model.train()
    return np.mean(FDR_per_seconds) * 3600

if __name__ == "__main__":
    pass
