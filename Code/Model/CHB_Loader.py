import re

import numpy as np
import os
import torch
import multiprocessing
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

project_path = os.path.abspath(os.path.relpath('../../../', os.getcwd()))
import lightning as L
from torch.utils.data import DataLoader, Dataset, TensorDataset
import sys

sys.path.append('../')



class CHBDependentDETTECTION(L.LightningDataModule):

    def __init__(self, data_dir: str, patient_id: int, leave_out_id: int, batch_size: int = 32):
        super().__init__()

        assert data_dir is not None
        assert patient_id is not None
        assert leave_out_id is not None

        self.batch_size = batch_size
        self.data_dir = data_dir
        self.patient_id = patient_id
        self.leave_out_id = leave_out_id
        file_plan = self.get_files_plan()
        file_plan_fit, file_plan_leave_out = file_plan['fit'], file_plan['leave_out']

        def load_and_concatenate(_file_plan):
            if type(_file_plan)==str:
                data = np.load(os.path.join(data_dir, _file_plan))
                return data['X'],data['y']
            X_list = []
            y_list = []
            for f in _file_plan:
                data = np.load(os.path.join(data_dir, f))
                X_list.append(data['X'])
                y_list.append(data['y'])

            X_concat = np.concatenate(X_list, axis=0)
            y_concat = np.concatenate(y_list, axis=0)

            return X_concat, y_concat

        X_fit,y_fit=load_and_concatenate(_file_plan=file_plan_fit)
        X_leaveout, y_leaveout = load_and_concatenate(_file_plan=file_plan_leave_out)

        # X_leaveout = np.load(os.path.join(data_dir, file_plan_leave_out))['X']
        # y_leaveout = np.load(os.path.join(data_dir, file_plan_leave_out))['y']

        print('Train Count: Negative {}, Positive {} Medium {}'.format(len(X_fit[y_fit == 0]), len(X_fit[y_fit == 1]),
                                                                       len(X_fit[(y_fit != 0) & (y_fit != 1)])))

        print('Leaveout Count: Negative {}, Positive {} Medium {}'.format(len(X_leaveout[y_leaveout == 0]), len(X_leaveout[y_leaveout == 1]),
                                                                       len(X_leaveout[(y_leaveout != 0) & (y_leaveout != 1)])))

        X_train, X_valid, y_train, y_valid = train_test_split(X_fit, y_fit, test_size=0.1, shuffle=True,
                                                              random_state=42)

        valid_indices = (y_valid == 0) | (y_valid == 1)
        X_valid, y_valid = X_valid[valid_indices], y_valid[valid_indices]

        X_leaveout,y_leaveout=shuffle(X_leaveout,y_leaveout,random_state=42)



        self.ds_train = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32))
        self.ds_valid = TensorDataset(torch.tensor(X_valid, dtype=torch.float32),
                                      torch.tensor(y_valid, dtype=torch.float32))
        self.ds_leaveout = TensorDataset(torch.tensor(X_leaveout, dtype=torch.float32),
                                         torch.tensor(y_leaveout, dtype=torch.float32))

    def get_files_plan(self):
        files = os.listdir(self.data_dir)
        prelix = 'chb' + str(self.patient_id).zfill(2)
        files_filter = [f for f in files if re.match(prelix, f)]
        files_normal = [f for f in files_filter if re.match('.*normal.*', f)]
        files_cross = [f for f in files_filter if re.match('.*cross.*', f)]
        file_interictal=[f for f in files_filter if re.match('.*interictal.*', f)]
        leave_outs = []
        file_trains = []
        assert self.leave_out_id < len(files_normal)
        for i in range(len(files_normal)):
            file_trains.append(files_normal[0:i] + files_normal[i + 1:] + files_cross[0:i] + files_cross[i + 1:]+file_interictal)
            leave_outs.append(files_normal[i])
        file_plan={'leave_out': leave_outs[self.leave_out_id], 'fit': file_trains[self.leave_out_id]}
        print(file_plan)
        return file_plan

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=8,
                          persistent_workers=True, prefetch_factor=5)

    def val_dataloader(self):
        return DataLoader(self.ds_valid, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=8,
                          persistent_workers=True, prefetch_factor=5)

    def test_dataloader(self):
        return DataLoader(self.ds_leaveout, batch_size=196, shuffle=False, pin_memory=True, num_workers=8,
                          persistent_workers=True, prefetch_factor=5)

    def predict_dataloader(self):
        return None


def main():
    CHBDependentDETTECTION(data_dir='E:\Research\BilinearNetwork\Data\PreprocessedData\CHB-MIT\detection', patient_id=1, leave_out_id=5, batch_size=32)

if __name__ == '__main__':
    main()

#
# class CHBDependentDMT(L.LightningDataModule):
#
#     def __init__(self, data_dir: str, patient_id: int, leave_out_id: int, batch_size: int = 32):
#         super().__init__()
#
#         assert data_dir is not None
#         assert patient_id is not None
#         assert leave_out_id is not None
#
#         self.batch_size = batch_size
#         self.data_dir = data_dir
#         self.patient_id = patient_id
#         self.leave_out_id = leave_out_id
#
#         file_plan = self.get_files_plan(data_dir, patient_id, leave_out_id)
#         interictal_name, preictal_list, leave_out_name = file_plan['interictal'], file_plan['preictal'], file_plan[
#             'leave_out']
#
#         # interictal_data = batch_zscore_normalize(np.load(os.path.join(data_dir, interictal_name)))
#         # preictal_data_list = [np.load(os.path.join(data_dir, f)) for f in preictal_list]
#         # preictal_data = batch_zscore_normalize(np.concatenate(preictal_data_list))
#         # leave_out_val_data = batch_zscore_normalize(np.load(os.path.join(data_dir, leave_out_name)))
#
#         interictal_data = np.load(os.path.join(data_dir, interictal_name))
#         preictal_data_list = [np.load(os.path.join(data_dir, f)) for f in preictal_list]
#         preictal_data = np.concatenate(preictal_data_list)
#         leave_out_val_data = np.load(os.path.join(data_dir, leave_out_name))
#
#         print("preictal_name:", preictal_list)
#         print("leave_out_name:", leave_out_name)
#
#         np.random.seed(42)
#         np.random.shuffle(interictal_data)
#
#         interictal_data_train, interictal_data_val = interictal_data[
#                                                      :len(interictal_data) - len(leave_out_val_data)], interictal_data[
#                                                                                                        len(interictal_data) - len(
#                                                                                                            leave_out_val_data):]
#
#         label_test = np.concatenate([np.zeros(len(interictal_data_val)), np.ones(len(leave_out_val_data))])
#         data_test = np.concatenate([interictal_data_val, leave_out_val_data])
#         indic = np.random.permutation(len(data_test))
#         data_test = data_test[indic]
#         label_test = label_test[indic]
#
#         label_fit = np.concatenate([np.zeros(len(interictal_data_train)), np.ones(len(preictal_data))])
#         data_fit = np.concatenate([interictal_data_train, preictal_data])
#         X_train, X_valid, y_train, y_valid = train_test_split(data_fit, label_fit, test_size=0.1, shuffle=True,
#                                                               random_state=42)
#
#         print('Train Count: Negative {}, Positive {}'.format(len(X_train[y_train == 0]), len(X_train[y_train == 1])))
#         print(
#             'Validation Count: Negative {}, Positive {}'.format(len(X_valid[y_valid == 0]), len(X_valid[y_valid == 1])))
#         print('Test Count: Negative {}, Positive {}'.format(len(data_test[label_test == 0]),
#                                                             len(data_test[label_test == 1])))
#
#         # __sampler`
#         class_counts = torch.bincount(torch.tensor(y_train, dtype=torch.int32))
#         class_weights = 1.0 / class_counts.float()
#         weights = class_weights[y_train]
#         self.sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(y_train), replacement=True)
#
#         self.trainset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
#                                       torch.tensor(y_train, dtype=torch.float32))
#         self.valset = TensorDataset(torch.tensor(X_valid, dtype=torch.float32),
#                                     torch.tensor(y_valid, dtype=torch.float32))
#         self.testset = TensorDataset(torch.tensor(data_test, dtype=torch.float32),
#                                      torch.tensor(label_test, dtype=torch.int32))
#
#         del interictal_data, preictal_data, leave_out_val_data, interictal_data_train, interictal_data_val, label_test, data_test, label_fit, data_fit, X_train, X_valid, y_train, y_valid
#
#     def get_files_plan(self, data_dir, patient_code, leave_one_code=None):
#         files = os.listdir(data_dir)
#         prelix = 'chb' + str(patient_code).zfill(2)
#         files_filter = [f for f in files if re.match(prelix, f)]
#         files_preictal = [f for f in files_filter if re.match('.*preictal.*', f)]
#         files_interictal = [f for f in files_filter if re.match('.*interictal.*', f)]
#         files_preictal_post = []
#         leave_outs = []
#         for i in range(len(files_preictal)):
#             files_preictal_post.append(files_preictal[0:i] + files_preictal[i + 1:])
#             leave_outs.append(files_preictal[i])
#         assert len(files_interictal) == 1
#         return {'interictal': files_interictal[0], 'preictal': files_preictal_post[leave_one_code],
#                 'leave_out': leave_outs[leave_one_code]}
#
#     def prepare_data(self):
#         pass
#
#     def setup(self, stage: str):
#         pass
#
#     def train_dataloader(self):
#         return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4,
#                           persistent_workers=True, prefetch_factor=5, sampler=self.sampler)
#
#     def val_dataloader(self):
#         return DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4,
#                           persistent_workers=True, prefetch_factor=5)
#
#     def test_dataloader(self):
#         return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4,
#                           persistent_workers=True, prefetch_factor=5)
#
#     def predict_dataloader(self):
#         return None
#
#
# class CHBDependentDM(L.LightningDataModule):
#     def __init__(self, data_dir: str, patient_id: int, batch_size: int = 32):
#         super().__init__()
#         self.trainset = None
#         self.valset = None
#         self.testset = None
#         self.batch_size = batch_size
#         self.data_dir = data_dir
#         self.patient_id = patient_id
#         self.save_hyperparameters()
#
#         interictal_path = os.path.join(self.data_dir,
#                                        os.path.join('chb' + str(self.patient_id).zfill(2) + "_interictal.npz"))
#         preictal_path = os.path.join(self.data_dir,
#                                      os.path.join('chb' + str(self.patient_id).zfill(2) + "_preictal.npz"))
#
#         self.interictal = np.load(interictal_path)['data']
#         self.preictal = np.load(preictal_path)['data']
#
#         print("Sample of inter:" + str(self.interictal.shape[0]))
#         print("Sample of pre:" + str(self.preictal.shape[0]))
#
#         X = np.concatenate(
#             (self.interictal[:int(0.8 * len(self.interictal))], self.preictal[:int(0.8 * len(self.preictal))]))
#         y = np.concatenate((np.zeros(int(0.8 * len(self.interictal))), np.ones(int(0.8 * len(self.preictal)))))
#
#         self.trainset = torch.utils.data.TensorDataset(torch.Tensor(X.astype(np.float32)),
#                                                        torch.Tensor(y.astype(np.float32)))
#         # --
#         X = np.concatenate(
#             (self.interictal[int(0.8 * len(self.interictal)):], self.preictal[int(0.8 * len(self.preictal)):]))
#         y = np.concatenate(((np.zeros(len(self.interictal) - int(0.8 * len(self.interictal)))),
#                             (np.ones(len(self.preictal) - int(0.8 * len(self.preictal))))))
#
#         self.valset = torch.utils.data.TensorDataset(torch.Tensor(X.astype(np.float32)),
#                                                      torch.Tensor(y.astype(np.float32)))
#         del X, y
#
#     def train_dataloader(self):
#         return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=16, pin_memory=True,
#                           persistent_workers=True, prefetch_factor=5)
#
#     def val_dataloader(self):
#         return DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True,
#                           persistent_workers=True, prefetch_factor=5)
#
#     def test_dataloader(self):
#         return DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True,
#                           persistent_workers=True, prefetch_factor=5)
#
#     def predict_dataloader(self):
#         return None
#
#
# class CHBindependant_train(Dataset):
#
#     def __init__(self, data_dir, num_leave_out=500):
#         self.data_dir = data_dir
#         self.file_list = os.listdir(data_dir)
#         self.file_list.sort()
#         self.num_samples_per_file = None
#         self.leave_out = num_leave_out
#
#         first_file = np.load(os.path.join(data_dir, self.file_list[0]))['data']
#         self.num_samples_per_file = len(first_file)
#         del first_file
#         last_file = np.load(os.path.join(data_dir, self.file_list[-1]))['data']
#         num_samples_last_file = len(last_file)
#         del last_file
#         self.total_num_samples = int(((len(self.file_list) - 1) * (self.num_samples_per_file)) + num_samples_last_file)
#
#         self.lock = multiprocessing.Lock()
#         self.current_file_idx = None
#         self.current_data = None
#         self.current_label = None
#
#     def __len__(self):
#         return self.total_num_samples
#
#     def __getitem__(self, idx):
#         file_idx = idx // self.num_samples_per_file
#         sample_idx = idx % self.num_samples_per_file
#
#         if sample_idx >= self.num_samples_per_file:
#             raise ValueError('sample_idx out of range')
#         elif sample_idx >= self.leave_out:
#             sample_idx -= self.leave_out
#
#         if self.current_file_idx != file_idx:
#             self.current_file_idx = file_idx
#             self.current_file = np.load(os.path.join(self.data_dir, self.file_list[file_idx]), mmap_mode='r',
#                                         allow_pickle=False)
#             self.current_data = self.current_file['data']
#             self.current_label = self.current_file['label']
#
#         data = self.current_data[sample_idx]
#         label = self.current_label[sample_idx]
#         return data, label
#
#
# class CHBindependant_valid(Dataset):
#     def __init__(self, data_dir):
#         file = np.load(os.path.join(data_dir, "valid_data.npz"))
#         self.data = file['data']
#         self.label = file['label']
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx], self.label[idx]
#
#
# class CHBIndependentDM(L.LightningDataModule):
#     def __init__(self, root_dir: str, batch_size: int = 32):
#         super().__init__()
#
#         self.trainset = CHBindependant_train(os.path.join(root_dir, 'Train'))
#
#         self.valset = CHBindependant_valid(os.path.join(root_dir, 'Valid'))
#
#         self.testset = CHBindependant_train(os.path.join(root_dir, 'Test'))
#         self.batch_size = batch_size
#         self.save_hyperparameters()
#
#     def prepare_data(self):
#         pass
#
#     def setup(self, stage: str):
#         pass
#
#     def train_dataloader(self):
#         return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=1)
#
#     def val_dataloader(self):
#         return DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=8)
#
#     def test_dataloader(self):
#         return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=1)
#
#     def predict_dataloader(self):
#         return None
#
#
# class SanityCheckDataset(Dataset):
#     def __init__(self, data_dir):
#         file = np.load(os.path.join(data_dir, "valid_data.npz"))
#         self.data = file['data']
#         self.label = file['label']
#         #   将label所有的值都改为0，除了最后一个
#         self.label = np.zeros_like(self.label)
#         self.label[-1] = 1.0
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx], self.label[idx]
#
#
# class SanityCheckDM(L.LightningDataModule):
#     def __init__(self, root_dir: str, batch_size: int = 32):
#         super().__init__()
#
#         self.trainset = SanityCheckDataset(os.path.join(root_dir, 'Valid'))
#         self.batch_size = batch_size
#         self.save_hyperparameters()
#
#     def prepare_data(self):
#         pass
#
#     def setup(self, stage: str):
#         pass
#
#     def train_dataloader(self):
#         return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=1)
#
#     def val_dataloader(self):
#         return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=8)
#
#     def test_dataloader(self):
#         return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=1)
#
#     def predict_dataloader(self):
#         return None
