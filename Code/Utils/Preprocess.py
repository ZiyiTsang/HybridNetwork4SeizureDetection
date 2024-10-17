import os
import warnings

import mne
import numpy as np
from mne import use_log_level
from mne.io import RawArray


class PreprocessTool():

    def __init__(self, data_path: str):
        self.data_path = data_path
        assert os.path.exists(data_path)
        self.raw = None
        self.events=None
        self.epochs=None
        self.duration=None
        self._once_epoch_flag=False
        self.preprocessed=False

        self.channels_to_pick = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                                 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
                                 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8']

    def do_preprocess(self,truncate_time=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.raw = mne.io.read_raw_edf(self.data_path, preload=True, verbose=False)
            if truncate_time is True:
                self.raw.crop(tmin=0, tmax=int(self.raw.n_times/self.raw.info['sfreq'])-1)

        self.pre_process_EEG()
        self.normalize_raw_per_channel()
        self._preprocessed=True
        return self


    def pre_process_EEG(self):
        if 'T8-P8-1' in self.raw.info['ch_names']:
            self.raw = self.raw.drop_channels(['T8-P8-1'])
            self.raw = self.raw.rename_channels(mapping={'T8-P8-0': 'T8-P8'})
        self.raw = self.raw.pick(picks=self.channels_to_pick, verbose=False)
        self.raw = self.raw.reorder_channels(self.channels_to_pick)
        self.raw = self.raw.filter(l_freq=0.5, h_freq=None,verbose=False)
        self.raw = self.raw.notch_filter(np.array([60,120]), verbose=False)


    def normalize_raw_per_channel(self):
        data = self.raw.get_data()
        mean_per_channel = np.mean(data, axis=1, keepdims=True)
        std_per_channel = np.std(data, axis=1, keepdims=True)
        normalized_data = (data - mean_per_channel) / std_per_channel
        self.raw = mne.io.RawArray(normalized_data, self.raw.info,verbose=False)

    def create_fixed_length_events(self, start, stop, overlap):
        self.events = mne.make_fixed_length_events(self.raw, start=start, stop=stop, duration=self.duration,overlap=overlap)
        return self

    def create_group_slicing_event(self,group_interval=10,num_events_per_group=10,duration=5):
        self.duration = duration
        sfreq = self.raw.info['sfreq']  # 采样频率
        events = []
        for group_start in np.arange(0, self.raw.times[-1], group_interval):
            for event_idx in range(num_events_per_group):
                event_time = group_start + event_idx/sfreq
                event_sample = int(event_time * sfreq)
                events.append([event_sample, 0, 1])  # [sample, 0, event_id]
        self.events = np.array(events)
        return self

    def no_overlap_events(self, start, stop,duration=5):
        self.duration=duration
        self.create_fixed_length_events(start=start, stop=stop,overlap=0)
        return self

    def overlap_events(self,start, stop,overlap,duration=5):
        self.duration = duration
        self.create_fixed_length_events(start=start, stop=stop,overlap=overlap)
        return self
    def overlap_events_slice_all(self,start, stop,duration=5):
        self.duration = duration
        sampling_rate = self.raw.info['sfreq']
        self.create_fixed_length_events(start=start, stop=stop, overlap=duration-1/sampling_rate)
        return self


    def cut_epochs(self):
        assert self.events is not None, "Events are not created,run function create_fixed_length_events or no_overlap_events or overlap_events"
        with use_log_level('WARNING'):
            self.epochs = mne.Epochs(self.raw, self.events, tmin=0, tmax=self.duration, baseline=None, verbose=False)
        return self

    def get_epochs(self):
        if self._once_epoch_flag:
            warnings.warn("Epochs are already created before, if you want to get new epochs please run function cut_epochs.")
        self._once_epoch_flag=True
        assert self.epochs is not None, "Epochs are not created,run function cut_epochs"
        return self.epochs

    def get_epochs_stft(self):
        if self._once_epoch_flag:
            warnings.warn("Epochs are already created before, if you want to get new epochs please run function cut_epochs.")
        self._once_epoch_flag = True
        assert self.epochs is not None, "Epochs are not created,run function cut_epochs"
        time_data = self.epochs.get_data()
        stft_data = []
        for sample in time_data:
            frequency_data = mne.time_frequency.stft(sample, wsize=128, verbose=False)
            stft_data.append(frequency_data)
        return np.abs(np.stack(stft_data, axis=0))

    def clear(self):
        self.events=None
        self.epochs=None
        self._once_epoch_flag=False
