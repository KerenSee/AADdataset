import os
import mne
import torch.nn as nn
import numpy as np
import torch
from plot_metrics import plot_brain_grandplot

class Data_Processing(nn.Module):
    def __init__(self, trial_path, output_path, num_segments, segment_size):
        super(Data_Processing, self).__init__()
        self.trial_path = trial_path
        self.output_path = output_path
        self.num_segments = num_segments
        self.segment_size = segment_size

    def data_processing(self):
        file_paths = [os.path.join(self.trial_path, file) for file in os.listdir(self.trial_path) if file.endswith('.set')]
        d = []
        l = []
        for file_path in file_paths:
            # raw = mne.io.read_raw_fif(file_path, preload=True)
            raw = mne.io.read_raw_eeglab(file_path, preload=True)
            data = raw.get_data()
            print('VVVV',data.shape)
            # segment_size = data.shape[1] // self.num_segments
            if not os.path.exists('Fig_grandplot/'):
                os.mkdir('Fig_grandplot/')
            # plot_brain_grandplot(data, 'Fig_grandplot/')
            d.append(data)
            l.append(file_path.split('/')[-1].split('.set')[0])


            for i in range(self.num_segments):
                start_idx = i * self.segment_size
                end_idx = (i + 1) * self.segment_size
                print(file_path)
                output_file_name = f"{os.path.basename(file_path).split('.')[0]}_segment_{i + 1}raw.fif"
                output_file_path = os.path.join(self.output_path, output_file_name)

                info = mne.create_info(ch_names=raw.ch_names, sfreq=raw.info['sfreq'], ch_types='eeg')
                raw_segment = mne.io.RawArray(data[:, start_idx:end_idx], info)

                raw_segment.save(output_file_path, overwrite=True)

        print("Processing completed.")
        return self.num_segments, np.array(d), np.array(l)

    # def data_processing(self):
    #     file_paths = [os.path.join(self.trial_path, file) for file in os.listdir(self.trial_path) if
    #                   file.endswith('.fif')]
    #
    #     for file_path in file_paths:
    #         raw = mne.io.read_raw_fif(file_path, preload=True)
    #         data = raw.get_data()
    #         sfreq = raw.info['sfreq']
    #         segment_duration = 1.0
    #         step_duration = 0.3
    #         segment_size = int(segment_duration * sfreq)
    #         # step_size = int(step_duration * sfreq)
    #         # segment_num = (data.shape[1] - segment_size) // step_size + 1
    #         for i in range(0, data.shape[1] - segment_size + 1, int(step_duration * sfreq)):
    #             start_idx = i
    #             end_idx = i + segment_size
    #             print(file_path)
    #             output_file_name = f"{os.path.basename(file_path).split('.')[0]}_segment_{i + 1}.fif"
    #             output_file_path = os.path.join(self.output_path, output_file_name)
    #
    #             info = mne.create_info(ch_names=raw.ch_names, sfreq=sfreq, ch_types='eeg')
    #             raw_segment = mne.io.RawArray(data[:, start_idx:end_idx], info)
    #
    #             raw_segment.save(output_file_path, overwrite=True)
    #
    #
    #     print("Processing completed.")
