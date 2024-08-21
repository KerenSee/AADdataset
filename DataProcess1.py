import os
import mne
import torch.nn as nn
import numpy as np
import torch

class Data_Processing(nn.Module):
    def __init__(self, PathName, output_directory, num_segments):
        super(Data_Processing, self).__init__()
        self.PathName = PathName
        self.output_directory = output_directory
        self.num_segments = num_segments

    # 统计文件个数
    def CountFiles(self, path):
        count = 0
        for files in os.listdir(path):
            count += 1
        return count

    def Data_processing(self):
        file_paths = [os.path.join(self.PathName, file) for file in os.listdir(self.PathName) if file.endswith('.set')]

        channels_list = ['P8', 'T8', 'CP6', 'FC6', 'F8', 'F4', 'C4', 'P4', 'AF4', 'Fp2', 'Fp1', 'AF3', 'Fz', 'FC2', 'Cz',
                         'CP2', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'Pz', 'CP1', 'FC1', 'P3', 'C3', 'F3', 'F7', 'FC5', 'CP5', 'T7',
                         'P7']

        for file_path in file_paths:
            raw = mne.io.read_raw_eeglab(file_path, eog=channels_list, preload=True)
            data = raw.get_data()
            segment_size = data.shape[1] // self.num_segments

            for i in range(self.num_segments):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size

                output_file_name = f"{os.path.basename(file_path).split('.')[0]}_segment_{i + 1}.fif"
                output_file_path = os.path.join(self.output_directory, output_file_name)

                # 创建新的 Raw 对象
                info = mne.create_info(ch_names=raw.ch_names, sfreq=raw.info['sfreq'], ch_types='eeg')
                raw_segment = mne.io.RawArray(data[:, start_idx:end_idx], info)

                # 保存 Raw 对象到新的文件
                raw_segment.save(output_file_path, overwrite=True)

        print("Processing completed.")

    def read(self):
        dataset_list = []
        label_list = []
        file_paths = [os.path.join(self.output_directory, file) for file in os.listdir(self.output_directory) if file.endswith('.fif')]
        channels_list = ['P8', 'T8', 'CP6', 'FC6', 'F8', 'F4', 'C4', 'P4', 'AF4', 'Fp2', 'Fp1', 'AF3', 'Fz', 'FC2',
                         'Cz','CP2', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'Pz', 'CP1', 'FC1', 'P3', 'C3', 'F3', 'F7', 'FC5',
                         'CP5', 'T7', 'P7']
        self.Data_processing()
        for file_path in file_paths:
            raw = mne.io.read_raw_fif(file_path, preload=True)
            raw.pick_channels(ch_names=channels_list)
            data = raw.get_data()[:32, :]
            dataset_list.append(data)
        for i in file_paths:
            if 'Female' in i:
                label_list.append(0)
            else:
                label_list.append(1)

        return np.array(dataset_list), np.array(label_list)
