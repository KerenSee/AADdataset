import torch
import torch.nn.functional as F
import numpy as np
import plot_metrics
from data_processing import Data_Processing
import matplotlib.pyplot as plt
from train_eegnet import Train_Test
import os
from data_label import Data_Label
from create_label import Create_Label
from split_T import Split_Trial
from LOC import Location
from sklearn.model_selection import KFold
import openpyxl


def main_process():
    for sub_number in range(16, 31):
        for epoch_number in range(0,5):
            plt.close('all')
            root = 'D:/WorkSpace/'
            all_subname = ['Input sub name(number)']
            subname = all_subname[sub_number]
            original_path = root + subname + '/data/'
            original_file_path = original_path + 'pre.set'
            trial_path = original_path + '/trials/'
            record_path = root + subname + '/record/'

            lr_flag, fm_flag = True, False
            second = 0.5
            model_select = 'eegnet'
            if lr_flag:
                ff = 'lr'
            elif fm_flag:
                ff = 'fm'

            result_file = ''

            output_path = original_path + '/ProcessedData_' + str(second) + '/'
            new_data_path = ''
            num_segments = int(69/second)
            segment_size = int(500*second)
            trainstyle = 'shuffle' # shuffle trials
            real_trial_index = [1]  # val+test
            test_index = [1]

            loc_file = 'Loc_NE.loc'
            threshold = 50
            adjacency_matrix = Location.threshold_similarity(Location.compute_similarity(Location.read_loc_file(loc_file)),
                                                             threshold)



    # plt.show()
main_process()
