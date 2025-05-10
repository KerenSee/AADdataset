#Backup
class V:
    ica_data = []
import numpy as np
import  mne
from scipy.io import savemat
from scipy.signal import stft
import matplotlib.pyplot as plt


eeg_data = []
#Desired path for the preprocessed files
file_path = ''
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        # print(line.strip())
        numbers = [int(num) for num in line.strip().split()]
        eeg_data.append(numbers)
start_list = np.where(np.transpose(np.array(eeg_data))[-2,:]==1)
end_list = np.where(np.transpose(np.array(eeg_data))[-2,:]==2)
eeg_data = np.array(eeg_data).transpose()[:32]
n_channels = eeg_data.shape[0]
sfreq = 500
ch_names = ['P8', 'T8', 'CP6', 'FC6', 'F8', 'F4', 'C4', 'P4', 'AF4', 'Fp2', 'Fp1', 'AF3', 'Fz', 'FC2',
                     'Cz', 'CP2', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'Pz', 'CP1', 'FC1', 'P3', 'C3', 'F3', 'F7', 'FC5',
                     'CP5', 'T7', 'P7']
info = mne.create_info(ch_names, sfreq, ch_types='eeg')
info.set_montage('standard_1020')
raw = mne.io.RawArray(eeg_data, info)
raw.set_eeg_reference(ref_channels='average')
raw.notch_filter(freqs=[50, 100, 150])
raw.filter(l_freq=30, h_freq=100, fir_design='firwin')
raw.resample(sfreq=500)
processed_data = raw.get_data()
eeg_data = processed_data
d = []
l = [0,0,1,1,0,1,0,1,0,1,1,0]
label = []
k = 0
for i in range(len(start_list[0])):
    data = np.array(eeg_data)[:,start_list[0][i]:start_list[0][i]+35000]
    if i!=len(start_list[0])-1:
        print(end_list[0][i]-start_list[0][i])
    if i not in [0, 3, 6, 13]:
        label.append([l[k]]*69)
        k+=1
        for j in range(69):
            d.append(data[:,j*500:(j+1)*500])
np.save('sub2.npy', np.array(d))
np.save('sub2_label.npy', np.array(label).flatten())
eeg_data = []
#read
eeg_data = []
file_path = 'easyfile'
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        # print(line.strip())
        numbers = [int(num) for num in line.strip().split()]
        eeg_data.append(numbers)
print(np.array(eeg_data).shape, np.array(eeg_data)[:,0])
np.save('easyfile'+'_baseline.npy', eeg_data)
for i in range(len(start_list[0])):
    data = np.transpose(np.array(eeg_data))[:32,start_list[0][i]:end_list[0][i]]/1000
    savemat('easyfile'+'_baseline_trail_'+str(i+1)+'.mat',{'data':data})
raw = mne.io.read_raw_eeglab('P1'+'_task.set', preload=True)
channel_names = ['C3', 'C4', 'Cz']
raw_selected = raw.pick_channels(channel_names)
f, t, Zxx = stft(raw_selected.get_data()[:,:334*500], fs=500, nperseg=500)
alpha_mask = (f >= 8) & (f <= 12)
# alpha_amplitude = np.abs(Zxx[:,alpha_mask, :])**2
alpha_amplitude = np.real(Zxx[:, alpha_mask, :])**2 + np.imag(Zxx[:, alpha_mask, :])**2
plt.figure(figsize=(12, 6))
plt.pcolormesh(t, f[alpha_mask], alpha_amplitude[1, :, :], shading='gouraud')
plt.colorbar(label='Amplitude')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Alpha Band Power (C4 Channel)')
plt.show()
#roll data
n_channels = eeg_data.shape[0]
sfreq = 500
ch_names = ['P8', 'T8', 'CP6', 'FC6', 'F8', 'F4', 'C4', 'P4', 'AF4', 'Fp2', 'Fp1', 'AF3', 'Fz', 'FC2',
                     'Cz', 'CP2', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'Pz', 'CP1', 'FC1', 'P3', 'C3', 'F3', 'F7', 'FC5',
                     'CP5', 'T7', 'P7']
info = mne.create_info(ch_names, sfreq, ch_types='eeg')
info.set_montage('standard_1020')
raw = mne.io.RawArray(eeg_data, info)
#     mne.export.export_raw('rs_baseline', raw, fmt="edf")
raw.filter(l_freq=4, h_freq=40)
raw.plot(scalings={'eeg': 70})
#ICA
from mne.preprocessing import ICA
from scipy.signal import welch
channels_list = ['P8', 'T8', 'CP6', 'FC6', 'F8', 'F4', 'C4', 'P4', 'AF4', 'Fp2', 'Fp1', 'AF3', 'Fz', 'FC2',
                 'Cz', 'CP2', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'Pz', 'CP1', 'FC1', 'P3', 'C3', 'F3', 'F7', 'FC5',
                 'CP5', 'T7', 'P7']
subname = 'P1'

channels = 32
num_segments = 69
freq = 500
eeg_data = []
#Desired path for the preprocessed files
file_path = ''
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        # print(line.strip())
        numbers = [int(num) for num in line.strip().split()]
        eeg_data.append(numbers[:-5])
print(np.array(eeg_data).shape, np.array(eeg_data)[:,0])
np.save(subname+'_tACS_post_task.npy', eeg_data)
rs_baseline = np.transpose(np.load('P1_task.npy'),axes=(1,0))
task_baseline = np.transpose(np.load(subname+'_tACS_baseline_task.npy'),axes=(1,0))
print(rs_baseline.shape)
def sort_ica_components_by_energy(ica, raw):
    sources = ica.get_sources(raw).get_data()
    v = np.mean(sources**2, axis=1)
    sorted_indices = np.argsort(v)[::-1]
    return sorted_indices, v[sorted_indices]
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
def plot_ic_spectra(ica, raw, sorted_indices,sfreq=500):
    sources = ica.get_sources(raw).get_data()
    n_ics, n_samples = sources.shape
    fig, axes = plt.subplots(n_ics // 4 + 1, 4, figsize=(15, n_ics * 2))
    for i, source in enumerate(sources):
        ax = axes[i // 4, i % 4]
        freqs = fftfreq(n_samples, 1/sfreq)[:n_samples // 2]
        power_spectrum = np.abs(fft(source))[:n_samples // 2] ** 2
        ax.plot(freqs, power_spectrum, color='blue')
        ax.set_title(f"IC {i+1} Spectrum")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        ax.set_xlim(4, 48)
    plt.tight_layout()
    plt.show()
def plot_PSD(ica_sources, idx, i):
    freqs, psd = welch(ica_sources[idx], fs=500, nperseg=250)
    psd_log = 10 * np.log10(psd * 1e6)  # 转换为 μV²/Hz
    mask = freqs<=80
    plt.plot(freqs[mask], psd_log[mask], color='r')
    plt.xlabel('Frequency (Hz)')
    if i%4==0:
        plt.ylabel('Power (10*log10(uV²/Hz))')
    plt.title(f'IC  00{idx} - PS')
def preprocessing(eeg_data):
    sfreq = 500
    ch_names = ['P8', 'T8', 'CP6', 'FC6', 'F8', 'F4', 'C4', 'P4', 'AF4', 'Fp2', 'Fp1', 'AF3', 'Fz', 'FC2',
                         'Cz', 'CP2', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'Pz', 'CP1', 'FC1', 'P3', 'C3', 'F3', 'F7', 'FC5',
                         'CP5', 'T7', 'P7']
    info = mne.create_info(ch_names, sfreq, ch_types='eeg')
    info.set_montage('standard_1020')
    raw = mne.io.RawArray(eeg_data, info)
    raw.filter(l_freq=4, h_freq=40)
    ica = ICA(n_components=32, random_state=97, max_iter=800, method='infomax', fit_params={'extended':True})
    ica.fit(raw)
    data = ica.get_sources(raw).get_data()
    V.ica_data = data
    sorted_indices, sorted_variances = sort_ica_components_by_energy(ica, raw)
    ica.plot_components(picks=sorted_indices)
    ica.plot_sources(raw, picks=sorted_indices)
    # ica.plot_components()
    plt.subplots(8, 4, figsize=(9,16))
    for i in range(len(sorted_indices)):
        plt.subplot(8, 4, i+1)
        plot_PSD(data, sorted_indices[i], i)
        plt.axvline(x=10, color='green', linestyle='--')
    plt.tight_layout()
    plt.show()
    return raw, ica
raw1, ica1 = preprocessing(rs_baseline/1000)
plt.plot(preprocessing(task_baseline))
plt.axis('off')
plt.show()
def processing2(ica,raw, p):
    ica.exclude = [6,11,23]
    raw_clean = ica.apply(raw)
    print(raw_clean.info['ch_names'])
    raw_clean.set_eeg_reference('average', projection=True)
    np.save(p, raw_clean.get_data())
processing2(ica1, raw1, '../tACS/rs_baseline_pre.npy')