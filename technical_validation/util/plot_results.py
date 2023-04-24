import mne
# import seaborn as sns
import pickle
import glob
import json
from collection import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats

# generate plots from all the different results and plsave them in the figures folder

# load the results
results_folder ="/esat/spchtemp/scratch/lbollens/experiments/march_2023/dataset_paper/experiments"
plot_dilation = True
plot_linear_backward = True
plot_linear_forward = True
plot_vlaai = True

if plot_dilation:
    # dilation model, match mismatch results
    # plot boxplot of the results per window length
    # load evaluation results for all window lengths

    files = glob.glob(os.path.join(results_folder, "results_dilated_convolutional_model/eval_*.json"))
    # sort the files
    files.sort()

    # create dict to save all results per sub
    results = []
    windows = []
    for f in files:
        # load the results
        with open(f, "rb") as ff:
            res = json.load(ff)
        #loop over res and get accuracy in a list
        acc = []
        for sub, sub_res in res.items():

            if 'acc' in sub_res:
                acc.append(sub_res['acc']*100)

        results.append(acc)

        # get the window length
        windows.append(int(f.split("_")[-1].split(".")[0]))

    # sort windows and results according to windows
    windows, results = zip(*sorted(zip(windows, results)))
    # convert windows to seconds
    windows = ['%d' %int(w/64) for w in windows]

    #boxplot of the results
    plt.boxplot(results, labels=windows)
    plt.xlabel("Window length (s)")
    plt.ylabel("Accuracy (%)")
    # plt.title("Accuracy of dilation model, per window length")
    plt.savefig(os.path.join(results_folder,'figures', "boxplot_dilated_conv.pdf"))

if plot_linear_backward:
    ## linear backward regression plots
    # load the results
    results_files = glob.glob(os.path.join(results_folder,"results_linear_backward/eval*.json"))

    # load all the files per frequency band
    results = {}
    for f in results_files:
        # get the frequency band
        band = f.split("_")[-2:]
        band = "_".join(band)
        band = band.split(".j")[0]
        # load the results
        with open(f, "rb") as ff:
            res = json.load(ff)
        #loop over res and get accuracy in a list
        acc = []
        for sub, sub_res in res.items():
            if 'pearson_metric_cut' in sub_res:
                acc.append(sub_res['pearson_metric_cut'])
        results[band] = acc

    bands = list(results.keys())
    bands= sorted(bands, key=lambda x: sum([float(x) for x in x.split("_")]))
    mapping_bands_to_label_names = defaultdict(lambda x:x, {
        '0.5_4.0': 'Delta [0.5-4]',
        '4.0_8.0': 'Theta [4-8]',
        '4_8': 'Theta [4-8]',
        '8_14': 'Alpha [8-14]',
        '8.0_14.0': 'Alpha [8-14]',
        '14.0_30.0': 'Beta [14-30]',
        '0.5_None': 'Broad [0.5-32]',
        '0.5_31.0': 'Broad [0.5-31]',
    })

    # plot the results
    plt.figure()
    plt.boxplot([results[band] for band in bands], labels=[mapping_bands_to_label_names[band] for band in bands])
    plt.xlabel("Frequency band [Hz]")
    plt.ylabel("Correlation")

    plt.savefig(os.path.join(results_folder,'figures',"boxplot_decoder_frequency_bans.pdf" ))

if plot_linear_forward:
    ## linear forward regression plots/ topoplot
    # load the results
    results_files=glob.glob(os.path.join(results_folder,"results_linear_forward/eval*.json"))
    # load results pickle file
    results = {}

    for f in results_files:
        # get the frequency band
        band = f.split("_")[-2:]
        band = "_".join(band)
        band = band.split(".j")[0]
        # load the results
        with open(f, "rb") as ff:
            res = json.load(ff)
        #loop over res and get accuracy in a list
        acc = list(res.values())

        results[band] = acc


    # plot the mean results, over channels
    bands = list(results.keys())
    bands = sorted(bands, key=lambda x: sum([float(x) for x in x.split("_")]))
    mapping_bands_to_label_names = defaultdict(lambda x: x, {
        '0.5_4.0': 'Delta [0.5-4]',
        '4.0_8.0': 'Theta [4-8]',
        '4_8': 'Theta [4-8]',
        '8_14': 'Alpha [8-14]',
        '8.0_14.0': 'Alpha [8-14]',
        '14.0_30.0': 'Beta [14-30]',
        '0.5_None': 'Broad [0.5-32]',
        '0.5_31.0': 'Broad [0.5-31]',
    })

    # plot the results
    plt.figure()
    plt.boxplot([[np.mean(x) for x in results[band]] for band in bands], labels=[mapping_bands_to_label_names[band] for band in bands])
    plt.xlabel("Frequency band [Hz]")
    plt.title(f"Mean correlation per channel frequency band")
    plt.ylabel("Correlation")
    plt.savefig(os.path.join(results_folder,'figures',"boxplot_linear_forward_frequency_bands.pdf" ))

    for band in bands:
        # plot the topoplot
        mean_per_channel = np.mean(np.stack(results[band]), axis=0)
        montage = mne.channels.make_standard_montage('biosemi64')
        sfreq = 64
        info = mne.create_info(ch_names=montage.ch_names, sfreq=sfreq, ch_types='eeg').set_montage(montage)

        fig = plt.figure()
        axes = plt.gca()
        max_coef = np.max(np.abs(mean_per_channel))
        min_coef = np.min(np.abs(mean_per_channel))
        im , cn = mne.viz.plot_topomap(mean_per_channel, pos=info, axes=axes ,show=False, cmap='jet', vlim=(min_coef,max_coef))
        mne.viz.tight_layout()

        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])


        plt.colorbar(im, cax=cbar_ax, label='Spearman corr')
        # plt.show()
        plt.savefig(os.path.join(results_folder, 'figures',f"topoplot_linear_forward_{band}.pdf"))

if plot_vlaai:

    ## vlaai results
    # load the results
    results_files = glob.glob(os.path.join(results_folder,"results_vlaai/eval*.json"))

    results = {}
    for f in results_files:

        with open(f, "rb") as ff:
            res = json.load(ff)
        #loop over res and get accuracy in a list
        acc = []
        subs = []
        for sub, sub_res in res.items():
            if 'pearson_metric' in sub_res:
                acc.append(sub_res['pearson_metric'])
                subs.append(sub)

        results = acc

    # plot the results
    plt.figure()
    plt.boxplot(results, labels=['vlaai'])
    plt.xlabel("Model")
    plt.ylabel("Correlation")
    # plt.title("Correlation, per subject")
    plt.savefig(os.path.join(results_folder, 'figures',"boxplot_vlaai.pdf" ))


