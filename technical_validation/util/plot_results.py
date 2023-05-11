# import seaborn as sns
import glob
import json
import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy.stats

# generate plots from all the different results and plsave them in the figures folder

# load the results
base_results_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments')
os.makedirs(os.path.join(base_results_folder, 'figures'), exist_ok=True)
plot_dilation = False
plot_linear_backward = True
plot_linear_forward = False
plot_vlaai = False

freq_bands = {
    'Delta [0.5-4]': (0.5, 4.0),
    'Theta [4-8]': (4, 8.0),
    'Alpha [8-14]': (8, 14.0),
    'Beta [14-30]': (14, 30.0),
    'Broadband [0.5-32]': (0.5, 32.0),
}

if plot_dilation:
    # dilation model, match mismatch results
    # plot boxplot of the results per window length
    # load evaluation results for all window lengths

    files = glob.glob(os.path.join(base_results_folder, "results_dilated_convolutional_model/eval_*.json"))
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
    plt.savefig(os.path.join(base_results_folder, 'figures', "boxplot_dilated_conv.pdf"))

if plot_linear_backward:
    ## linear backward regression plots
    # load the results
    results_files_glob = os.path.join(base_results_folder, "results_linear_backward", "eval*.json")
    results_files = glob.glob(results_files_glob)
    all_results = []

    for result_file in results_files:
        with open(result_file, "r") as f:
            data = json.load(f)
        for index, stim in enumerate(data['stim_filename']):
            percentile = np.percentile(data['null_distr'][index], 95)
            score = data['score'][index]
            highpass = data['highpass'] if data['highpass'] is not None else 0.5
            lowpass = data['lowpass'] if data['lowpass'] is not None else 32.0

            all_results.append([data['subject'], stim, score, highpass, lowpass, percentile, score > percentile])

    df = pd.DataFrame(all_results, columns=['subject', 'stim', 'score', 'highpass', 'lowpass', 'percentile', 'significant'])

    print('Confirming that we found neural tracking for each subject')
    nb_subjects = len(pd.unique(df['subject']))
    nb_significant = df.groupby('subject').agg({'significant': 'any'}).sum()
    print('Found {} subjects, {} of which had at least one significant result'.format(nb_subjects, nb_significant))


    subject_stories_group = df.groupby(['subject', 'stim']).agg({'significant': 'any'})
    nb_recordings = len(subject_stories_group)
    nb_significant_recordings = subject_stories_group.sum()
    print("Found {} recordings, {} of which were significant".format(nb_recordings, nb_significant_recordings))
    non_significant_recordings_series = df.groupby(['subject', 'stim']).agg({'significant': 'any'})['significant'] == False
    non_significant_recordings = non_significant_recordings_series[non_significant_recordings_series].index

    # Table of non-significant results
    print("Non-significant results:")
    for subject, stimulus in non_significant_recordings:
        print("{} & {} \\\\".format(subject, stimulus))

    # plot the results
    ## General frequency plot
    values_for_boxplot = []
    names_for_boxplot = []
    for band_name, (highpass, lowpass) in freq_bands.items():

        selected_df = df[(df['highpass'] == highpass) & (df['lowpass'] == lowpass)]
        values_for_boxplot.append(selected_df['score'].tolist())
    plt.figure(figsize=(8, 5))
    plt.boxplot(values_for_boxplot, labels=freq_bands.keys())
    plt.ylabel('Pearson correlation')
    plt.xlabel('Frequency band')
    #plt.grid(True)
    plt.title('Linear decoder performance across frequency bands')
    # plt.show()
    plt.savefig(os.path.join(base_results_folder, 'figures', "boxplot_linear_backward_frequency.pdf"))
    #plt.close()

    ## plot the results per stimulus
    values_for_boxplot = []
    names_for_boxplot = []
    def sort_key_fn(x):
        split = x.split('_')
        number = ord(split[0][0])*1e6+ int(split[1])*1e3
        # artifact audiobook
        if len(split) > 2 and split[2].isdigit():
            number += int(split[2])
        return number



    for stimulus in sorted(pd.unique(df['stim']), key=sort_key_fn):
        selected_df = df[(df['stim'] == stimulus) &(df['highpass'] == 0.5) & (df['lowpass'] == 4.0)]
        values_for_boxplot.append(selected_df['score'].tolist())
        names_for_boxplot.append(stimulus + ' ({})'.format(len(selected_df)))
    plt.figure(figsize=(11,6))
    plt.boxplot(values_for_boxplot, labels=names_for_boxplot)
    plt.ylabel('Pearson correlation')
    plt.xlabel('Stimulus (Number of recordings)')
    plt.xticks(rotation=90)

    # plt.grid(True)
    plt.title('Linear decoder performance across stimuli')
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(base_results_folder, 'figures', "boxplot_linear_backward_stimuli.pdf"))
    # plt.close()


    ## plot the results per stimulus type
    values_for_boxplot = []
    names_for_boxplot = []
    stim_type_selectors = {
        'Audiobook': df['stim'].str.startswith('audiobook_'),
        'Podcast': df['stim'].str.startswith('podcast_'),
    }
    test_data  = []
    for stimulus_type, stimulus_selector in stim_type_selectors.items():
        selected_df = df[(stimulus_selector) &(df['highpass'] == 0.5) & (df['lowpass'] == 4.0) & (~(df['stim'].str.endswith('artefact'))) & (~(df['stim'].str.endswith('shifted'))) & (~(df['stim'].str.endswith('audiobook_1_1')))& (~(df['stim'].str.endswith('audiobook_1_2')))]
        values_for_boxplot.append(selected_df['score'].tolist())
        names_for_boxplot.append(stimulus_type + ' ({})'.format(len(selected_df)))
        test_data += [selected_df['score'].tolist()]

    print("MannWhitneyU test: {}, medians: {}, {}".format(scipy.stats.mannwhitneyu(test_data[0], test_data[1]),np.median(test_data[0]), np.median(test_data[1])))
    plt.figure()
    plt.boxplot(values_for_boxplot, labels=names_for_boxplot)
    plt.ylabel('Pearson correlation')
    plt.xlabel('Stimulus type (Number of recordings)')
    plt.xticks()

    #plt.grid(True)
    plt.title('Linear decoder performance across stimuli types')
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(base_results_folder, 'figures', "boxplot_linear_backward_stimulus_type.pdf"))
    # plt.close()


if plot_linear_forward:
    results_files_glob = os.path.join(base_results_folder, "results_linear_forward",
                                      "eval*.json")
    results_files = glob.glob(results_files_glob)
    all_results = []

    for result_file in results_files:
        with open(result_file, "r") as f:
            data = json.load(f)
        for index, stim in enumerate(data['stim_filename']):
            percentile = np.percentile(data['null_distr'][index], 95)
            score = data['score'][index]
            null_distr = data['null_distr'][index]
            highpass = data['highpass'] if data['highpass'] is not None else 0.5
            lowpass = data['lowpass'] if data['lowpass'] is not None else 32.0

            all_results.append([data['subject'], stim, score, null_distr, highpass, lowpass, percentile, score > percentile])

    df = pd.DataFrame(all_results, columns=['subject', 'stim', 'scores_per_channel', 'null_distr', 'highpass', 'lowpass', 'percentile', 'significant'])

    montage = mne.channels.make_standard_montage('biosemi64')
    sfreq = 64
    info = mne.create_info(ch_names=montage.ch_names, sfreq=sfreq, ch_types='eeg').set_montage(montage)

    fig, axes = plt.subplots(3, 5, figsize=(16, 12))
    temp_df = df.copy()
    temp_df['filterband'] = temp_df['highpass'].astype(str) + '-' + temp_df['lowpass'].astype(str)
    scores = temp_df.groupby(['filterband']).agg({'scores_per_channel': lambda x: list(x)})
    all_scores = np.mean(scores['scores_per_channel'].tolist(), axis=1)
    max_coef = np.max(all_scores)
    min_coef = np.min(all_scores)

    stim_type_selectors = {
        'All Stimuli': df['stim'].str.startswith(''),
        'Audiobook': df['stim'].str.startswith('audiobook_'),
        'Podcast': df['stim'].str.startswith('podcast_'),

    }
    for index, (stim_type, selector) in enumerate(stim_type_selectors.items()):

        axes[index][0].set_ylabel(stim_type)

        for index2, (band_name, (highpass, lowpass)) in enumerate(freq_bands.items()):
            ax = axes[index][index2]
            selected_df = df[(df['highpass'] == highpass) & (df['lowpass'] == lowpass) & (selector)]
            mean_scores = np.mean(selected_df['scores_per_channel'].tolist(), axis=0)
            percentile = np.percentile(np.concatenate(selected_df['null_distr'].tolist(), axis=0), 95, axis=0)
            # plot the topoplot
            im , cn = mne.viz.plot_topomap(mean_scores, pos=info, axes=ax ,show=False, cmap='Reds', vlim=(min_coef,max_coef))
            mne.viz.tight_layout()
            ax.set_title(f"{band_name} Hz")

            # cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])



    #plt.colorbar(im, cax=axes[5], label='Pearson correlation') #cax=cbar_ax,
    fig.suptitle('Forward model performance across stimuli types and frequency bands',y=0.94, fontsize=18)
    fig.tight_layout()

    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
    fig.colorbar(im, cax=cbar_ax, label='Pearson correlation')
    plt.savefig(os.path.join(base_results_folder, 'figures', f"topoplot_linear_forward.pdf"))
    #plt.show()
    plt.close(fig)






if plot_vlaai:

    ## vlaai results
    # load the results
    results_files = glob.glob(os.path.join(base_results_folder, "results_vlaai/eval*.json"))

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
    plt.savefig(os.path.join(base_results_folder, 'figures', "boxplot_vlaai.pdf"))


