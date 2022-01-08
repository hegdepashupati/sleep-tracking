import os

import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sleep_tracking.utils import get_root_directory
from sleep_tracking.context import Context
from sleep_tracking.classification.data_loader import SleepDatasetLoader
from sleep_tracking.preprocessing.cropped_data_loader import CroppedDatasetLoader

PALETTE = "Set2"


# 1. histogram of PSG labels and targets
def plot_psg_histogram(features):
    table_sleep_state_percentages = features.groupby(["subject"])["psg_labels_cat"].value_counts(
        normalize=True).reset_index().rename(
        columns={"level_1": "state", "psg_labels_cat": "ptime"})
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 5))

    sns.barplot(data=features["psg_labels_cat"].value_counts().reset_index(),
                x="index", y="psg_labels_cat", ax=ax1, palette=PALETTE)
    ax1.set_title("(a) Epoch counts across sleep stages")
    ax1.set_xlabel("")
    ax1.set_ylabel("Number of epochs")
    sns.barplot(
        data=table_sleep_state_percentages, ax=ax2,
        x="state", y="ptime", ci="sd", palette=PALETTE,
        order=table_sleep_state_percentages.state.value_counts().index)
    ax2.set_xlabel("")
    ax2.set_ylabel("% time")
    ax2.set_title("(b) Time spent (%) across sleep stages")
    fig.savefig(os.path.join(get_root_directory(), "outputs/figures/histogram_sleep_counts.pdf"), bbox_inches="tight")
    plt.close()


# 2. lineplot of motion and PSG labels and heart rate data
def plot_cropped_data_all_subjects(cropped_psg, cropped_hr, cropped_motion):
    for subject in tqdm.tqdm(Context.SUBJECTS):
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(14, 4 * 3))
        sidx = cropped_psg.subject == subject
        ax1.plot(cropped_psg.loc[sidx, "timestamp"], cropped_psg.loc[sidx, "psg_labels_three_cat"], color='k', alpha=.7)
        ax1.set_ylabel("Sleep stage")
        for label in ax1.get_yticklabels():
            label.set_ha("right")
            label.set_rotation(90)
        sidx = cropped_hr.subject == subject
        ax2.plot(cropped_hr.loc[sidx, "timestamp"], cropped_hr.loc[sidx, "hr"], color='k', alpha=.7, marker='.')
        ax2.set_ylabel("Heartrate (BPM)")
        sidx = cropped_motion.subject == subject
        ax3.plot(cropped_motion.loc[sidx, "timestamp"], cropped_motion.loc[sidx, "accx"], label='x', marker='.')
        ax3.plot(cropped_motion.loc[sidx, "timestamp"], cropped_motion.loc[sidx, "accy"], label='y', marker='.')
        ax3.plot(cropped_motion.loc[sidx, "timestamp"], cropped_motion.loc[sidx, "accz"], label='z', marker='.')
        ax3.set_xlabel("Timestamp")
        ax3.set_ylabel("Acceleration (G)")
        ax3.legend()
        fig.savefig(os.path.join(get_root_directory(), f"outputs/figures/{subject}_cropped.png"),
                    bbox_inches='tight', dpi=100)
        plt.close()


# 3. lineplot of features data
def plot_feature_data_all_subjects(features):
    sleep_categories = {0: "wake", 1: "nrem", 2: "rem"}
    features['psg_labels_cat'] = pd.Categorical(features['psg_labels'].map(sleep_categories),
                                                ["wake", "nrem", "rem"], ordered=True)

    for subject in tqdm.tqdm(Context.SUBJECTS):
        fig, axs = plt.subplots(nrows=6, ncols=1, figsize=(14, 6 * 3))
        sidx = features.subject == subject

        axs[0].plot(features.loc[sidx, "timestamp"], features.loc[sidx, "psg_labels_cat"], color='k', alpha=.7)
        axs[0].set_ylabel("Sleep stage")
        for label in axs[0].get_yticklabels():
            label.set_ha("right")
            label.set_rotation(90)

        axs[1].plot(features.loc[sidx, "timestamp"], features.loc[sidx, "hr_avg"], color='k', alpha=.7, marker='.')
        axs[1].set_ylabel("Heartrate average")

        axs[2].plot(features.loc[sidx, "timestamp"], features.loc[sidx, "hr_variation"], color='k', alpha=.7,
                    marker='.')
        axs[2].set_ylabel("Heartrate variation")

        axs[3].plot(features.loc[sidx, "timestamp"], features.loc[sidx, "motion_avg"], color='k', alpha=.7, marker='.')
        axs[3].set_ylabel("Motion total")

        axs[4].plot(features.loc[sidx, "timestamp"], features.loc[sidx, "motion_std"], color='k', alpha=.7, marker='.')
        axs[4].set_ylabel("Motion variation")

        axs[5].plot(features.loc[sidx, "timestamp"], features.loc[sidx, "cosine_proxy"], color='k', alpha=.7,
                    marker='.')
        axs[5].set_ylabel("Circadian proxy")
        axs[5].set_xlabel("Timestamp")
        fig.savefig(os.path.join(get_root_directory(), f"outputs/figures/{subject}_features.png"),
                    bbox_inches='tight', dpi=100)
        plt.close()


def plot_sleep_duration_histogram(cropped_psg):
    sleep_time = cropped_psg.groupby(['subject'])['timestamp'].max() - cropped_psg.groupby(['subject'])[
        'timestamp'].min()
    sleep_size = cropped_psg.groupby(['subject']).size()
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(14, 5))
    axs[0].hist(sleep_time.to_numpy() / (60 * 60))
    axs[0].set_title("(b) PSG recording duration (hours)")
    axs[0].set_ylabel('Number of subjects')
    axs[0].set_xlabel('')

    axs[1].hist(sleep_size.to_numpy())
    axs[1].set_title("(a) Number of epochs")
    axs[1].set_ylabel('Number of subjects')
    axs[1].set_xlabel('')

    fig.savefig(os.path.join(get_root_directory(), f"outputs/figures/sleep_duration_histogram.pdf"),
                bbox_inches='tight')
    plt.close()


# calculate transition matrix for different individuals
def calculate_transitions(features):
    features['psg_labels_cat'] = features['psg_labels'].map({0: "Wake", 1: "NREM", 2: "REM"})

    unique_states = ["Wake", "NREM", "REM"]

    transition_counts = np.full((len(unique_states), len(unique_states)), fill_value=0, dtype=np.int32)
    for subject in Context.SUBJECTS:
        transition_sequence = features.loc[features['subject'] == subject, 'psg_labels'].to_numpy()
        for i, j in zip(transition_sequence[:-1], transition_sequence[1:]):
            transition_counts[i, j] += 1
    transition_probs = transition_counts / transition_counts.sum(axis=1, keepdims=True)

    transition_counts = pd.DataFrame(transition_counts, index=unique_states, columns=unique_states)
    transition_probs = pd.DataFrame(transition_probs, index=unique_states, columns=unique_states)

    return transition_counts, transition_probs


def plot_sleep_stage_transition_probs(features):
    transition_counts_all, transition_probs_all = calculate_transitions(features)

    # visualize transition matrices
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(transition_counts_all, cmap="YlGnBu", annot=True, ax=axs[0], fmt='6d')
    axs[0].set_xlabel('Transition FROM state')
    axs[0].set_ylabel('Transition TO state')
    axs[0].set_title('(a) Epoch counts')
    sns.heatmap(transition_probs_all, cmap="YlGnBu", annot=True, ax=axs[1])
    axs[1].set_xlabel('Transition FROM state')
    axs[1].set_ylabel('Transition TO state')
    axs[1].set_title('(b) Transition probability')
    fig.savefig(os.path.join(get_root_directory(), f"outputs/figures/sleep_stage_transitions.pdf"),
                bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # load cropped dataset
    cropped_hr = CroppedDatasetLoader.load_hr(Context.SUBJECTS)
    cropped_psg = CroppedDatasetLoader.load_psg(Context.SUBJECTS)
    cropped_motion = CroppedDatasetLoader.load_motion(Context.SUBJECTS)
    cropped_psg["psg_labels_all_cat"] = pd.Categorical(cropped_psg["psg_labels_all"].map(
        {"unscored": "Unscored", "wake": "Wake", "n1": "N1", "n2": "N2", "n3": "N3", "n4": "N4", "rem": "REM"}),
        ["Unscored", "Wake", "N1", "N2", "N3", "N4", "REM"],
        ordered=True)
    cropped_psg["psg_labels_three_cat"] = pd.Categorical(
        cropped_psg["psg_labels_three"].map({"unscored": "Unscored", "wake": "Wake", "nrem": "NREM", "rem": "REM"}),
        ["Unscored", "Wake", "NREM", "REM"],
        ordered=True)

    # load modeling dataset
    features = SleepDatasetLoader.load_modeling_data(Context.SUBJECTS)
    features['psg_labels_cat'] = pd.Categorical(features['psg_labels'].map(
        {0: "Wake", 1: "NREM", 2: "REM"}),
        ["Wake", "NREM", "REM"],
        ordered=True)

    plot_psg_histogram(features)
    plot_sleep_duration_histogram(cropped_psg)
    plot_sleep_stage_transition_probs(features)
    plot_feature_data_all_subjects(features)
    plot_cropped_data_all_subjects(cropped_psg, cropped_hr, cropped_motion)
