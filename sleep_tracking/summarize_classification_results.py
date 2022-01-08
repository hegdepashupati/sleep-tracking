import os
import pickle
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, cohen_kappa_score, \
    matthews_corrcoef

import seaborn as sns
from sleep_tracking.classification.performance_builder import PerformanceSummaryBuilder
from sleep_tracking.classification.performance_summary import ROCPerformance
from sleep_tracking.utils import get_root_directory
import matplotlib.pyplot as plt


def get_base_dir(classifier_name):
    if classifier_name == 'logistic-regression' or classifier_name == 'random-forest':
        return os.path.join(get_root_directory(), "outputs/baselines")
    elif classifier_name == 'rnn':
        return os.path.join(get_root_directory(), "rnn")
    else:
        raise ValueError("wrong classifier name passed")


def load_results(experiment_dir):
    with open(os.path.join(get_root_directory(), "outputs/models/", experiment_dir, "output.pkl"), "rb") as handle:
        results = pickle.load(handle)
    return results


def combine_all_results_into_dataframe(experiments, experiments_summary):
    results = []
    for expmnt in experiments:
        num_runs = experiments_summary[expmnt].results['args'].num_runs
        for run_id in range(num_runs):
            fulldf = experiments_summary[expmnt].results['data_splits'][run_id].testing_set.df
            subdfs = []
            for subject in experiments_summary[expmnt].results['data_splits'][run_id].testing_set.subjects:
                subdfs.append(fulldf.loc[fulldf.subject == subject, :].reset_index())
            df = pd.concat(subdfs, ignore_index=True)
            df['true_labels'] = experiments_summary[expmnt].results['results'][run_id].true_labels
            df['predicted_labels'] = experiments_summary[expmnt].results['results'][run_id].predicted_labels
            for cid in range(experiments_summary[expmnt].results['results'][run_id].predicted_probs.shape[1]):
                df['predicted_probs_' + str(cid)] = experiments_summary[expmnt].results['results'][
                                                        run_id].predicted_probs[
                                                    :, cid]
            df['classifier'] = expmnt
            df['run'] = run_id
            results.append(df.reset_index())
    return pd.concat(results, ignore_index=True)


def build_one_vs_rest_roc(actual_labels, predicted_probs, positive_class):
    false_positive_spread, true_positive_spread = PerformanceSummaryBuilder.get_axes_bins()
    count = 0

    for labels, predprobs in zip(actual_labels, predicted_probs):
        one_vs_rest_lables = np.zeros_like(labels)
        one_vs_rest_lables[labels == positive_class] = 1
        one_vs_rest_predprobs = predprobs[:, positive_class]

        false_positive_rates, true_positive_rates, thresholds = roc_curve(
            one_vs_rest_lables,
            one_vs_rest_predprobs,
            pos_label=1,
            drop_intermediate=False)
        count = count + 1
        true_positive_spread += np.interp(false_positive_spread, false_positive_rates,
                                          true_positive_rates)

    true_positive_spread = true_positive_spread / count

    false_positive_spread = np.insert(false_positive_spread, 0, 0)
    true_positive_spread = np.insert(true_positive_spread, 0, 0)

    return ROCPerformance(false_positive_rates=false_positive_spread, true_positive_rates=true_positive_spread)


def make_one_vs_rest_roc_plots(experiments, experiments_summary):
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(5 * 3, 4))
    for expmnt in experiments:
        wake_roc_results = build_one_vs_rest_roc(
            actual_labels=[r.true_labels for r in
                           experiments_summary[expmnt].results['results']],
            predicted_probs=[r.predicted_probs for r in
                             experiments_summary[expmnt].results['results']],
            positive_class=0)
        nrem_roc_results = build_one_vs_rest_roc(
            actual_labels=[r.true_labels for r in
                           experiments_summary[expmnt].results['results']],
            predicted_probs=[r.predicted_probs for r in
                             experiments_summary[expmnt].results['results']],
            positive_class=1)
        rem_roc_results = build_one_vs_rest_roc(
            actual_labels=[r.true_labels for r in
                           experiments_summary[expmnt].results['results']],
            predicted_probs=[r.predicted_probs for r in
                             experiments_summary[expmnt].results['results']],
            positive_class=2)
        axs[0].plot(wake_roc_results.false_positive_rates,
                    wake_roc_results.true_positive_rates,
                    c=experiments_summary[expmnt].color, lw=2, alpha=0.7)
        axs[1].plot(nrem_roc_results.false_positive_rates,
                    nrem_roc_results.true_positive_rates,
                    c=experiments_summary[expmnt].color, lw=2, alpha=0.7)
        axs[2].plot(rem_roc_results.false_positive_rates,
                    rem_roc_results.true_positive_rates,
                    c=experiments_summary[expmnt].color, lw=2, alpha=0.7)
        for ax in axs:
            ax.plot([0, 1], [0, 1], ls='--', c='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        axs[0].plot([], [], c=experiments_summary[expmnt].color, label=experiments_summary[expmnt].name)
    axs[0].legend()

    axs[0].set_title("Wake vs others")
    axs[0].set_xlabel("Fraction of NREM/REM scored as wake")
    axs[0].set_ylabel("Fraction of wake scored as wake")
    axs[1].set_title("NREM vs others")
    axs[1].set_xlabel("Fraction of Wake/REM scored as NREM")
    axs[1].set_ylabel("Fraction of NREM scored as NREM")
    axs[2].set_title("REM vs others")
    axs[2].set_xlabel("Fraction of Wake/NREM scored as REM")
    axs[2].set_ylabel("Fraction of REM scored as REM")

    fig.savefig(os.path.join(get_root_directory(), "outputs/figures/one_vs_rest_roc_comparison.pdf"),
                bbox_inches='tight')
    plt.close()


SubjectLevelResults = namedtuple('SubjectLevelPredictions', ['labels', 'subjects', 'timestamps', 'predictions'])


def extract_subject_levels_results(expmnt, experiments_summary, run_idx):
    true_labels = experiments_summary[expmnt].results['data_splits'][run_idx].testing_set.df[
        'psg_labels'].to_numpy()
    subject_labels = experiments_summary[expmnt].results['data_splits'][run_idx].testing_set.df[
        'subject'].to_numpy()
    subject_timestamps = experiments_summary[expmnt].results['data_splits'][run_idx].testing_set.df[
        'timestamp'].to_numpy()
    predicted_labels = experiments_summary[expmnt].results['results'][run_idx].predicted_labels
    return SubjectLevelResults(true_labels, subject_labels, subject_timestamps, predicted_labels)


def make_predictive_plots_for_all_subjects(experiments, experiments_summary):
    all_subject_results = {expmnt: extract_subject_levels_results(expmnt, experiments_summary, 0) for expmnt in
                           experiments}

    for subject in np.unique(all_subject_results[experiments[0]].subjects):

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 4))
        linewidths = [5, 3., 2.]
        linestyles = ['solid'] * 3
        alphas = [0.4, 0.6, 0.9]
        for expmnt, lw, ls, alpha in zip(all_subject_results.keys(), linewidths, linestyles, alphas):
            predictions = all_subject_results.get(expmnt).predictions[
                all_subject_results.get(expmnt).subjects == subject]
            times = all_subject_results.get(expmnt).timestamps[
                all_subject_results.get(expmnt).subjects == subject]
            ax.plot(times, predictions,
                    c=experiments_summary[expmnt].color,
                    label=experiments_summary[expmnt].name,
                    zorder=1, alpha=alpha, ls=ls, lw=lw)
        times = all_subject_results.get(experiments[0]).timestamps[
            all_subject_results.get(experiments[0]).subjects == subject]
        labels = all_subject_results.get(experiments[0]).labels[
            all_subject_results.get(experiments[0]).subjects == subject]
        ax.scatter(times, labels, marker='.', s=20,
                   c='k', label='Ground truth', zorder=2, alpha=0.8)
        ax.legend(loc='upper right')
        ax.set_title(f"Sleep stage predictions for subject {subject}")
        ax.set_xlabel("Timestamp (seconds)")
        ax.set_ylabel("Sleep stage")

        ax.set_yticks([0.0, 1.0, 2.0])
        ax.set_yticklabels(['Wake', 'NREM', 'REM'], rotation=90)
        fig.subplots_adjust()
        fig.savefig(os.path.join(get_root_directory(), f"outputs/figures/{subject}_predictions.pdf"),
                    bbox_inches='tight')
        plt.close()


def plot_subjectlevel_classifier_accuracy(experiments_resultsdf):
    aggdf = experiments_resultsdf.groupby(['classifier', 'subject', 'true_labels']).agg(
        {'correctly_predicted': ['count', 'mean']}).reset_index()
    aggdf.columns = ['_'.join(col).replace('Count', '').strip().rstrip("_") for col in aggdf.columns.values]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.boxplot(data=aggdf, x='classifier', y='correctly_predicted_mean', hue="true_labels",
                palette="Set3", ax=ax)
    ax.set_title('Subject-level classifier accuracies')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Classifier')
    ax.set_xticklabels(['Logistic Regression', 'Random Forest', 'RNN'])

    leg = ax.get_legend()
    new_title = 'True sleep stages'
    leg.set_title(new_title)
    new_labels = ['wake', 'NREM', 'REM']
    for t, l in zip(leg.texts, new_labels):
        t.set_text(l)
    fig.savefig(os.path.join(get_root_directory(), f"outputs/figures/subject_level_accuracy.pdf"),
                bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(experiments_resultsdf, experiments_summary):
    fig, axs = plt.subplots(ncols=3, figsize=(6 * 3, 5))
    for ax, expmnt in zip(axs, experiments):
        true = experiments_resultsdf.loc[experiments_resultsdf['classifier'] == expmnt, 'true_labels'].to_numpy()
        pred = experiments_resultsdf.loc[experiments_resultsdf['classifier'] == expmnt, 'predicted_labels'].to_numpy()
        cm = confusion_matrix(true, pred, normalize='all')
        df_cm = pd.DataFrame(cm, index=['Wake', 'NREM', 'REM'], columns=['Wake', 'NREM', 'REM'])
        sns.set(font_scale=1.4)
        sns.heatmap(df_cm, annot=True, fmt='.2f', annot_kws={"size": 16}, ax=ax, vmin=0.0, vmax=1.0, cmap="rocket_r")
        ax.set_title(experiments_summary.get(expmnt).name)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    fig.subplots_adjust()
    fig.savefig(os.path.join(get_root_directory(), f"outputs/figures/confusion_matrix.pdf"),
                bbox_inches='tight')
    plt.close()


def print_overall_summary_metrics(experiments_resultsdf, experiments_summary):
    with open(os.path.join(get_root_directory(), f"outputs/figures/overall_summary.txt"), "w") as output_file:
        print(''.join(['*'] * 26 + [' Overall summary '] + ['*'] * 26), file=output_file)
        print(f'Method & Acuracy & Balanced accuracy & ROC AUC & Cohen Kappa & Mathews corrcoef', file=output_file)
        for classifier in experiments:
            true = experiments_resultsdf.loc[
                experiments_resultsdf['classifier'] == classifier, 'true_labels'].to_numpy()
            pred = experiments_resultsdf.loc[
                experiments_resultsdf['classifier'] == classifier, 'predicted_labels'].to_numpy()
            pred_probs = experiments_resultsdf.loc[
                experiments_resultsdf['classifier'] == classifier, ['predicted_probs_0', 'predicted_probs_1',
                                                                    'predicted_probs_2']].to_numpy()
            combined_accuracy = accuracy_score(true, pred)
            combined_balanced_accuracy = balanced_accuracy_score(true, pred)
            combined_auc = roc_auc_score(true, pred_probs, average="macro", multi_class="ovo")
            combined_cohen_kappa_score = cohen_kappa_score(true, pred)
            combined_mathews_corrcoef = matthews_corrcoef(true, pred)
            print(
                f"{experiments_summary.get(classifier).name} & {combined_accuracy:.3f} & {combined_balanced_accuracy:.3f} & "
                f"{combined_auc: .3f} & {combined_cohen_kappa_score:.3f} & {combined_mathews_corrcoef:.3f}",
                file=output_file)


def print_threshold_summary_metrics(experiments_summary):
    with open(os.path.join(get_root_directory(), f"outputs/figures/threshold_summary.txt"), "w") as output_file:
        print(''.join(['*'] * 26 + [' Threshold summary '] + ['*'] * 26), file=output_file)
        print(f'Method & Wake correct & NREM correct & REM correct & Kappa', file=output_file)
        for expmnt in experiments:
            classifier_summary = experiments_summary.get(expmnt).results['summary']
            print(
                f"{experiments_summary.get(expmnt).name} & {classifier_summary.wake_correct:.3f} & "
                f"{classifier_summary.nrem_correct: .3f} & {classifier_summary.rem_correct:.3f} & "
                f"{classifier_summary.kappa:.3f}", file=output_file)


if __name__ == "__main__":
    experiments = ['rnn', 'logreg', 'rf']
    experiments_dirs = {'rnn': 'rnn',
                        'logreg': 'logistic-regression',
                        'rf': 'random-forest'}
    experiments_colours = {'rnn': '#ee694f',
                           'logreg': '#077c94',
                           'rf': '#8b8f2b'}
    experiments_names = {'rnn': 'RNN',
                         'logreg': 'Logistic Regression',
                         'rf': 'Random Forest'}

    ExperimentSummary = namedtuple('ExperimentInfo', ['name', 'dir', 'color', 'results'])

    experiments_summary = {expmnt: ExperimentSummary(experiments_names.get(expmnt),
                                                     experiments_dirs.get(expmnt),
                                                     experiments_colours.get(expmnt),
                                                     load_results(experiments_dirs.get(expmnt)))
                           for expmnt in experiments}

    experiments_resultsdf = combine_all_results_into_dataframe(experiments, experiments_summary)
    assert np.all(experiments_resultsdf['true_labels'] == experiments_resultsdf['psg_labels'])
    experiments_resultsdf['correctly_predicted'] = experiments_resultsdf['true_labels'] == experiments_resultsdf[
        'predicted_labels']

    make_one_vs_rest_roc_plots(experiments, experiments_summary)
    make_predictive_plots_for_all_subjects(experiments, experiments_summary)

    plot_subjectlevel_classifier_accuracy(experiments_resultsdf)
    plot_confusion_matrices(experiments_resultsdf, experiments_summary)

    print_overall_summary_metrics(experiments_resultsdf, experiments_summary)
    print_threshold_summary_metrics(experiments_summary)
