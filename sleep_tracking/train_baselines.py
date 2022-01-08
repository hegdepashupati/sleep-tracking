from sleep_tracking.context import Context
from sleep_tracking.utils import get_root_directory
from sleep_tracking.classification.performance_builder import PerformanceSummaryBuilder
from sleep_tracking.classification.data_loader import SleepDatasetLoader
from sleep_tracking.classification.prediction_instances import PredictionInstances
from sleep_tracking.classification.baselines import BaselineClassifier, RandomForestClassifier, \
    LogisticRegressionClassifier

import os
import datetime
import sys
import pickle
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

CLASSIFIERS = ['logistic-regression', 'random-forest']


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_name", "-c", type=str, default="logistic-regression", choices=CLASSIFIERS,
                        help="Environment to use")
    parser.add_argument("--num_runs", type=int, default=8,
                        help="Number of independent training runs")
    return parser.parse_args(args)


def get_classifier(classifier):
    if classifier == 'logistic-regression':
        return LogisticRegressionClassifier()
    elif classifier == 'random-forest':
        return RandomForestClassifier()
    else:
        raise ValueError("wrong classifier name passed")


def compute_test_predictions(model: BaselineClassifier, testing_x: np.ndarray):
    predicted_labels = model.classifier.predict(testing_x)
    predicted_probs = model.classifier.predict_proba(testing_x)
    return predicted_labels, predicted_probs


def trainer(args):
    trainer_id, data_split, classifier_name = args
    print("Trainer id", trainer_id, "started")

    # setup classifier
    classifier = get_classifier(classifier_name)

    # train classifier
    # TODO: modify the SleepDataset implementation to access the full dataset better!
    training_x, training_y = data_split.training_set[np.arange(len(data_split.training_set))]
    classifier.fit(training_x, training_y)

    # generate test predictions
    testing_x, testing_y = data_split.testing_set[np.arange(len(data_split.testing_set))]
    test_predicted_labels, test_predicted_probs = compute_test_predictions(classifier, testing_x)
    test_predictions = PredictionInstances(true_labels=testing_y,
                                           predicted_labels=test_predicted_labels,
                                           predicted_probs=test_predicted_probs)

    print("Trainer id", trainer_id, "finished")

    return test_predictions


# main function
def main(args):
    # create a pool with cpu_count() workers
    pool = Pool(processes=cpu_count())

    # generate random test and train splits
    data_loader = SleepDatasetLoader(num_splits=args.num_runs)
    data_splits = data_loader.generate_data_splits_without_validation(subject_ids=Context.SUBJECTS,
                                                                      features=Context.FEATURES)

    # setup experiment name and a directory
    experiment_name = f"{args.classifier_name}_{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}"
    experiment_dir = os.path.join(get_root_directory(), f"outputs/models/{experiment_name}")
    os.makedirs(experiment_dir)

    # run classifier num_runs times
    results = pool.map(trainer, zip(range(args.num_runs), data_splits, [args.classifier_name] * args.num_runs))

    # collate results into a pickle file
    sleep_wake_roc, rem_roc, nrem_roc, performance_metrics = PerformanceSummaryBuilder.build_three_class_roc(results)
    with open(os.path.join(experiment_dir, "summary.txt"), "w") as output_file:
        print("Accuracy : {:.3f}".format(performance_metrics.accuracy), file=output_file)
        print("Kappa : {:.3f}".format(performance_metrics.kappa), file=output_file)
        print("Wake correct : {:.3f}".format(performance_metrics.wake_correct), file=output_file)
        print("REM correct : {:.3f}".format(performance_metrics.rem_correct), file=output_file)
        print("NREM correct : {:.3f}".format(performance_metrics.nrem_correct), file=output_file)

    # save all the results
    output = {"args": args,
              "data_splits": data_splits,
              "results": results,
              "sleep_wake_roc": sleep_wake_roc,
              "rem_roc": rem_roc,
              "nrem_roc": nrem_roc,
              "summary": performance_metrics
              }

    with open(os.path.join(experiment_dir, "output.pkl"), "wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Experiment results saved at %s" % os.path.join(experiment_dir, "output.pkl"))


# entry point of the script
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
