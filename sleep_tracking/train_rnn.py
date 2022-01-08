from sleep_tracking.classification.performance_builder import PerformanceSummaryBuilder
from sleep_tracking.classification.data_loader import SleepDatasetLoader
from sleep_tracking.context import Context
from sleep_tracking.classification.prediction_instances import PredictionInstances
from sleep_tracking.classification.rnn import ClassifierRNN
from sleep_tracking.utils import get_root_directory
from sleep_tracking.classification.utils import compute_class_weights

import os
import datetime
import sys
import pickle
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

import torch
from torch import nn
from torch.utils.data import DataLoader
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=100,
                        help="Maximum number of epochs for training")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience parameter for early stopping")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Learning rate for training")
    parser.add_argument("--num_latents", type=int, default=16,
                        help="Latent space dimensions")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="Number of RNN layers")
    parser.add_argument("--num_runs", type=int, default=8,
                        help="Number of independent training runs")
    return parser.parse_args(args)


def compute_test_predictions(model, dataloader):
    model.eval()
    true_labels_all = []
    predicted_probs_all = []
    predicted_labels_all = []

    with torch.no_grad():
        for x, y in dataloader:
            predicted_probs = model.predict_proba(x).detach().cpu().numpy()
            predicted_labels = model.predict(x).detach().cpu().numpy()
            true_labels_all.append(y.squeeze(0))
            predicted_probs_all.append(predicted_probs.squeeze(0))
            predicted_labels_all.append(predicted_labels.squeeze(0))

    return PredictionInstances(true_labels=np.concatenate(true_labels_all),
                               predicted_labels=np.concatenate(predicted_labels_all),
                               predicted_probs=np.concatenate(predicted_probs_all))


def trainer(fargs):
    trainer_id, data_split, args, experiment_dir = fargs
    print("Trainer id", trainer_id, "started")

    # setup data loaders
    train_loader = DataLoader(data_split.training_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(data_split.validation_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(data_split.testing_set, batch_size=1, shuffle=False)

    # setup model, optimizer and loss function
    model = ClassifierRNN(num_features=len(Context.FEATURES),
                          num_latents=args.num_latents,
                          num_layers=args.num_layers,
                          num_classes=3)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)
    training_labels = data_split.training_set.df[data_split.training_set.target].to_numpy()
    criterion = nn.CrossEntropyLoss(reduction='mean',
                                    weight=torch.tensor(compute_class_weights(training_labels)))

    # setup trainer
    trainer = create_supervised_trainer(model, optimizer, criterion)
    val_metrics = {
        "accuracy": Accuracy(),
        "nll": Loss(criterion)
    }
    evaluator = create_supervised_evaluator(model, metrics=val_metrics)

    @trainer.on(Events.ITERATION_COMPLETED(every=5))
    def log_training_loss(engine):
        print(f"Run[{trainer_id}] Epoch[{engine.state.epoch}] Loss: {engine.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print(
            f"Run[{trainer_id}] Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(
            f"Run[{trainer_id}] Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

    def early_stropping_score_function(engine):
        val_loss = engine.state.metrics['nll']
        return -val_loss

    handler = EarlyStopping(patience=args.patience, score_function=early_stropping_score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    # training event
    trainer.run(train_loader, max_epochs=args.max_epochs)

    # save classifier checkpoint
    checkpoint_path = os.path.join(experiment_dir,
                                   "model_final_{}.ckpt".format(trainer_id))
    model.save(checkpoint_path)

    # generate test predictions
    test_predictions = compute_test_predictions(model, test_loader)

    print("Trainer id", trainer_id, "finished")

    return test_predictions


# main function
def main(args):
    # create a pool with cpu_count() workers
    pool = Pool(processes=cpu_count())

    # generate random test and train splits
    data_loader = SleepDatasetLoader(num_splits=args.num_runs)
    data_splits = data_loader.generate_data_splits_with_validation(subject_ids=Context.SUBJECTS,
                                                                   features=Context.FEATURES)

    # setup experiment name and directory
    experiment_name = f"rnn_{args.num_latents}_{args.max_epochs}_{args.patience}_{args.lr}_{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}"
    experiment_dir = os.path.join(get_root_directory(), f"outputs/models/{experiment_name}")
    os.makedirs(experiment_dir)

    # train the classifier num_runs times
    results = pool.map(trainer, zip(range(args.num_runs), data_splits,
                                    [args] * args.num_runs, [experiment_dir] * args.num_runs))

    # save results from all the runs into a pickle file
    sleep_wake_roc, rem_roc, nrem_roc, performance_metrics = PerformanceSummaryBuilder.build_three_class_roc(results)
    with open(os.path.join(experiment_dir, "summary.txt"), "w") as output_file:
        print("Accuracy : {:.3f}".format(performance_metrics.accuracy), file=output_file)
        print("Kappa : {:.3f}".format(performance_metrics.kappa), file=output_file)
        print("Wake correct : {:.3f}".format(performance_metrics.wake_correct), file=output_file)
        print("REM correct : {:.3f}".format(performance_metrics.rem_correct), file=output_file)
        print("NREM correct : {:.3f}".format(performance_metrics.nrem_correct), file=output_file)

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
