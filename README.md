This codebase provides an RNN model to classify sleep stages based on acceleration and heart rate measurements from Apple Watch. The predictive performance of RNN model is compared with logistic-regression and random-forest baselines. 

### Data

Data collected using the Apple Watch is available on PhysioNet: [link](https://alpha.physionet.org/content/sleep-accel/1.0.0/)

## Pre-processing the data

To convert the raw data into copped data run, we utilize the pre-processing script provided as a part of this [codebase](https://github.com/ojwalch/sleep_classifiers/). 
1. Download the [data](https://alpha.physionet.org/content/sleep-accel/1.0.0/).
2. Paste the `heart_rate`, `labels`, and `motion` folders into the `data` directory in this repository, overwriting the empty folders.
3. Download this [codebase](https://github.com/ojwalch/sleep_classifiers/) and run `sleep_classifiers/preprocessing/preprocessing_runner.py`.
4. Move the cropped data files into `data/cropped/` folder.   

## Generating features and training classification models

First we convert the cropped data into features that can be used for classification modeling
1. Run `PYTHONPATH=$(pwd) python sleep_tracking/generate_features.py`.

We can run the baselines and RNN classification models as below 
1. For logistic regression classifier, run `PYTHONPATH=$(pwd) python sleep_tracking/train_baselines.py -c logistic-regression`.
2. For random forest classifier, run `PYTHONPATH=$(pwd) python sleep_tracking/train_baselines.py -c random-forest`.
3. For RNN-based classifier, run `PYTHONPATH=$(pwd) python sleep_tracking/train_rnn.py`.


## Generating model summary and analysis figures
This codebase also provides some accompanying scripts to make data visualizations and model summarization. 
1. Generate data visualizations with `PYTHONPATH=$(pwd) python sleep_tracking/summarize_data.py`.
2. Generate model analysis with `PYTHONPATH=$(pwd) python sleep_tracking/summarize_classification_results.py`. 

Note that this requires training the models first and modifying the `summarize_classification_results.py` script to point to the modeling output directory. 
The trained model outputs are stored in  `outputs/models` directory, and the summary figures will be stored in `outputs/figures` directory. 

## License

This software is open source and under an MIT license.
