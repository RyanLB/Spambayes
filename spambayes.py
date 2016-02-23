import helpers as h
import numpy as np

datasets = h.create_datasets(h.load_data())

positive_probability = float(len(filter(h.positive_filter, datasets['training']))) / len(datasets['training'])
negative_probability = float(len(filter(h.negative_filter, datasets['training']))) / len(datasets['training'])

means = np.mean(datasets['training'], axis=0)
standard_deviations = np.stdev(datasets['training'], axis=0)


