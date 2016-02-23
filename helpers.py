import numpy as np
import math

DATA_LOCATION = "spambase/spambase.data"

def bigN(target, mean, stdev):
  if stdev == 0:
    stdev = 0.01
  exponent = -pow((target - mean) / (2 * stdev), 2)
  coefficient = 1.0 / (pow(2 * math.pi, 0.5) * stdev)
  return coefficient * pow(math.e, exponent)

def conditional_probabilities(targets, means, standard_deviations):
  ns = map(lambda (x, y, z) : bigN(x, y, z), zip(targets, means, standard_deviations))
  return sum(np.log(ns))

def class_probabilities(stats, targets):
  pos_conditional_prob = math.log(stats['pos_prob']) + conditional_probabilities(targets,
                                                                                stats['pos_means'],
                                                                                stats['pos_stdevs'])
  neg_conditional_prob = math.log(stats['neg_prob']) + conditional_probabilities(targets,
                                                                                 stats['neg_means'],
                                                                                 stats['neg_stdevs'])

  return {
    "positive": pos_conditional_prob,
    "negative": neg_conditional_prob
  }

def class_statistics(training_set):
  positives = map(lambda x : x[:-1], filter(positive_filter, training_set))
  negatives = map(lambda x : x[:-1], filter(negative_filter, training_set))

  pos_prob = float(len(positives)) / len(training_set)
  neg_prob = float(len(negatives)) / len(training_set)

  pos_means = np.mean(positives, axis=0)
  neg_means = np.mean(negatives, axis=0)

  pos_stdevs = np.std(positives, axis=0)
  neg_stdevs = np.std(negatives, axis=0)

  return {
    "pos_prob": pos_prob,
    "neg_prob": neg_prob,
    "pos_means": pos_means,
    "neg_means": neg_means,
    "pos_stdevs": pos_stdevs,
    "neg_stdevs": neg_stdevs
  }

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Everything from here down is just data processing stuff.                  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Read data from spambase
def load_data(location=DATA_LOCATION):
  examples = []
  with open(location, 'r') as f:
    for line in f:
      examples.append(map(lambda x : float(x), line.split(",")))
  
  return examples

def create_datasets(data):
  positives = filter(positive_filter, data)
  negatives = filter(negative_filter, data)

  np.random.shuffle(positives)
  np.random.shuffle(negatives)

  training_data = positives[:len(positives) / 2] + negatives[:len(negatives) / 2]
  test_data = positives[len(positives) / 2:] + negatives[len(negatives) / 2:]

  np.random.shuffle(training_data)
  np.random.shuffle(test_data)
  
  return {
    "training": np.array(training_data),
    "test": np.array(test_data)
  }

def positive_filter(x):
  return x[-1] == 1.0

def negative_filter(x):
  return x[-1] == 0.0
