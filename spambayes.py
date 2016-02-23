import helpers as h
import numpy as np

datasets = h.create_datasets(h.load_data())

training_statistics = h.class_statistics(datasets['training'])

results = zip(datasets['test'], map(lambda x : h.classify(training_statistics, x), datasets['test']))

tp = h.true_positives(results)
tn = h.true_negatives(results)
fp = h.false_positives(results)
fn = h.false_negatives(results)

print "True positives:", tp
print "True negatives:", tn
print "False positives:", fp
print "False negatives:", fn

print "Accuracy:", float(tp + tn) / len(datasets['test'])
