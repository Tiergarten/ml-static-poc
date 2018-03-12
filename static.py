#!/usr/bin/env python

"""
Samples were extracted from theZoo malware repository (https://github.com/ytisf/theZoo.git) with the following commands:

for i in $(ls);do 7z x -pinfected $i/$i.zip;done
find . -exec file {} \; | grep PE | grep executable | while read line;do FN=$(echo $line | cut -d ':' -f1);cp -f -v "$FN" ~/dev/ml-poc1/malware/;done

benign:
	find "Program Files" "Program Files (x86)"/ -name '*.exe' -exec cp {} /mnt/c/benign/ \;
	for i in $(find Windows -name '*.exe' 2>/dev/null | head -n 309);do cp "$i" /mnt/c/benign/;done
"""

import os
import yara
import string
import pefile
import re

from os import listdir
from os.path import isfile, join
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.linalg import *
from pyspark.mllib.classification import SVMWithSGD, SVMModel
import pdb
import time
import sys
import numpy as np
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
import random
import logging
from features_extractor import *
from stats import *


def standardise_features(labeled_rdd):
    rdd = labeled_rdd.map(lambda row: row[0])

    std = StandardScaler()
    model = std.fit(rdd)
    features_transform = model.transform(rdd)
    return features_transform


def add_labels(labeled, unlabeled):
    return zip(labeled.map(lambda row: row[1]).collect(), unlabeled.collect())


def get_sc():
    return SparkContext("local", "static-poc")


def train_classifier_and_measure(ctype, training_data, test_data):
    if ctype == "svm":
        model = SVMWithSGD.train(training_data, iterations=100)
    elif ctype == "rf":
        model = DecisionTree.trainClassifier(training_data, 2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5,
                                             maxBins=32)

    output = []
    for lp in test_data.collect():
        output.append((lp.label, float(model.predict(lp.features))))

    return output


# UGLY: rewrite
def align_training_data(data):
    if data.count() % 2 != 0:
        data = get_df(data.take(data.count() - 1)).rdd

    pos_rdd = data.filter(lambda lp: lp.label == 1.0)
    neg_rdd = data.filter(lambda lp: lp.label == 0.0)

    if pos_rdd.count() > neg_rdd.count():
        delta = pos_rdd.count() - neg_rdd.count()
        return neg_rdd.union(get_df(pos_rdd.take(pos_rdd.count() - delta)).rdd).map(
            lambda row: LabeledPoint(row.label, row.features))
    else:
        delta = neg_rdd.count() - pos_rdd.count()
        return pos_rdd.union(get_df(neg_rdd.take(neg_rdd.count() - delta)).rdd).map(
            lambda row: LabeledPoint(row.label, row.features))


def main():
    d = get_dir_features("./benign/", 0.0, sc).union(get_dir_features("./malware/", 1.0, sc))
    sample_size = 0.1 if len(sys.argv) == 1 else float(sys.argv[1])
    logging.info("Using sample size: %f", sample_size)

    if sample_size != 1:
        i = d.sample(True, sample_size)
    else:
        i = d

    std = standardise_features(i)

    # Convert to RDD for classifier train function
    rdd = get_df(add_labels(i, std))
    stuff = rdd.rdd.map(lambda row: LabeledPoint(row[0], row[1]))

    for m in ["svm", "rf"]:
        # Generate random split for test / train
        #	test, train = stuff.randomSplit(weights=[0.3, 0.7], seed=int(time.time()))

        # K-folds (n=10)
        split_n = 10
        splits = stuff.randomSplit([float(1) / split_n] * split_n, seed=int(time.time()))
        results = []
        for i in range(0, split_n):
            test = splits[i]

            train = list(splits)
            train.remove(test)

            agg_train = train[0]
            for train_idx in range(1, len(train)):
                agg_train.union(train[train_idx])

            train = agg_train

            if m == "svm":
                # {over,under}sample training data so its 50/50 split pos/neg
                train = align_training_data(train)

            logging.debug("%s", debug_samples(train, test))

            model_predictions = train_classifier_and_measure(m, train, test)
            iteration_results = ResultStats(m, model_predictions)
            logging.debug("Fold: %d %s", i, iteration_results)

            results.append(iteration_results.to_numpy())

        logging.info("[%s] K-Folds avg: %s", m, ResultStats.print_numpy(np.average(results, axis=0)))

if __name__ == "__main__":
    # np.set_printoptions(suppress=True)
    sc = get_sc()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
