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


def get_train_test_split(labelled_rdd, split_n=10):
    ret = []
    splits = labelled_rdd.randomSplit([float(1) / split_n] * split_n, seed=int(time.time()))
    for i in range(0, split_n):
        test = splits[i]

        train_list = list(splits)
        train_list.remove(test)

        train_rdd = train_list[0]
        for train_idx in range(1, len(train_list)):
            train_rdd.union(train_list[train_idx])

        ret.append((test, train_rdd))

    return ret


def get_std_scaler(labeledpoints):
    std = StandardScaler()
    train_features = labeledpoints.map(lambda lp: lp.features)

    scaler_model = std.fit(train_features)
    transformed_features = scaler_model.transform(train_features)

    transformed_label_features = \
        zip(labeledpoints.map(lambda lp: lp.label).collect(), transformed_features.collect())

    return to_labeled_points(transformed_label_features), scaler_model


def to_labeled_points(rows):
    return get_df(rows).rdd.map(lambda row: LabeledPoint(row[0], row[1]))


def get_samples(sample_sz=1):
    # Row(features=DenseVector[...], label=0.0)
    features_from_disk_rdd = get_dir_features_rdd("./benign/", 0.0, sc).union(
        get_dir_features_rdd("./malware/", 1.0, sc))

    logging.info("Using sample size: %f", sample_sz)
    if sample_sz != 1:
        features_from_disk_rdd = features_from_disk_rdd.sample(True, sample_sz)

    return get_df(features_from_disk_rdd)

def main():
    # Map label, [features] -> LabeledPoint(label, [features])
    labelled_rdd = get_samples().rdd.map(lambda row: LabeledPoint(row[1], row[0]))

    for m in ["svm", "rf"]:
        for scaler in [None, 'std']:
            results = []
            count = 0
            for test, train in get_train_test_split(labelled_rdd):
                if m == "svm":
                    # {over,under}sample training data so its 50/50 split pos/neg
                    train = align_training_data(train)

                logging.debug("%s", debug_samples(train, test))

                if scaler:
                    train_features, std = get_std_scaler(train)

                    test_features = std.transform(train_features.map(lambda lp: lp.features))
                    test_label_features = zip(train_features.map(lambda lp: lp.label).collect(), test_features.collect())
                    test = to_labeled_points(test_label_features)

                model_predictions = train_classifier_and_measure(m, train, test)
                iteration_results = ResultStats(m, model_predictions)
                logging.info("Fold: %d %s", count, iteration_results)

                results.append(iteration_results.to_numpy())
                count += 1

            logging.info("[%s] K-Folds avg: %s", '{}-{}'.format(m, scaler), ResultStats.print_numpy(np.average(results, axis=0)))


if __name__ == "__main__":
    sc = get_sc()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
