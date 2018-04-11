from pyspark.sql import *
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.mllib.evaluation import BinaryClassificationMetrics
import numpy as np
import json


def get_df(data):
    return SparkSession.builder.getOrCreate().createDataFrame(data)


class ResultStats:
    def __init__(self, label, results):
        self.results = results

        self.tp = self.get_result_cnt(results, actual=1.0, predicted=1)
        self.tn = self.get_result_cnt(results, actual=0.0, predicted=0)

        self.fp = self.get_result_cnt(results, actual=0.0, predicted=1)
        self.fn = self.get_result_cnt(results, actual=1.0, predicted=0)

        self.total_measures = self.tp + self.tn + self.fp + self.fn

        self.label = label

    @staticmethod
    def get_result_cnt(results, actual, predicted):
        return len(filter(lambda pair: pair[0] == actual and pair[1] == predicted, results))

    def get_total_measures(self): return self.tp + self.tn + self.fp + self.fn

    def div(self, numerator, denominator):
        if denominator == 0:
            return 0

        return float(numerator) / denominator

    def get_accuracy(self):
        return self.div(self.tp + self.fn, self.get_total_measures())

    def get_precision(self):
        return self.div(self.tp, self.tp + self.fp)

    def get_recall(self):
        return self.div(self.tp, self.tp + self.fn)

    def get_specificity(self):
        return self.div(self.tn, self.tn+self.fp)

    def get_f1_score(self):
        return 2 * self.div(self.get_recall() * self.get_precision(), self.get_recall() + self.get_precision())

    def get_area_under_roc(self):
        return BinaryClassificationMetrics(get_df(self.results).rdd).areaUnderROC

    def to_numpy(self):
        return np.array(
            [self.get_accuracy(), self.get_precision(), self.get_recall(), self.get_specificity(), self.get_f1_score(),
             self.get_area_under_roc()])

    @staticmethod
    def print_numpy(np):
        return "[Accuracy: %f, Precision: %f, Recall: %f, Specificity: %f, F1: %f AuROC: %f]" % (
            np[0], np[1], np[2], np[3], np[4], np[5])

    def __str__(self):
        return "[%s][TP:%d TN:%d FP:%d FN:%d] [Acc: %f] [Prec: %f] [Recall: %f] [Specif: %f] [F1: %f] [AuROC:%f]" % (
            self.label, self.tp, self.tn, self.fp, self.fn, self.get_accuracy(), self.get_precision(), self.get_recall(),
            self.get_specificity(), self.get_f1_score(), self.get_area_under_roc())

    def to_json(self):
        return json.dumps({"accuracy": self.get_accuracy(), "precision": self.get_precision(), "recall": self.get_recall(),
                           "specificity": self.get_specificity(), "f1_score": self.get_f1_score(), "AuROC": self.get_area_under_roc()})


class SampleStats:
    def __init__(self, data):
        self.data = data

    def std(self):
        return np.std(np.array(self.data.map(lambda row: row.features.toArray()).collect()), axis=0)


def debug_samples(train, test):
    _train = train.collect()
    _test = test.collect()
    return "[TEST sz: %d, pos: %d, neg: %d] [TRAIN sz: %d, pos: %d, neg: %d]" % \
           ( len(_test), len(filter(lambda lp: lp.label == 1.0, _test)), len(filter(lambda lp: lp.label == 0.0, _test)),
             len(_train), len(filter(lambda lp: lp.label == 1.0, _train)), len(filter(lambda lp: lp.label == 0.0, _train)))