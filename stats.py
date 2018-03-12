from pyspark.sql import *
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.mllib.evaluation import BinaryClassificationMetrics
import numpy as np

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

    def get_accuracy(self):
        return float(self.tp + self.tn) / self.get_total_measures() if self.tp + self.tn > 0 else 0

    def get_precision(self):
        return float(self.tp) / (self.tp + self.tn) if self.tp > 0 else 0

    def get_recall(self):
        return float(self.tp) / self.get_total_measures() if self.tp > 0 else 0

    def get_specificity(self):
        return float(self.tn) / self.get_total_measures() if self.tn > 0 else 0

    def get_f1_score(self):
        return 2 * float(self.get_recall() + self.get_precision()) / self.get_recall() + self.get_precision() \
            if self.get_recall() + self.get_precision() > 0 else 0

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
        return "[Label: %s][TP: %d TN: %d FP: %d FN: %d] [Accuracy: %f] [Precision: %f] [Recall: %f] [Specificity: %f] [F1 Score: %f]" % (
        self.label, self.tp, self.tn, self.fp, self.fn, self.get_accuracy(), self.get_precision(), self.get_recall(),
        self.get_specificity(), self.get_f1_score())

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