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
from pyspark.sql import *
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
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
from pyspark.mllib.evaluation import BinaryClassificationMetrics
	
def strings_cnt(file_path):
	printable_cnt = 0

	with open(file_path) as fd:
		while True:
			c = fd.read(1)
			if not c:
				break
			if c in string.printable:
				printable_cnt+=1

	return printable_cnt

def get_section_size(pe, section_name):
	for section in pe.sections:
		if section.Name.rstrip('\0') == section_name:
			return section.SizeOfRawData

# TODO: Use YARA rules to enrich features: [ is_packed, contains_base64, ... ]
def get_feature_header():
	return [ "strings_cnt", "import_cnt", "text_sz", "rdata_sz", "data_sz", "rsrc_sz" ]

def get_features(file_path):

	pe = pefile.PE(file_path, fast_load=False)

	data =  [ strings_cnt(file_path), len(pe.DIRECTORY_ENTRY_IMPORT) if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') else 0]
	for section in [ ".text", ".rdata", ".data", ".rsrc" ]:
		sz = get_section_size(pe, section)
		data.append(sz if sz != None else 0)

	return data

def get_dir_features(dir_path, label, overwrite=False):
	data = []
	fromPickle=False
	
	install_dir = os.path.dirname(os.path.realpath(__file__))
	pq_dir = install_dir+"/"+re.sub("[^A-Za-z0-9]", "", dir_path)+"-analysed"

	if os.path.isdir(pq_dir) and not overwrite:
		fromPickle = True
		ret = sc.pickleFile(pq_dir)
	else:
		for f in listdir(dir_path):
			fq = join(dir_path, f)
			if isfile(fq):
				features = get_features(fq)
				logging.debug("%s -> %s", fq, str(features))
				data.append(LabeledPoint(label, get_features(fq)))

			df = get_df(data)
			df.rdd.saveAsPickleFile(pq_dir)
			ret = df.rdd

	logging.info("Loaded %d samples from %s (Pre-processed: %s)", len(ret.collect()), dir_path, fromPickle)
	return ret

def standardise_features(labeled_rdd):
	# Drop Label
	rdd = labeled_rdd.map(lambda row: row[0])

	std = StandardScaler()
	model = std.fit(rdd)
	features_transform = model.transform(rdd)
	return features_transform

def remove_labels(labeled):
	return rdd.map(lambda row: row[0])

# TODO: How can we assert we're attaching the correct label? This relies solely on list ordering?
def add_labels(labeled, unlabeled):
	return zip(labeled.map(lambda row: row[1]).collect(), unlabeled.collect())

def get_df(data):
	return SparkSession.builder.getOrCreate().createDataFrame(data)

def get_sc():
	return SparkContext("local", "static-poc")

def debug_samples(train, test):
	_train = train.collect()
	_test = test.collect()
	return "[TEST sz: %d, pos: %d, neg: %d] [TRAIN sz: %d, pos: %d, neg: %d]" % \
		( len(_test), len(filter(lambda lp: lp.label == 1.0, _test)), len(filter(lambda lp: lp.label == 0.0, _test)),
			len(_train), len(filter(lambda lp: lp.label == 1.0, _train)), len(filter(lambda lp: lp.label == 0.0, _train)))

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
		return float(self.tp+self.tn)/self.get_total_measures() if self.tp+self.tn > 0 else 0

	def get_precision(self):
		return float(self.tp)/(self.tp+self.tn) if self.tp > 0 else 0

	def get_recall(self):
		return float(self.tp)/self.get_total_measures() if self.tp > 0 else 0

	def get_specificity(self):
		return float(self.tn)/self.get_total_measures() if self.tn > 0 else 0

	def get_f1_score(self):
		return 2*float(self.get_recall()+self.get_precision())/self.get_recall()+self.get_precision() \
			if self.get_recall() + self.get_precision() > 0 else 0

	def get_area_under_roc(self):
		return BinaryClassificationMetrics(get_df(self.results).rdd).areaUnderROC

	def to_numpy(self):
		return np.array([self.get_accuracy(), self.get_precision(), self.get_recall(), self.get_specificity(), self.get_f1_score(), self.get_area_under_roc()])

	@staticmethod
	def print_numpy(np):
		return "[Accuracy: %f, Precision: %f, Recall: %f, Specificity: %f, F1: %f AuROC: %f]" % (np[0], np[1], np[2], np[3], np[4], np[5])

	def __str__(self):
		return "[Label: %s][TP: %d TN: %d FP: %d FN: %d] [Accuracy: %f] [Precision: %f] [Recall: %f] [Specificity: %f] [F1 Score: %f]" % (self.label, self.tp, self.tn, self.fp, self.fn, self.get_accuracy(), self.get_precision(), self.get_recall(), self.get_specificity(), self.get_f1_score())

class SampleStats:
	def __init__(self, data):
		self.data = data

	def std(self):
		return np.std(np.array(self.data.map(lambda row: row.features.toArray()).collect()), axis=0)

def train_classifier_and_measure(ctype, training_data, test_data):
	if ctype == "svm":
		model = SVMWithSGD.train(training_data, iterations=100)
	elif ctype == "rf":
		model = DecisionTree.trainClassifier(training_data, 2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5, maxBins=32)

	output = []
	for lp in test_data.collect():
		output.append((lp.label, float(model.predict(lp.features))))

	return output

# UGLY: rewrite
def align_training_data(data):

	if data.count() % 2 != 0:
		data = get_df(data.take(data.count()-1)).rdd

	pos_rdd = data.filter(lambda lp: lp.label == 1.0)
	neg_rdd = data.filter(lambda lp: lp.label == 0.0)

	if pos_rdd.count() > neg_rdd.count():
		delta = pos_rdd.count() - neg_rdd.count()
		return neg_rdd.union(get_df(pos_rdd.take(pos_rdd.count()-delta)).rdd).map(lambda row: LabeledPoint(row.label, row.features))
	else:
		delta = neg_rdd.count() - pos_rdd.count()
		return pos_rdd.union(get_df(neg_rdd.take(neg_rdd.count()-delta)).rdd).map(lambda row: LabeledPoint(row.label, row.features))
	
def main():
	
	d = get_dir_features("./benign/", 0.0).union(get_dir_features("./malware/", 1.0))
	sample_size = 0.1 if len(sys.argv) == 1 else float(sys.argv[1])
	logging.info("Using sample size: %f", sample_size)

	if sample_size != 1:
		i = d.sample(True, sample_size)
	else:
		i = d
	
	# Visualise feature values
	#print("std dev:\n%s\n%s\n" %(get_feature_header(), SampleStats(i).std()))

	# Standardise feature values
	std = standardise_features(i)

	# Convert to RDD for classifier train function
	rdd = get_df(add_labels(i, std))
	stuff = rdd.rdd.map(lambda row: LabeledPoint(row[0], row[1]))

	for m in [ "svm", "rf" ]:
		# Generate random split for test / train
		#	test, train = stuff.randomSplit(weights=[0.3, 0.7], seed=int(time.time()))

		# K-folds (n=10)
		split_n = 10
		splits = stuff.randomSplit([float(1)/split_n]*split_n, seed=int(time.time()))
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
	#np.set_printoptions(suppress=True)
	sc = get_sc()
	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
	main()
