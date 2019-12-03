# Copyright 2019 snactclass software
# Author: Emille E. O. Ishida
#         Based on initial prototype developed by the CRP #4 team
#
# created on 10 August 2019
# modified by Marco Leoni on 4 November 2019
# Licensed GNU General Public License v3.0;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ['random_forest_spark']

import numpy as np
from pyspark.ml import Pipeline
from sklearn import svm
from sklearn.datasets import dump_svmlight_file
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext


def random_forest_spark(train_features:  np.array, train_labels: np.array,
                        test_features: np.array, nest=1000, seed=42):
    """Random Forest classifier.

    Parameters
    ----------
    train_features: np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    nest: int (optional)
        Number of estimators (trees) in the forest.
        Default is 1000.
    seed: float (optional)
        Seed for random number generator. Default is 42.

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    importances: np.array
        Features importance leading the classification.
    """

    dump_svmlight_file(train_features, train_labels, 'tmp.txt',
                       ze.ro_based=False)
    example = spark.read.format("libsvm").load("tmp.txt")
    dump_svmlight_file(test_features, np.zeros(len(test_features)),
                       'tmp_test.txt', zero_based=False)
    example_test = spark.read.format("libsvm").load("tmp_test.txt")

    labelIndexer = StringIndexer(inputCol="label",
                                 outputCol="indexedLabel").fit(example)
    featureIndexer = VectorIndexer(inputCol="features",
                                   outputCol="indexedFeatures",
                                   maxCategories=100).fit(example)
    clf = RandomForestClassifier(labelCol="indexedLabel",
                                 featuresCol="indexedFeatures",
                                 numTrees=nest, seed=seed)
    labelConverter = IndexToString(inputCol="prediction",
                                   outputCol="predictedLabel",
                                   labels=labelIndexer.labels)
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer,
                                clf, labelConverter])

    model = pipeline.fit(example)
    predictions = model.transform(example_test)

    feature_importances = model.stages[2].featureImportances
    importances = feature_importances.toArray()

    pred = np.concatenate(np.array(np.array(
                            predictions.select("predictedLabel").collect(),
                            dtype=float), dtype=int), axis=0)

    prob = 1-np.concatenate(np.array(
            predictions.select("probability").collect()), axis=0)

    return pred, prob, importances


def main():
    return None


if __name__ == '__main__':
    main()
