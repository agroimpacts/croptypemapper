import math
import numpy as np
import pandas as pd
from sklearn import metrics


class BinaryMetrics:
    """Metrics measuring model performance."""

    def __init__(self, ref_array, score_array, pred_array=None):
        """
        Params:
            ref_array (ndarray) -- Array of reference (e.g. label)
            score_array (ndarray) -- Array of pixel scores of positive class
            pred_array (ndarray) -- Array of hardened scores of positive class
        """

        self.observation = ref_array.flatten()
        self.score = score_array.flatten()

        if self.observation.shape != self.score.shape:
            raise Exception("Inconsistent size between label and prediction arrays.")

        if pred_array is not None:
            self.prediction = pred_array.flatten()
        else:
            self.prediction = np.where(self.score > 0.5, 1, 0)

        self.confusion_matrix = self.confusion_matrix()

    def __add__(self, other):
        """
        Add two BinaryMetrics instances
        Params:
            other (''BinaryMetrics''): A BinaryMetrics instance
        Return:
            ''BinaryMetrics''
        """

        return BinaryMetrics(np.append(self.observation, other.observation),
                             np.append(self.score, other.score),
                             np.append(self.prediction, other.prediction))

    def __radd__(self, other):
        """
        Add a BinaryMetrics instance with reversed operands.
        Params:
            other
        Returns:
            ''BinaryMetrics
        """

        if other == 0:
            return self
        else:
            return self.__add__(other)

    def confusion_matrix(self):
        """
        Calculate confusion matrix of given ground truth and predicted label
        Returns:
            ''pandas.dataframe'' of observation on the column and prediction on the row
        """

        # set_trace()
        refArray = self.observation
        predArray = self.prediction

        if (refArray.max() > 1) or (predArray.max() > 1):
            raise Exception("Invalid array")

        predArray = predArray * 2
        sub = refArray - predArray

        self.tp = np.sum(sub == -1)
        self.fp = np.sum(sub == -2)
        self.fn = np.sum(sub == 1)
        self.tn = np.sum(sub == 0)

        confusionMatrix = pd.DataFrame(data=np.array([[self.tn, self.fp], [self.fn, self.tp]]),
                                       index=['observation = 0', 'observation = 1'],
                                       columns=['prediction = 0', 'prediction = 1'])

        return confusionMatrix

    def ir(self):
        """
        Imbalance Ratio (IR) is defined as the proportion between positive and negative instances of the label.
        This value lies within the [0, ∞] range, having a value IR = 1 in the balanced case.
        Returns:
             float
        """
        try:
            ir = (self.tp + self.fn) / (self.fp + self.tn)

        except ZeroDivisionError:
            ir = np.nan_to_num(float("NaN"))

        return ir

    def oa(self):
        """
        Calculate Overall Accuracy.
        Returns:
            float
        """

        oa = metrics.accuracy_score(self.observation, self.prediction)

        return oa

    def producers_accuracy(self):
        """
        Calculate Producer's Accuracy (True Positive Rate |Sensitivity |hit rate | recall).
        Returns:
            float
        """

        return metrics.recall_score(self.observation, self.prediction)

    def users_accuracy(self):
        """
        Calculate User’s Accuracy (Positive Prediction Value (PPV) | Precision).
        Returns:
            float
        """

        ua = metrics.precision_score(self.observation, self.prediction)

        return ua

    def npv(self):
        """
        Calculate Negative Predictive Value or true negative accuracy.
        Returns:
             float
        """

        try:
            npv = self.tn / (self.tn + self.fn)

        except ZeroDivisionError:
            npv = np.nan_to_num(float("NaN"))

        return npv

    def specificity(self):
        """
        Calculate Specificity aka. True negative rate (TNR), or inverse recall.
        Returns:
             float
        """
        try:
            spc = self.tn / (self.tn + self.fp)

        except ZeroDivisionError:
            spc = np.nan_to_num(float("NaN"))

        return spc

    def f1_measure(self):
        """
        Calculate F1 score.
        Returns:
            float
        """

        f1 = metrics.f1_score(self.observation, self.prediction)

        return f1

    def iou(self):
        """
        Calculate interception over union for the positive class.
        Returns:
            float
        """

        return metrics.jaccard_score(self.observation, self.prediction)

    def miou(self):
        """
        Calculate mean interception over union considering both positive and negative classes.
        Returns:
            float
        """
        try:
            miou = np.nanmean([self.tn / (self.tn + self.fn + self.fp), self.tp / (self.tp + self.fn + self.fp)])

        except ZeroDivisionError:
            miou = np.nan_to_num(float("NaN"))

        return miou

    def tss(self):
        """
        Calculates true scale statistic (TSS). Also called Bookmaker Informedness (BM).
        Scale of the metric:[-1,1].
        Returns:
            float
        """
        tss = self.tp / (self.tp + self.fn) + self.tn / (self.tn + self.fp) - 1

        return tss
