import numpy as np

def confusion_matrix(y_true, y_pred):
    """
    Computes the confusion matrix.
    """
    classes = np.unique(y_true)
    cm = np.zeros((len(classes), len(classes)))
    for i in range(len(classes)):
        for j in range(len(classes)):
           cm[i, j] = np.sum((y_true == classes[i]) & (y_pred == classes[j]))
    return cm

def accuracy_score(y_true, y_pred):
    """
    Computes the accuracy score.
    """
    cm = confusion_matrix(y_true, y_pred)
    return np.sum(np.diag(cm)) / np.sum(cm)

def precision_score(y_true, y_pred):
    """
    Computes the precision score.
    """
    cm = confusion_matrix(y_true, y_pred)
    return np.mean(np.diag(cm) / np.sum(cm, axis=0))

def recall_score(y_true, y_pred):
    """
    Computes the recall score.
    """
    cm = confusion_matrix(y_true, y_pred)
    return np.mean(np.diag(cm) / np.sum(cm, axis=1))
