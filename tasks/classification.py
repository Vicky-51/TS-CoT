import numpy as np
from . import _eval_protocols as eval_protocols
from utils import *


def eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='mlp', args = None):

    train_repr = model.encode(train_data)
    test_repr = model.encode(test_data)

    repr_results = {}
    repr_results['train_repr'] = train_repr
    repr_results['test_repr'] = test_repr
    repr_results['train_labels'] = train_labels
    repr_results['test_labels'] = test_labels
    if eval_protocol == 'mlp':
        y_score, metricss = eval_protocols.fit_mlp(train_repr, train_labels, test_repr, test_labels)
        return y_score, metricss
    else:
        assert False, 'unknown evaluation protocol'
