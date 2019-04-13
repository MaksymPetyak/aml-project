'''
This will analyze the distribution of uncertainty.
'''

import sys
import numpy as np
import pickle

def load_stochastic_predictions(filename):
    with open(filename, 'rb') as f:
        probs_mc = pickle.load(f)['stoch_out']
        assert ((0.0 <= probs_mc) & (probs_mc <= 1.0 + 1e-6)).all()
        return probs_mc

def binary_probs(probs):
    assert probs.shape[1] == 2
    return np.squeeze(probs[:, 1:])

def posterior_statistics(probs_mc_bin):
    predictive_mean = probs_mc_bin.mean(axis=1)
    predictive_std = probs_mc_bin.std(axis=1)
    assert (0.0 <= predictive_std).all()
    return predictive_mean, predictive_std

def main(pickle_file):
    probs_mc = load_stochastic_predictions(pickle_file)
    probs_mc_bin = binary_probs(probs_mc)
    pred_mean, pred_std = posterior_statistics(probs_mc_bin)
    uncertainties = list(pred_std)
    above = dict()
    for x in uncertainties:
        threshold_key = 5
        while x*100 >= threshold_key:
            above[threshold_key] = above.get(threshold_key, 0) + 1
            threshold_key += 5
    print('Average uncertainty: %.2f' % (sum(uncertainties) / len(uncertainties)))
    print()
    print('Distribution:')
    for k, v in above.items():
        print('%.2f%% are above %.2f' % (100*v/len(uncertainties), k/100.0))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage:')
        print('  python analyze_uncertainty_dist.py <predictions.pkl>')
        exit()
    pickle_file = sys.argv[1]
    main(pickle_file)
