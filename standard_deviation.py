# env: graphsage_env

import re
import numpy as np

cancer_type = 'BLCA2'

def extract_values(filename):
    filepath = 'inputFilesGraphsage/stats/{0}_stats/{1}.txt'.format(cancer_type, filename)
    with open(filepath, 'r') as file:
        content = file.read()
        values = re.findall(r'\b\d+\.\d+\b', content)
        return list(map(float, values))

def calculate_standard_deviation():
    test_stats = []
    val_stats = []

    for i in range(10):
        test_stats.append(extract_values('test_stats{}'.format(i)))
        val_stats.append(extract_values('val_stats{}'.format(i)))

    # print("Test stats: {}".format(test_stats))  # testing
    # print("Validation stats: {}".format(val_stats))  # testing

    test_stats = np.array(test_stats)
    val_stats = np.array(val_stats)

    test_std = np.std(test_stats, axis=0)
    val_std = np.std(val_stats, axis=0)

    return test_std, val_std

test_std, val_std = calculate_standard_deviation()
print("Test Stats Standard Deviation:", test_std)
print("Validation Stats Standard Deviation:", val_std)