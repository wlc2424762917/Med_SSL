from collections import OrderedDict
import glob
import os
import re
import json
from batchgenerators.utilities.file_and_folder_operations import *
import random


def list_sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [tryint(c) for c in re.split('([0-9]+)', s) ]
    l.sort(key=alphanum_key)
    return l


# path_originalData = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\acdc_seg_sr"
# path_originalData = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_LGE"
path_originalData = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_LGE_paired"

shuffle = False

A_image = list_sort_nicely(glob.glob(path_originalData+"/A/*"))
B_image = list_sort_nicely(glob.glob(path_originalData+"/B/*"))
AB_image = list_sort_nicely(glob.glob(path_originalData+"/AB/*"))

infarct_label = list_sort_nicely(glob.glob(path_originalData+"/label_segTr/*"))
myo_label = list_sort_nicely(glob.glob(path_originalData+"/labelsTs/*"))

A_image = ["{}".format(item.split('\\')[-1]) for item in A_image]
B_image = ["{}".format(item.split('\\')[-1]) for item in B_image]
AB_image = ["{}".format(item.split('\\')[-1]) for item in AB_image]
infarct_label = ["{}".format(item.split('\\')[-1]) for item in infarct_label]
myo_label = ["{}".format(item.split('\\')[-1]) for item in myo_label]

if shuffle:
    random.seed(24)
    random.shuffle(A_image)
    random.seed(24)
    random.shuffle(infarct_label)
    random.seed(32)
    random.shuffle(B_image)
    random.seed(32)
    random.shuffle(myo_label)

test_image = []
json_dict = OrderedDict()
json_dict['name'] = "LGE_DTI_aligned"
json_dict['description'] = " "
json_dict['tensorImageSize'] = "2D"
json_dict['reference'] = "www"
json_dict['licence'] = "www"
json_dict['release'] = "0.0"
json_dict['modality'] = {
    "0": "MR",
    "A": "LGE",
    "B": "DTI",
}
json_dict['labels'] = {
    "0": "background",
    "1": "infarction or myo",

}
json_dict['numTraining'] = len(A_image)
json_dict['numTest'] = len(test_image)
json_dict['training'] = [{'A': "./A/%s" % i, "B": "./B/%s" % j, "AB": "./AB/%s" % i, "myo_label": "./infarct_gt/%s" % j, "infarct_label": "./infarct_gt/%s" % i} for i, j in zip(A_image, B_image)]
json_dict['test'] = ["./imagesTs/%s" % i for i in test_image]
json_dict['validation'] = ["./imagesTs/%s" % i for i in test_image]

if shuffle:
    save_json(json_dict, join(path_originalData, "dataset_shuffle.json"))
else:
    save_json(json_dict, join(path_originalData, "dataset.json"))
