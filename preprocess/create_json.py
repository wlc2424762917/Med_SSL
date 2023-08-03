from collections import OrderedDict
import glob
import os
import re
import json
from batchgenerators.utilities.file_and_folder_operations import *


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
# path_originalData = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\LGE_post_p"
path_originalData = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_post_paired"

train_image = list_sort_nicely(glob.glob(path_originalData+"/imagesTr/*"))
train_label = list_sort_nicely(glob.glob(path_originalData+"/label_segTr/*"))
test_image = list_sort_nicely(glob.glob(path_originalData+"/imagesTs/*"))
test_label = list_sort_nicely(glob.glob(path_originalData+"/labelsTs/*"))

train_image = ["{}".format(item.split('\\')[-1]) for item in train_image]
train_label = ["{}".format(item.split('\\')[-1]) for item in train_label]
test_image = ["{}".format(item.split('\\')[-1]) for item in test_image]
test_label = ["{}".format(item.split('\\')[-1]) for item in test_label]

print(train_image)
print(train_label)
print(test_image)
print(test_label)

json_dict = OrderedDict()
json_dict['name'] = "LGE"
json_dict['description'] = " "
json_dict['tensorImageSize'] = "2D"
json_dict['reference'] = "www"
json_dict['licence'] = "www"
json_dict['release'] = "0.0"
json_dict['modality'] = {
    "0": "MR",

}
json_dict['labels'] = {
    "0": "background",
    "1": "infarction",

}
json_dict['numTraining'] = len(train_image)
json_dict['numTest'] = len(test_image)
json_dict['training'] = [{'image': "./imagesTr/%s" % i, "label_seg": "./label_segTr/%s" % i, "label_sr": "./label_srTr/%s" % i} for i in train_label]
json_dict['test'] = ["./imagesTs/%s" % i for i in test_image]
json_dict['validation'] = ["./imagesTs/%s" % i for i in test_image]

save_json(json_dict, join(path_originalData, "dataset_0.json"))
