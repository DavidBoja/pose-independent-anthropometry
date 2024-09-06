

import torch
import smplx
import os
from datetime import datetime
import yaml
import re
import numpy as np
import open3d as o3d
import tempfile
import gzip
from glob import glob
import pickle
import json
from typing import List
import random
import trimesh
import torch.nn.functional as F

##############################################################################################################
# MEASUREMENT UTILS
##############################################################################################################

CAESAR_LANDMARK_MAPPING =  {
     '10th Rib Midspine': '10th Rib Midspine',
     'AUX LAND': 'AUX LAND',
     'Butt Block': 'Butt Block',
     'Cervical': 'Cervicale', # FIXED
     'Cervicale': 'Cervicale', 
     'Crotch': 'Crotch',
     'Lt. 10th Rib': 'Lt. 10th Rib',
     'Lt. ASIS': 'Lt. ASIS',
     'Lt. Acromio': 'Lt. Acromion', # FIXED
     'Lt. Acromion': 'Lt. Acromion',
     'Lt. Axilla, An': 'Lt. Axilla, Ant.', # FIXED
     'Lt. Axilla, Ant': 'Lt. Axilla, Ant.', # FIXED
     'Lt. Axilla, Post': 'Lt. Axilla, Post.', # FIXED
     'Lt. Axilla, Post.': 'Lt. Axilla, Post.',
     'Lt. Calcaneous, Post.': 'Lt. Calcaneous, Post.', 
     'Lt. Clavicale': 'Lt. Clavicale',
     'Lt. Dactylion': 'Lt. Dactylion',
     'Lt. Digit II': 'Lt. Digit II',
     'Lt. Femoral Lateral Epicn': 'Lt. Femoral Lateral Epicn',
     'Lt. Femoral Lateral Epicn ': 'Lt. Femoral Lateral Epicn', # FIXED
     'Lt. Femoral Medial Epicn': 'Lt. Femoral Medial Epicn',
     'Lt. Gonion': 'Lt. Gonion',
     'Lt. Humeral Lateral Epicn': 'Lt. Humeral Lateral Epicn',
     'Lt. Humeral Medial Epicn': 'Lt. Humeral Medial Epicn',
     'Lt. Iliocristale': 'Lt. Iliocristale',
     'Lt. Infraorbitale': 'Lt. Infraorbitale',
     'Lt. Knee Crease': 'Lt. Knee Crease',
     'Lt. Lateral Malleolus': 'Lt. Lateral Malleolus',
     'Lt. Medial Malleolu': 'Lt. Medial Malleolus', # FIXED
     'Lt. Medial Malleolus': 'Lt. Medial Malleolus',
     'Lt. Metacarpal-Phal. II': 'Lt. Metacarpal Phal. II', # FIXED
     'Lt. Metacarpal-Phal. V': 'Lt. Metacarpal Phal. V', # FIXED
     'Lt. Metatarsal-Phal. I': 'Lt. Metatarsal Phal. I', # FIXED
     'Lt. Metatarsal-Phal. V': 'Lt. Metatarsal Phal. V', # FIXED
     'Lt. Olecranon': 'Lt. Olecranon',
     'Lt. PSIS': 'Lt. PSIS',
     'Lt. Radial Styloid': 'Lt. Radial Styloid',
     'Lt. Radiale': 'Lt. Radiale',
     'Lt. Sphyrio': 'Lt. Sphyrion', # FIXED
     'Lt. Sphyrion': 'Lt. Sphyrion',
     'Lt. Thelion/Bustpoin': 'Lt. Thelion/Bustpoint', # FIXED
     'Lt. Thelion/Bustpoint': 'Lt. Thelion/Bustpoint',
     'Lt. Tragion': 'Lt. Tragion',
     'Lt. Trochanterion': 'Lt. Trochanterion',
     'Lt. Ulnar Styloid': 'Lt. Ulnar Styloid',
     'Nuchale': 'Nuchale',
     'Rt. 10th Rib': 'Rt. 10th Rib',
     'Rt. ASIS': 'Rt. ASIS',
     'Rt. Acromio': 'Rt. Acromion', # FIXED
     'Rt. Acromion': 'Rt. Acromion',
     'Rt. Axilla, An': 'Rt. Axilla, Ant.', # FIXED
     'Rt. Axilla, Ant': 'Rt. Axilla, Ant.', # FIXED
     'Rt. Axilla, Post': 'Rt. Axilla, Post.', # FIXED
     'Rt. Axilla, Post.': 'Rt. Axilla, Post.',
     'Rt. Calcaneous, Post.': 'Rt. Calcaneous, Post.',
     'Rt. Clavicale': 'Rt. Clavicale',
     'Rt. Dactylion': 'Rt. Dactylion',
     'Rt. Digit II': 'Rt. Digit II',
     'Rt. Femoral Lateral Epicn': 'Rt. Femoral Lateral Epicn',
     'Rt. Femoral Lateral Epicn ': 'Rt. Femoral Lateral Epicn', # FIXED
     'Rt. Femoral Medial Epic': 'Rt. Femoral Medial Epicn', # FIXED
     'Rt. Femoral Medial Epicn': 'Rt. Femoral Medial Epicn',
     'Rt. Gonion': 'Rt. Gonion',
     'Rt. Humeral Lateral Epicn': 'Rt. Humeral Lateral Epicn',
     'Rt. Humeral Medial Epicn': 'Rt. Humeral Medial Epicn',
     'Rt. Iliocristale': 'Rt. Iliocristale',
     'Rt. Infraorbitale': 'Rt. Infraorbitale',
     'Rt. Knee Creas': 'Rt. Knee Crease', # FIXED
     'Rt. Knee Crease': 'Rt. Knee Crease',
     'Rt. Lateral Malleolus': 'Rt. Lateral Malleolus',
     'Rt. Medial Malleolu': 'Rt. Medial Malleolus', # FIXED
     'Rt. Medial Malleolus': 'Rt. Medial Malleolus',
     'Rt. Metacarpal Phal. II': 'Rt. Metacarpal Phal. II',
     'Rt. Metacarpal-Phal. V': 'Rt. Metacarpal Phal. V', # FIXED
     'Rt. Metatarsal-Phal. I': 'Rt. Metatarsal Phal. I', # FIXED
     'Rt. Metatarsal-Phal. V': 'Rt. Metatarsal Phal. V', # FIXED
     'Rt. Olecranon': 'Rt. Olecranon',
     'Rt. PSIS': 'Rt. PSIS',
     'Rt. Radial Styloid': 'Rt. Radial Styloid',
     'Rt. Radiale': 'Rt. Radiale',
     'Rt. Sphyrio': 'Rt. Sphyrion', # FIXED
     'Rt. Sphyrion': 'Rt. Sphyrion',
     'Rt. Thelion/Bustpoin': 'Rt. Thelion/Bustpoint', # FIXED
     'Rt. Thelion/Bustpoint': 'Rt. Thelion/Bustpoint',
     'Rt. Tragion': 'Rt. Tragion',
     'Rt. Trochanterion': 'Rt. Trochanterion',
     'Rt. Ulnar Styloid': 'Rt. Ulnar Styloid',
     'Sellion': 'Sellion',
     'Substernale': 'Substernale',
     'Supramenton': 'Supramenton',
     'Suprasternale': 'Suprasternale',
     'Waist, Preferred, Post.': 'Waist, Preferred, Post.'
    }


SMPL_INDEX_LANDMARKS = {
     '10th Rib Midspine': 3024,
    #  'AUX LAND': 0, ## ??
    #  'Butt Block': 0, ## ??
     'Cervicale': 828,
     'Crotch': 1210, ## ??
     'Lt. 10th Rib': 1481,
     'Lt. ASIS': 3157,
     'Lt. Acromion': 636,
     'Lt. Axilla, Ant.': 772,
     'Lt. Axilla, Post.': 1431,
     'Lt. Calcaneous, Post.': 3387,
     'Lt. Clavicale': 700,
     'Lt. Dactylion': 2446,
     'Lt. Digit II': 3302,
     'Lt. Femoral Lateral Epicn': 1053,
     'Lt. Femoral Medial Epicn': 1059,
     'Lt. Gonion': 147,
     'Lt. Humeral Lateral Epicn': 1738,
     'Lt. Humeral Medial Epicn': 1627,
     'Lt. Iliocristale': 654,
     'Lt. Infraorbitale': 357,
     'Lt. Knee Crease': 1050,
     'Lt. Lateral Malleolus': 3327,
     'Lt. Medial Malleolus': 3432,
     'Lt. Metacarpal Phal. II': 2135,
     'Lt. Metacarpal Phal. V': 2628,
     'Lt. Metatarsal Phal. I': 3232,
     'Lt. Metatarsal Phal. V': 3283,
     'Lt. Olecranon': 1643,
     'Lt. PSIS': 3098,
     'Lt. Radial Styloid': 2110,
     'Lt. Radiale': 1701,
     'Lt. Sphyrion': 3417,
     'Lt. Thelion/Bustpoint': 670,
     'Lt. Tragion': 448,
     'Lt. Trochanterion': 1454,
     'Lt. Ulnar Styloid': 2108,
     'Nuchale': 445,
     'Rt. 10th Rib': 4953,
     'Rt. ASIS': 6573,
     'Rt. Acromion': 4124,
     'Rt. Axilla, Ant.': 4874,
     'Rt. Axilla, Post.': 4906,
     'Rt. Calcaneous, Post.': 6786,
     'Rt. Clavicale': 4187,
     'Rt. Dactylion': 5907,
     'Rt. Digit II': 6702,
     'Rt. Femoral Lateral Epicn': 4538,
     'Rt. Femoral Medial Epicn': 4543,
     'Rt. Gonion': 3659,
     'Rt. Humeral Lateral Epicn': 5207,
     'Rt. Humeral Medial Epicn': 5097,
     'Rt. Iliocristale': 4424,
     'Rt. Infraorbitale': 3847,
     'Rt. Knee Crease': 4535,
     'Rt. Lateral Malleolus': 6728,
     'Rt. Medial Malleolus': 6832,
     'Rt. Metacarpal Phal. II': 5595,
     'Rt. Metacarpal Phal. V': 6089,
     'Rt. Metatarsal Phal. I': 6634,
     'Rt. Metatarsal Phal. V': 6684,
     'Rt. Olecranon': 5112,
     'Rt. PSIS': 6523,
     'Rt. Radial Styloid': 5480,
     'Rt. Radiale': 5170,
     'Rt. Sphyrion': 6817,
     'Rt. Thelion/Bustpoint': 4158,
     'Rt. Tragion': 3941,
     'Rt. Trochanterion': 4927,
     'Rt. Ulnar Styloid': 5520,
     'Sellion': 410,
     'Substernale': 1330,
     'Supramenton': 3051,
     'Suprasternale': 3073,
     'Waist, Preferred, Post.': 3159
    }

# revised after comparing original and transferred landmarks
# analysed the closest SMPL indices of the fitted NRD body
# to every original landmark and manually corrected the mapping
# removing Rib landmarks cause they are predominantly missing
SMPL_INDEX_LANDAMRKS_REVISED = {
#                                      '10th Rib Midspine': 3024,
                                     'Cervicale': 829, 
                                     'Crotch': 1353, 
#                                      'Lt. 10th Rib': 1481, 
                                     'Lt. ASIS': 3157, 
                                     'Lt. Acromion': 1862, 
                                     'Lt. Axilla, Ant.': 1871, 
                                     'Lt. Axilla, Post.': 2991, 
                                     'Lt. Calcaneous, Post.': 3387,
                                     'Lt. Clavicale': 1300,
                                     'Lt. Dactylion': 2446,
                                     'Lt. Digit II': 3222,
                                     'Lt. Femoral Lateral Epicn': 1008,
                                     'Lt. Femoral Medial Epicn': 1016,
                                     'Lt. Gonion': 148,
                                     'Lt. Humeral Lateral Epicn': 1621,
                                     'Lt. Humeral Medial Epicn': 1661,
                                     'Lt. Iliocristale': 677,
                                     'Lt. Infraorbitale': 341,
                                     'Lt. Knee Crease': 1050,
                                     'Lt. Lateral Malleolus': 3327,
                                     'Lt. Medial Malleolus': 3432,
                                     'Lt. Metacarpal Phal. II': 2258,
                                     'Lt. Metacarpal Phal. V': 2082,
                                     'Lt. Metatarsal Phal. I': 3294,
                                     'Lt. Metatarsal Phal. V': 3348,
                                     'Lt. Olecranon': 1736,
                                     'Lt. PSIS': 3097,
                                     'Lt. Radial Styloid': 2112,
                                     'Lt. Radiale': 1700,
                                     'Lt. Sphyrion': 3417,
                                     'Lt. Thelion/Bustpoint': 598,
                                     'Lt. Tragion': 448,
                                     'Lt. Trochanterion': 808,
                                     'Lt. Ulnar Styloid': 2108,
                                     'Nuchale': 445,
#                                      'Rt. 10th Rib': 4953,
                                     'Rt. ASIS': 6573,
                                     'Rt. Acromion': 5342,
                                     'Rt. Axilla, Ant.': 5332,
                                     'Rt. Axilla, Post.': 6450,
                                     'Rt. Calcaneous, Post.': 6786,
                                     'Rt. Clavicale': 4782,
                                     'Rt. Dactylion': 5907,
                                     'Rt. Digit II': 6620,
                                     'Rt. Femoral Lateral Epicn': 4493,
                                     'Rt. Femoral Medial Epicn': 4500,
                                     'Rt. Gonion': 3661,
                                     'Rt. Humeral Lateral Epicn': 5090,
                                     'Rt. Humeral Medial Epicn': 5131,
                                     'Rt. Iliocristale': 4165,
                                     'Rt. Infraorbitale': 3847,
                                     'Rt. Knee Crease': 4535,
                                     'Rt. Lateral Malleolus': 6728,
                                     'Rt. Medial Malleolus': 6832,
                                     'Rt. Metacarpal Phal. II': 5578,
                                     'Rt. Metacarpal Phal. V': 5545,
                                     'Rt. Metatarsal Phal. I': 6694,
                                     'Rt. Metatarsal Phal. V': 6715,
                                     'Rt. Olecranon': 5205,
                                     'Rt. PSIS': 6521,
                                     'Rt. Radial Styloid': 5534,
                                     'Rt. Radiale': 5170,
                                     'Rt. Sphyrion': 6817,
                                     'Rt. Thelion/Bustpoint': 4086,
                                     'Rt. Tragion': 3941,
                                     'Rt. Trochanterion': 4310,
                                     'Rt. Ulnar Styloid': 5520,
                                     'Sellion': 410,
                                     'Substernale': 3079,
                                     'Supramenton': 3051,
                                     'Suprasternale': 3171,
                                     'Waist, Preferred, Post.': 3021}


SMPL_SIMPLE_LANDMARKS = {
    "HEAD_TOP": 412,
    "HEAD_LEFT_TEMPLE": 166,
    "NECK_ADAM_APPLE": 3050,
    "LEFT_HEEL": 3458,
    "RIGHT_HEEL": 6858,
    "LEFT_NIPPLE": 3042,
    "RIGHT_NIPPLE": 6489,

    "SHOULDER_TOP": 829,
    "INSEAM_POINT": 3149,
    "BELLY_BUTTON": 3501,
    "BACK_BELLY_BUTTON": 3022,
    "CROTCH": 1210,
    "PUBIC_BONE": 3145,
    "RIGHT_WRIST": 5559,
    "LEFT_WRIST": 2241,
    "RIGHT_BICEP": 4855,
    "RIGHT_FOREARM": 5197,
    "LEFT_SHOULDER": 3011,
    "RIGHT_SHOULDER": 6470,
    "LOW_LEFT_HIP": 3134,
    "LEFT_THIGH": 947,
    "LEFT_CALF": 1103,
    "LEFT_ANKLE": 3325,
    "LEFT_ELBOW": 1643,

    "BUTTHOLE":3119,

    # introduce CAESAR landmarks because
    # i need to measure arms in parts
    "Cervicale": 829,
    'Rt. Acromion': 5342,
    'Rt. Humeral Lateral Epicn': 5090,
    'Rt. Ulnar Styloid': 5520,
}


SMPL_SIMPLE_LANDMARKS["HEELS"] = (SMPL_SIMPLE_LANDMARKS["LEFT_HEEL"], 
                                  SMPL_SIMPLE_LANDMARKS["RIGHT_HEEL"])


SMPL_SIMPLE_MEASUREMENTS = {
    "waist circumference": {"measurement_type": "circumference",
                            "indices": [3500, 1336, 917, 916, 919, 918, 665, 662,
                                    657, 654, 631, 632, 720, 799, 796, 890, 889, 3124, 3018,
                                    3019, 3502, 6473, 6474, 6545, 4376, 4375, 4284, 4285, 4208,
                                    4120, 4121, 4142, 4143, 4150, 4151, 4406, 4405,
                                    4403, 4402, 4812, 3500]},
    "chest circumference": {"measurement_type": "circumference",
                            "indices": [3076, 2870, 1254, 1255, 1349, 1351, 3033,
                                        3030, 3037, 3034, 3039, 611, 2868, 2864, 2866, 
                                        1760, 1419, 741, 738, 759, 2957, 2907, 1435, 1436, 
                                        1437, 1252, 1235, 749, 752, 3015, 4238, 4237, 4718, 
                                        4735, 4736, 4909, 6366, 4249, 4250, 4225, 4130, 4131, 
                                        4895, 4683,4682,4099, 4898, 4894, 4156, 4159, 4086, 
                                        4089, 4174, 4172, 4179, 6332,3076]},
    "hip circumference": {"measurement_type": "circumference",
                          "indices": [1806, 1805, 1804, 1803, 1802, 1801, 1800, 1798,
                                    1797, 1796, 1794, 1791, 1788, 1787, 3101, 3114, 3121, 3098,
                                    3099, 3159, 6522, 6523, 6542, 6537, 6525, 5252, 5251, 5255,
                                    5256, 5258, 5260, 5261, 5264, 5263, 5266, 5265, 5268, 5267,
                                    1806]},
    "thigh left circumference": {"measurement_type": "circumference",
                                 "indices": [877, 874, 873, 848, 849, 902, 851, 852, 897,
                                         900, 933, 936, 1359, 963, 908, 911, 1366,877]},
    "calf left circumference": {"measurement_type": "circumference",
                                "indices": [1154, 1372, 1074, 1077, 1470, 1094, 1095, 1473,
                                        1465, 1466, 1108, 1111, 1530, 1089, 1086, 1154]},
    "ankle left circumference": {"measurement_type": "circumference",
                                    "indices": [3322, 3323, 3190, 3188, 3185, 3206, 3182,
                                            3183, 3194, 3195, 3196, 3176, 3177, 3193, 3319, 3322]},
    "wrist left circumference": {"measurement_type": "circumference",
                                "indices": [1922, 1970, 1969, 2244, 1945, 1943, 1979,
                                        1938, 1935, 2286, 2243, 2242, 1930, 1927, 1926, 1924, 1922]},
    "forearm left circumference": {"measurement_type": "circumference",
                                "indices": [1573, 1915, 1914, 1577, 1576, 1912, 1911,
                                            1624, 1625, 1917, 1611, 1610, 1607, 1608, 1916, 1574, 1573]},
    "bicep left circumference": {"measurement_type": "circumference",
                                "indices": [789, 1311, 1315, 1379, 1378, 1394, 1393,
                                            1389, 1388, 1233, 1232, 1385, 1381, 1382, 1397, 1396, 628, 627,789]},
    "neck circumference": {"measurement_type": "circumference",
                           "indices": [3068, 1331, 215, 216, 440, 441, 452, 218, 219,
                                    222, 425, 426, 453, 829, 3944, 3921, 3920, 3734, 3731, 3730,
                                    3943, 3935, 3934, 3728, 3729, 4807, 3068]},
    "head circumference": {"measurement_type": "circumference",
                           "indices": [335, 259, 133, 0, 3, 135, 136, 160, 161, 166,
                                    167, 269, 179, 182, 252, 253, 384, 3765, 3766, 3694, 3693,
                                    3782, 3681, 3678, 3671, 3672, 3648, 3647, 3513, 3512, 3646,
                                    3771, 335]},
    "height": {"measurement_type": "length",
               "landmark_names": ["HEAD_TOP", "HEELS"]},
    "shoulder to crotch height": {"measurement_type": "length",
                                 "landmark_names": ["SHOULDER_TOP", "INSEAM_POINT"]},
    "arm right length": {"measurement_type": "length",
                       "landmark_names": ["RIGHT_SHOULDER", "RIGHT_WRIST"]},
    "crotch height": {"measurement_type": "length",
               "landmark_names": ["INSEAM_POINT", "HEELS"]},
    "Hip circumference max height": {"measurement_type": "length",
                                     "landmark_names": ["LOW_LEFT_HIP", "HEELS"]},
    "arm length (shoulder to elbow)": {"measurement_type": "length",
                                       "landmark_names": ["LEFT_SHOULDER", "LEFT_ELBOW"]},
    "arm length (spine to wrist)": {"measurement_type": "length",
                                     "landmark_names": ["SHOULDER_TOP", "RIGHT_WRIST"]}
}


# revised after visualizing point importance for measurements on SMPL
SMPL_SIMPLE_MEASUREMENTS_REVISED = {
    "waist circumference": {"measurement_type": "circumference",
                            "indices": [3500, 1336, 917, 916, 919, 918, 665, 662,
                                        657, 654, 631, 632, 720, 799, 796, 890, 889, 3124, 3018,
                                        3019, 3502, 6473, 6474, 6545, 4376, 4375, 4284, 4285, 4208,
                                        4120, 4121, 4142, 4143, 4150, 4151, 4406, 4405,
                                        4403, 4402, 4812, 3500]},
    "chest circumference": {"measurement_type": "circumference",
                            "indices": [6310,4380,4379,6367,4213,4250,
                                        4225,4130,4131,4895,4683,4681,4676,4893,4157,
                                        4158,4087,4088,4175,4179,6331,3077,2871,691,685,600,601,670,
                                        671,1191,1190,1193,1194,1756,1423,645,642,736,764,723,2908,1253,892,
                                        2850,2849,3025,3505,6475,6311,6310]
                           },
    "hip circumference": {"measurement_type": "circumference",
                          "indices": [1807, 864, 863, 1205, 1204, 915, 1511, 1513, 932, 1454, 1446, 
                                      1447, 3084, 3136, 3137, 3138, 3116, 3117,3118, 3119, 6541, 
                                      6539, 6540, 6559, 6558, 6557, 6509, 4919, 4920, 4927, 4418, 4353,
                                      4983, 4923, 4690, 4692, 4351, 4350, 1807]},
    "thigh left circumference": {"measurement_type": "circumference",
                                 "indices": [877, 874, 873, 848, 849, 902, 851, 852, 897,
                                             900, 933, 936, 1359, 963, 908, 911, 1366,877]},
    "calf left circumference": {"measurement_type": "circumference",
                                "indices": [1154, 1372, 1074, 1077, 1470, 1094, 1095, 1473,
                                            1465, 1466, 1108, 1111, 1530, 1089, 1086, 1154]},
    "ankle left circumference": {"measurement_type": "circumference",
                                    "indices": [3325, 3326, 3208, 3207, 3204, 3205, 3202, 3203, 
                                                3200, 3201, 3210, 3198, 3199, 3209, 3324, 3325]},
    "wrist left circumference": {"measurement_type": "circumference",
                                "indices": [1922, 1970, 1969, 2244, 1945, 1943, 1979,
                                            1938, 1935, 2286, 2243, 2242, 1930, 1927, 1926, 1924, 1922]},
    "forearm left circumference": {"measurement_type": "circumference",
                                "indices": [1573, 1915, 1914, 1577, 1576, 1912, 1911,
                                            1624, 1625, 1917, 1611, 1610, 1607, 1608, 1916, 1574, 1573]},
    "bicep left circumference": {"measurement_type": "circumference",
                                "indices": [789, 1311, 1315, 1379, 1378, 1394, 1393,
                                            1389, 1388, 1233, 1232, 1385, 1381, 1382, 1397, 1396, 628, 627,789]},
    "neck circumference": {"measurement_type": "circumference",
                           "indices": [829, 3944, 3921, 3920, 3734, 3731, 4761, 4309, 4072, 4781, 4187, 6333, 3078, 2872, 700, 1299, 584, 821,
                                         1280, 219, 222, 425, 426, 453, 829]},
    "head circumference": {"measurement_type": "circumference",
                           "indices": [335, 259, 133, 0, 3, 135, 136, 160, 161, 166,
                                        167, 269, 179, 182, 252, 253, 384, 3765, 3766, 3694, 3693,
                                        3782, 3681, 3678, 3671, 3672, 3648, 3647, 3513, 3512, 3646,
                                        3771, 335]},
    "height": {"measurement_type": "length",
               "landmark_names": ["HEAD_TOP", "HEELS"]},
    "shoulder to crotch height": {"measurement_type": "length",
                                 "landmark_names": ["SHOULDER_TOP", "INSEAM_POINT"]},
    "arm right length": {"measurement_type": "length",
                       "landmark_names": ["RIGHT_SHOULDER", "RIGHT_WRIST"]},
    "crotch height": {"measurement_type": "length",
                       "landmark_names": ["INSEAM_POINT", "HEELS"]},
    "Hip circumference max height": {"measurement_type": "length",
                                     "landmark_names": ["BUTTHOLE", "HEELS"]},
    "arm length (shoulder to elbow)": {"measurement_type": "length",
                                       "landmark_names": ["LEFT_SHOULDER", "LEFT_ELBOW"]},
    "arm length (spine to wrist)": {"measurement_type": "length",
                                     "landmark_names": ["SHOULDER_TOP", "RIGHT_WRIST"]},
    "leg length": {"measurement_type": "length",
                   "landmark_names": ["LOW_LEFT_HIP", "LEFT_HEEL"]},                           
}


# revised arm lenght to be able to measure it in parts
SMPL_SIMPLE_MEASUREMENTS_REVISED2 = {
    "waist circumference": {"measurement_type": "circumference",
                            "indices": [3500, 1336, 917, 916, 919, 918, 665, 662,
                                        657, 654, 631, 632, 720, 799, 796, 890, 889, 3124, 3018,
                                        3019, 3502, 6473, 6474, 6545, 4376, 4375, 4284, 4285, 4208,
                                        4120, 4121, 4142, 4143, 4150, 4151, 4406, 4405,
                                        4403, 4402, 4812, 3500]},
    "chest circumference": {"measurement_type": "circumference",
                            "indices": [6310,4380,4379,6367,4213,4250,
                                        4225,4130,4131,4895,4683,4681,4676,4893,4157,
                                        4158,4087,4088,4175,4179,6331,3077,2871,691,685,600,601,670,
                                        671,1191,1190,1193,1194,1756,1423,645,642,736,764,723,2908,1253,892,
                                        2850,2849,3025,3505,6475,6311,6310]
                           },
    "hip circumference": {"measurement_type": "circumference",
                          "indices": [1807, 864, 863, 1205, 1204, 915, 1511, 1513, 932, 1454, 1446, 
                                      1447, 3084, 3136, 3137, 3138, 3116, 3117,3118, 3119, 6541, 
                                      6539, 6540, 6559, 6558, 6557, 6509, 4919, 4920, 4927, 4418, 4353,
                                      4983, 4923, 4690, 4692, 4351, 4350, 1807]},
    "thigh left circumference": {"measurement_type": "circumference",
                                 "indices": [877, 874, 873, 848, 849, 902, 851, 852, 897,
                                             900, 933, 936, 1359, 963, 908, 911, 1366,877]},
    "calf left circumference": {"measurement_type": "circumference",
                                "indices": [1154, 1372, 1074, 1077, 1470, 1094, 1095, 1473,
                                            1465, 1466, 1108, 1111, 1530, 1089, 1086, 1154]},
    "ankle left circumference": {"measurement_type": "circumference",
                                    "indices": [3325, 3326, 3208, 3207, 3204, 3205, 3202, 3203, 
                                                3200, 3201, 3210, 3198, 3199, 3209, 3324, 3325]},
    "wrist left circumference": {"measurement_type": "circumference",
                                "indices": [1922, 1970, 1969, 2244, 1945, 1943, 1979,
                                            1938, 1935, 2286, 2243, 2242, 1930, 1927, 1926, 1924, 1922]},
    "forearm left circumference": {"measurement_type": "circumference",
                                "indices": [1573, 1915, 1914, 1577, 1576, 1912, 1911,
                                            1624, 1625, 1917, 1611, 1610, 1607, 1608, 1916, 1574, 1573]},
    "bicep left circumference": {"measurement_type": "circumference",
                                "indices": [789, 1311, 1315, 1379, 1378, 1394, 1393,
                                            1389, 1388, 1233, 1232, 1385, 1381, 1382, 1397, 1396, 628, 627,789]},
    "neck circumference": {"measurement_type": "circumference",
                           "indices": [829, 3944, 3921, 3920, 3734, 3731, 4761, 4309, 4072, 4781, 4187, 6333, 3078, 2872, 700, 1299, 584, 821,
                                         1280, 219, 222, 425, 426, 453, 829]},
    "head circumference": {"measurement_type": "circumference",
                           "indices": [335, 259, 133, 0, 3, 135, 136, 160, 161, 166,
                                        167, 269, 179, 182, 252, 253, 384, 3765, 3766, 3694, 3693,
                                        3782, 3681, 3678, 3671, 3672, 3648, 3647, 3513, 3512, 3646,
                                        3771, 335]},
    "height": {"measurement_type": "length",
               "landmark_names": ["HEAD_TOP", "HEELS"]},
    "shoulder to crotch height": {"measurement_type": "length",
                                 "landmark_names": ["SHOULDER_TOP", "INSEAM_POINT"]},
    "arm right length": {"measurement_type": "length",
                       "landmark_names": ["Rt. Acromion", "Rt. Humeral Lateral Epicn", "Rt. Ulnar Styloid"]},
    "crotch height": {"measurement_type": "length",
                       "landmark_names": ["INSEAM_POINT", "HEELS"]},
    "Hip circumference max height": {"measurement_type": "length",
                                     "landmark_names": ["BUTTHOLE", "HEELS"]},
    "arm length (shoulder to elbow)": {"measurement_type": "length",
                                       "landmark_names": ["Rt. Acromion", "Rt. Humeral Lateral Epicn"]},
    "arm length (spine to wrist)": {"measurement_type": "length",
                                     "landmark_names": ["Cervicale", "Rt. Acromion",
                                                        "Rt. Humeral Lateral Epicn","Rt. Ulnar Styloid"]},
    "leg length": {"measurement_type": "length",
                   "landmark_names": ["LOW_LEFT_HIP", "LEFT_HEEL"]},                 
}


SMPL_SIMPLE2CAESAR_MEASUREMENTS_MAP = {'ankle left circumference': 'Ankle Circumference (mm)',
                                        'arm length (shoulder to elbow)': 'Arm Length (Shoulder to Elbow) (mm)',
                                        'arm right length': 'Arm Length (Shoulder to Wrist) (mm)',
                                        'arm length (spine to wrist)': 'Arm Length (Spine to Wrist) (mm)',
                                        'chest circumference': 'Chest Circumference (mm)',
                                        'crotch height': 'Crotch Height (mm)',
                                        'head circumference': 'Head Circumference (mm)',
                                        'Hip circumference max height': 'Hip Circ Max Height (mm)',
                                        'hip circumference': 'Hip Circumference, Maximum (mm)',
                                        'neck circumference': 'Neck Base Circumference (mm)',
                                        'height': 'Stature (mm)'}


TSOLI_MALE_ERRORS_MM = dict(zip(["Ankle Circumference (mm)" ,
                                "Arm Length (Shoulder to Elbow) (mm)",
                                "Arm Length (Shoulder to Wrist) (mm)",
                                "Arm Length (Spine to Wrist) (mm)",
                                "Chest Circumference (mm)",
                                "Crotch Height (mm)",
                                "Head Circumference (mm)",
                                "Hip Circ Max Height (mm)",
                                "Hip Circumference, Maximum (mm)",
                                "Neck Base Circumference (mm)",
                                "Stature (mm)"],
                                [5.56, 13.32, 12.66, 10.40, 13.02, 8.36, 5.59, 19.05, 10.66, 13.47, 6.53]
                            ))


TSOLI_FEMALE_ERRORS_MM = dict(zip(["Ankle Circumference (mm)" ,
                                "Arm Length (Shoulder to Elbow) (mm)",
                                "Arm Length (Shoulder to Wrist) (mm)",
                                "Arm Length (Spine to Wrist) (mm)",
                                "Chest Circumference (mm)",
                                "Crotch Height (mm)",
                                "Head Circumference (mm)",
                                "Hip Circ Max Height (mm)",
                                "Hip Circumference, Maximum (mm)",
                                "Neck Base Circumference (mm)",
                                "Stature (mm)"],
                                [6.19, 6.65, 10.05, 11.87, 12.73, 5.50, 5.91, 18.59, 12.35, 15.43, 7.51]
                            ))


TSOLI_MALE_ERRORS_IN_CM = {m_name.replace("(mm)","(cm)"): (m_val / 10) for m_name, m_val in TSOLI_MALE_ERRORS_MM.items()}
TSOLI_FEMALE_ERRORS_IN_CM = {m_name.replace("(mm)","(cm)"): (m_val / 10) for m_name, m_val in TSOLI_FEMALE_ERRORS_MM.items()}


def get_average_landmark(vertices, landmark_indices, landmark_name):
    lm_ind1 = landmark_indices[landmark_name][0]
    lm_ind2 = landmark_indices[landmark_name][1]
    return (vertices[lm_ind1,:] + vertices[lm_ind2,:]) / 2
    

def get_simple_measurements(vertices, landmark_indices, measurement_definitions, use_measurements):

    estimated_measurements = torch.zeros(len(use_measurements))

    for i, m_name in enumerate(use_measurements):
        if m_name in measurement_definitions.keys():
            meas_def = measurement_definitions[m_name]
            if meas_def["measurement_type"] == "circumference":
                looped_indices = meas_def["indices"]
                verts1 = vertices[looped_indices[1:],:]
                verts2 = vertices[looped_indices[:-1]]
                estimated_measurements[i] = torch.sum(torch.sqrt(torch.sum((verts1-verts2)** 2,1)))
            else:
                landmark_names = meas_def["landmark_names"]
                accumulated_measurement = 0

                for j in range(len(landmark_names)-1):
                    if isinstance(landmark_indices[landmark_names[j]],tuple):
                        pt1 = get_average_landmark(vertices, landmark_indices, landmark_names[j])
                    else:
                        pt1 = vertices[landmark_indices[landmark_names[j]],:]

                    if isinstance(landmark_indices[landmark_names[j+1]],tuple):
                        pt2 = get_average_landmark(vertices, landmark_indices, landmark_names[j+1])
                    else:
                        pt2 = vertices[landmark_indices[landmark_names[j+1]],:]

                    accumulated_measurement += torch.norm(pt1.float()-pt2.float())
                estimated_measurements[i] = accumulated_measurement


                # if isinstance(landmark_indices[landmark_names[0]],tuple):
                #     pt1a = vertices[landmark_indices[landmark_names[0]][0],:]
                #     pt1b = vertices[landmark_indices[landmark_names[0]][1],:]
                #     pt1 = (pt1a+pt1b)/2
                # else:    
                #     pt1 = vertices[landmark_indices[landmark_names[0]],:]

                # if isinstance(landmark_indices[landmark_names[1]],tuple):
                #     pt2a = vertices[landmark_indices[landmark_names[1]][0],:]
                #     pt2b = vertices[landmark_indices[landmark_names[1]][1],:]
                #     pt2 = (pt2a+pt2b)/2
                # else:    
                #     pt2 = vertices[landmark_indices[landmark_names[1]],:]

                # estimated_measurements[i] = torch.norm(pt1.float()-pt2.float())
        else:
            raise ValueError(f"Measurement {m_name} not found in measurement_definitions")
            

    return estimated_measurements * 100 # convert to cm


def get_normalizing_landmark(landmark_names: list):
    """
    Find index of normalizing landmark 
    """

    if "Substernale" in landmark_names:
        landmark_normalizing_name = "Substernale"
    elif "Nose" in landmark_names:
        landmark_normalizing_name = "Nose"
    elif "BELLY_BUTTON" in landmark_names:
        landmark_normalizing_name =  "BELLY_BUTTON"
    else:
        landmark_normalizing_name =  landmark_names[0] # assign random landmark
    
    landmark_normalizing_ind = landmark_names.index(landmark_normalizing_name)
    return landmark_normalizing_name, landmark_normalizing_ind


def process_caesar_landmarks(landmark_path: str, scale: float = 1000.0):
    """
    Process landmarks from .lnd file - reading file from AUX to END flags.
    :param landmark_path (str): path to landmark .lnd file
    :param scale (float): scale of landmark coordinates

    Return: list of landmark names and coordinates
    :return landmark_dict (dict): dictionary with landmark names as keys and
                                    landmark coordinates as values
                                    landmark_coords are (np.array): (1,3) array 
                                    of landmark coordinates
    """

    landmark_coords = []
    landmark_names = []

    with open(landmark_path, 'r') as file:
        do_read = False
        for line in file:

            # start reading file when encounter AUX flag
            if line == "AUX =\n":
                do_read = True
                # skip to the next line
                continue
                
            # stop reading file when encounter END flag
            if line == "END =\n":
                do_read = False
            

            if do_read:
                # EXAMPLE OF LINE IN LANDMARKS
                # 1   0   1   43.22   19.77  -38.43  522.00 Sellion
                # where the coords should be
                # 0.01977, -0.03843, 0.522
                # this means that the last three floats before 
                # the name of the landmark are the coords
                
                # find landmark coordinates
                landmark_coordinate = re.findall(r"[-+]?\d+\.*\d*", line)
                # print(line)
                # print(landmark_coordinate)
                x = float(landmark_coordinate[-3]) / scale
                y = float(landmark_coordinate[-2]) / scale
                z = float(landmark_coordinate[-1]) / scale
                
                # find landmark name
                # (?: ......)+ repeats the pattern inside the parenthesis
                # \d* says it can be 0 or more digits in the beginning
                # [a-zA-Z]+ says it needs to be one or more characters
                # [.,/]* says it can be 0 or more symbols
                # \s* says it can be 0 ore more spaces
                # NOTE: this regex misses the case for landmarks with names
                # AUX LAND 79 -- it parses it as AUX LAND -- which is ok for our purposes
                landmark_name = re.findall(r" (?:\d*[a-zA-Z]+[-.,/]*\s*)+", line)
                landmark_name = landmark_name[0][1:-1]
                landmark_name_standardized = CAESAR_LANDMARK_MAPPING[landmark_name]

                # * zero or more of the preceding character. 
                # + one or more of the preceding character.
                # ? zero or one of the preceding character.
                
                landmark_coords.append([x,y,z])
                landmark_names.append(landmark_name_standardized)
                
    landmark_coords = np.array(landmark_coords)

    return dict(zip(landmark_names, landmark_coords))


##############################################################################################################
# TRAINING UTILS
##############################################################################################################

# from here https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
def weights_init_general_strategy(params,in_features, **kwargs):
    y = 1.0/np.sqrt(in_features)
    return torch.FloatTensor(params.data.shape).uniform_(-y,y)


def weights_init(network, strategy_name, strategy_params):
    """
    Initialize weights of the network according to the given strategy

    :param network: torch model to initialize weights
    :param strategy_name (Str): name of the strategy to use
    :param strategy_params (Dict): dictionary with parameters for the strategy
    """

    if isinstance(strategy_name,type(None)):
        return network

    if strategy_name == "output_layer_bias_to_mean_measurement":
        out_layer = getattr(network,"output_layer")
        out_layer_weights = getattr(out_layer,"weight")
        out_layer_bias = getattr(out_layer,"bias")

        set_bias_to = torch.tensor(strategy_params["output_layer"]["bias"])
        setattr(out_layer_bias,"data",set_bias_to)

        set_weights_to = strategy_params["output_layer"]["weight"]
        out_layer_weights.data.fill_(set_weights_to)


    elif strategy_name == "initialize_to_learned_LR":
        model_path = strategy_params["model_path"]
        with open(model_path,"rb") as f:
            model = pickle.load(f)

        set_weights_to = torch.from_numpy(model.coef_).float() # dim 11 x 170 (nr input features)
        set_bias_to = torch.from_numpy(model.intercept_).float() # dim 11

        out_layer = getattr(network,"output_layer")
        out_layer_weights = getattr(out_layer,"weight")
        out_layer_bias = getattr(out_layer,"bias")

        setattr(out_layer_weights,"data",set_weights_to)
        setattr(out_layer_bias,"data",set_bias_to)

    return network


def set_seed(sd):
    torch.manual_seed(sd)
    random.seed(sd)
    np.random.seed(sd)


def normal_sample_shape(batch_size, mean_shape, std_vector):
    """
    Gaussian sampling of shape parameter given deviations from the mean.
    """
    shape = mean_shape + torch.randn(batch_size, mean_shape.shape[0], device=mean_shape.device)*std_vector
    return shape  # (bs, num_smpl_betas)


def create_body_model(body_model_path: str, body_model_type: str, gender: str ="neutral", num_betas: int =10):
    '''
    Create SMPL body model
    :param: body_model_path (str): location to SMPL .pkl models
    :param body_model_type (str): smpl, smplx
    :param: gender (str): male, female or neutral
    :param: num_betas (int): number of SMPL shape coefficients
                            requires the model with num_coefs in smpl_path
   
    Return:
    :param: SMPL body model
    '''

    body_model_path = os.path.join(body_model_path,
                                   body_model_type,
                                   f"{body_model_type.upper()}_{gender.upper()}.pkl")
    
    return smplx.create(body_model_path,
                        # model_type=body_model_type.upper(),
                        # gender=gender.upper(),
                        num_betas=num_betas,
                        ext='pkl')


def create_results_directory(save_path: str = "/pose-independent-anthropometry/results",
                             continue_run: str = None):
    """
    Save results in save_path as YYYY_MM_DD_HH_MM_SS folder.
    If continue_run is folder of type YYYY_MM_DD_HH_MM_SS, then
    save results in {save_path}/{continue_run} folder.
    
    :param save_path: path to save results to
    :param continue_run: string of type YYYY_MM_DD_HH_MM_SS
    """


    if continue_run:
        # check if formatting of continue_run folder looks like "%Y_%m_%d_%H_%M_%S"
        # wil raise ValueError if not
        try:
            _ = datetime.strptime(continue_run.split("/")[-1],"%Y_%m_%d_%H_%M_%S")
        except Exception as e:
            raise ValueError("CONTINUE_RUN must be a folder of type YYYY_MM_DD_HH_MM_SS")

        print(f"Continuing run from previous checkpoint")
        save_path = os.path.join(save_path,continue_run)
    else:
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        save_path = os.path.join(save_path,current_time)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    print(f"Saving results to {save_path}")

    return save_path


def load_config(path="configs/config.yaml"):
    with open(path,'r') as f:
        cfg = yaml.safe_load(f)

    return cfg


def parse_landmark_txt_coords_formatting(data: List[str]):
    """
    Parse landamrk txt file where each line is formatted as:
    x_coord y_coord z_coord landmark_name

    :param data (List[str]) list of strings, each string
                represents one line from the txt file
    
    :return landmarks (dict) in formatting 
                      {landmark_name: [x,y,z]}
    """

    # get number of landmarks
    N = len(data)
    if data[-1] == "\n":
        N -= 1

    # define landmarks
    landmarks = {}

    for i in range(N):
        splitted_line = data[i].split(" ")
        x = float(splitted_line[0])
        y = float(splitted_line[1])
        z = float(splitted_line[2])

        remaining_line = splitted_line[3:]
        landmark_name = " ".join(remaining_line)
        if landmark_name[-1:] == "\n":
            landmark_name = landmark_name[:-1]

        landmarks[landmark_name] = [x,y,z]

    return landmarks


def parse_landmark_txt_index_formatting(data):
    """
    Parse landamrk txt file with formatting
    landmark_index landmark_name

    :param data (List[str]) list of strings, each string
                represents one line from the txt file
    
    :return landmarks (dict) in formatting 
                      {landmark_name: index}
    """

    # get number of landmarks
    N = len(data)
    if data[-1] == "\n":
        N -= 1

    # define landmarks
    landmark_indices = {}

    for i in range(N):
        splitted_line = data[i].split(" ")
        ind = int(splitted_line[0])

        remaining_line = splitted_line[1:]
        landmark_name = " ".join(remaining_line)
        if landmark_name[-1:] == "\n":
            landmark_name = landmark_name[:-1]

        landmark_indices[landmark_name] = ind

    return landmark_indices


def load_landmarks(landmark_path: str, 
                   landmark_subset: List[str] = None, 
                   scan_vertices: np.ndarray = None,
                   landmarks_scale: float = 1000,
                   verbose: bool = True):
    """
    Load landmarks from file and return the landmarks as
    torch tensor.

    Landmark file is defined in the following format:
    .txt extension
    Option1) x y z landmark_name
    Option2) landmark_index landmark_name
    .json extension
    Option1) {landmark_name: [x,y,z]}
    Option2) {landmark_name: index}

    :param landmark_path: (str) of path to landmark file
    :param landmark_subset: (list) list of strings of landmark
                            names to use
    :param scan_vertices: (torch.tensor) dim (N,3) of the vertices
                          if landmarks defined as indices of the
                          vertices, returin landmarks as 
                          scan_vertices[landmark_indices,:]

    Return: landmarks: np.array of landmarks 
                       with dim (K,3)
    """

    # if empty landmark subset, return None
    if landmark_subset == []:
        return {}

    ext = landmark_path.split(".")[-1]
    supported_extensions = [".txt",".json",".lnd"]
    formatting_type = "indices"

    if ext == "txt":
        # read txt file
        with open(landmark_path, 'r') as file:
            data = file.readlines()

        # check formatting type
        try:
            _ = float(data[0].split(" ")[1])
            formatting_type = "coords"
        except Exception as e:
            pass

        # parse landmarks
        if formatting_type == "coords":
            landmarks = parse_landmark_txt_coords_formatting(data)
        elif formatting_type == "indices":
            if isinstance(scan_vertices,type(None)):
                msg = "Scan vertices need to be provided for"
                msg += "index type of landmark file formatting"
                raise NameError(msg)
            landmark_inds = parse_landmark_txt_index_formatting(data)
            landmarks = {}
            for lm_name, lm_ind in landmark_inds.items():
                landmarks[lm_name] = scan_vertices[lm_ind,:]

    elif ext == "json":
        with open(landmark_path,"r") as f:
            data = json.load(f)

        # check formatting type
        first_lm = list(data.keys())[0]
        if isinstance(data[first_lm],list):
            formatting_type = "coords"

        if formatting_type == "coords":
            landmarks = landmarks = {lm_name: np.array(lm_val) for lm_name,lm_val in data.items()}
        elif formatting_type == "indices":
            if isinstance(scan_vertices,type(None)):
                msg = "Scan vertices need to be provided for"
                msg += "index type of landmark file formatting"
                raise NameError(msg)
            
            landmarks = {}
            for lm_name, lm_ind in data.items():
                landmarks[lm_name] = scan_vertices[lm_ind,:]

    elif ext == "lnd":
        if verbose:
            print("Be aware that the .lnd extension assumes you are using the caesar dataset.")
        landmarks = process_caesar_landmarks(landmark_path,landmarks_scale)

    else:
        supported_extensions_str = ', '.join(supported_extensions)
        msg = f"Landmark extensions supported: {supported_extensions_str}. Got .{ext}."
        raise ValueError(msg)

    # select subset of landmarks
    if not isinstance(landmark_subset,type(None)):
        landmarks_sub = {}
        for lm_name in landmark_subset:
            if lm_name in landmarks:
                landmarks_sub[lm_name] = landmarks[lm_name]

        landmarks =  landmarks_sub

    return landmarks


class LM2Features():
    """
    Create features from landmark coordaintes
    """
    def __init__(self, **kwargs):
        super(LM2Features, self).__init__()

        self.transform_landmarks = kwargs["transform_landmarks"]
        self.ravel_features = kwargs["ravel_features"] if "ravel_features" in kwargs else False

        # get output feature dim
        self.landmarks_dict = eval(kwargs["landmarks"])
        self.n_landmarks = len(self.landmarks_dict.keys())
        self.landmark_normalization = kwargs["landmark_normalization"] if "landmark_normalization" in kwargs else None
        if isinstance(self.landmark_normalization, str) and self.landmark_normalization in ["Substernale","Nose","BELLY_BUTTON"]:
            self.n_landmarks -= 1 # 1 landmark is deleted because used as normalizing landmark

        if self.transform_landmarks == "coords":
            self.out_dim = self.n_landmarks * 3

        if self.transform_landmarks == "vectors_all":
            self.out_dim = (self.n_landmarks * 3)

        # for distances_grouped
        if self.transform_landmarks == "distances_grouped":
            inds = np.load(kwargs["grouping_inds_path"])
            self.inds0 = inds[:,0].tolist()
            self.inds1 = inds[:,1].tolist()
            # self.biggest_group_n = np.max([len(group_inds[0]) for group_name, group_inds 
            #                                 in self.distances_grouped_dict.items()]).item()
            self.out_dim = len(self.inds0)

        if self.transform_landmarks == "distances_all":
            # pairwise distances are NxN and then subtract the diagonal elements
            self.out_dim = (self.n_landmarks * self.n_landmarks) - self.n_landmarks

        if self.transform_landmarks in ["angles_pose_grouped", "angles_shape_grouped"]:
            with open(kwargs["grouping_inds_path"], 'rb') as f:
                body_part_angles = pickle.load(f)
            self.inds0 = []
            self.inds1 = []
            self.inds2 = []
            for body_part_name, body_part_triplets in body_part_angles.items():
                for triplets in body_part_triplets:
                    self.inds0.append(triplets[0])
                    self.inds1.append(triplets[1])
                    self.inds2.append(triplets[2])
            
            self.out_dim = len(self.inds0)

    def coords(self, landmarks, **kwargs):
        """
        Return the same landmark coordinates but reshaped 
        to (B,n_landm*3) if ravel_features is True
        """
        if self.ravel_features:
            # (B,n_landm,3) -> (B,n_landm*3)
            landmarks = landmarks.reshape(landmarks.shape[0],-1)
        return landmarks

    def vectors_all(self, landmarks, **kwargs):
        """
        Transform landmarks of dim (B,n_landm,3) to features (B,n_landm,n_landm*3) where
        each landmark is now represented as vectors to all other landmarks
        If retain_orig_coord, then the first three coordinates of each landmark are 
        the original coordinates and the final shape is (B,n_landm,(n_landm+1)*3)

        :param landmarks: (torch.tensor) dim (B,n_landm,3)
        :param retain_orig_coord: (bool) if True, retain original coordinates of landmarks

        :return:
        if retain_orig_coord:
            first three corrds of landmarks[:,:,:3] are the original coordinates
            landmarks[:,:,3:6] are the 
        
        tested with:
        v = torch.randint(0,10,(2,5,3))
        B,N_landm, _ = v.shape
        v1 = torch.zeros(B,N_landm, (N_landm*3) + 3)
        v1[:,:,3:] = v.reshape(B,-1).unsqueeze(1).repeat(1,N_landm,1)
        v1[:,:,:3] = v
        v2 = torch.zeros(B,N_landm, (N_landm*3) + 3)
        v2[:,:,3:] = - v.repeat(1,1,N_landm)

        """

        B, n_landm, _ = landmarks.shape

        f1 = torch.zeros(B,n_landm, (n_landm*3) + 3)
        f1[:,:,3:] = landmarks.reshape(B,-1).unsqueeze(1).repeat(1,n_landm,1)
        f1[:,:,:3] = landmarks

        f2 = torch.zeros(B,n_landm, (n_landm*3) + 3)
        f2[:,:,3:] = - landmarks.repeat(1,1,n_landm)

        features = f1 - f2

        # FIXME: this might not work, think about it
        if not self.retain_orig_coord:
            features = features[:,:,3:]

        return features

    def distances_all(self, landmarks, **kwargs):
        # lm are shape B x n_landmarks x 3
        B = landmarks.shape[0]
        dists = pairwise_dist(landmarks, 
                              landmarks).squeeze()
        dists = torch.sqrt(dists)

        # get only non-diagonal elements and flatten to B x n_landmarks**2
        mask = torch.eye(self.n_landmarks).bool().repeat(B,1,1)
        dists = dists[~mask].reshape(B,-1)

        # if self.ravel_features:
        #     dists = dists.reshape(B,-1)
        return dists

    def distances_grouped(self, landmarks, **kwargs):
        """
        The features are distances between the landmarks given the indices
        of neighboring landamrks
        """
        
        # _, n_landm, _ = landmarks.shape

        dists = torch.norm(landmarks[:,self.inds0,:] - landmarks[:,self.inds1,:],dim=2) # of shape (B,len(self.inds0))
        #features = dists.unsqueeze(1).repeat(1,self.n_landmarks,1) # (B, n_landm, len(self.inds0))
        # if self.retain_orig_coord:
        #     features =  torch.cat([landmarks,features],dim=2).float() # (B, n_landm, n_features)
        # else:
        #     features =  features.float() # (B,n_landm,n_features)

        if self.ravel_features:
            features = dists # (B, len(self.inds0))
        else:
            features = dists.unsqueeze(1).repeat(1,self.n_landmarks,1) # (B, n_landm, len(self.inds0))

        return features

    def angles_pose_grouped(self, landmarks, **kwargs):
        '''
        Use landmark triplets to find angles between them.
        Use self.inds1 as the middle landmark -> put it in origin
        and find angle between the other two
        '''
        
        # try if works for batch of lm
        lm0 = landmarks[:,self.inds0]
        lm_mid = landmarks[:,self.inds1]
        lm1 = landmarks[:,self.inds2]

        lm0 = lm0 - lm_mid
        lm1 = lm1 - lm_mid

        dot_prod = torch.sum(torch.mul(lm0,lm1),dim=2)

        lm0_norm = torch.norm(lm0,dim=2)
        lm1_norm = torch.norm(lm1,dim=2)
        norms = torch.mul(lm0_norm,lm1_norm)

        dot_prod_normalized = torch.div(dot_prod,norms)

        return torch.acos(torch.clamp(dot_prod_normalized,-1,1)) # (B,len(self.inds0))

    def angles_shape_grouped(self, landmarks, **kwargs):
        '''
        Use landmark triplets to find angles between them.
        Use self.inds1 as the middle landmark -> put it in origin
        and find angle between the other two
        '''
        
        # try if works for batch of lm
        lm0 = landmarks[:,self.inds0]
        lm_mid = landmarks[:,self.inds1]
        lm1 = landmarks[:,self.inds2]

        lm0 = lm0 - lm_mid
        lm1 = lm1 - lm_mid

        dot_prod = torch.sum(torch.mul(lm0,lm1),dim=2)

        lm0_norm = torch.norm(lm0,dim=2)
        lm1_norm = torch.norm(lm1,dim=2)
        norms = torch.mul(lm0_norm,lm1_norm)

        dot_prod_normalized = torch.div(dot_prod,norms)

        return torch.acos(torch.clamp(dot_prod_normalized,-1,1)) # (B,len(self.inds0))


class CAESAR_Name2Gender(object):
    """
    Get CAESAR gender from subject name
    """

    def __init__(self,
                 gender_mapper_path: str="/pose-independent-anthropometry/data/gender/CAESAR_GENDER_MAPPER.npz"):
        
        self.gender_mapper = np.load(gender_mapper_path)

    def get_gender(self, subj_name):

        # if subject in sitting pose -> "convert" to standing pose
        if subj_name[-1] == "b":
            subj_name = subj_name[:-1] + "a"
    
        if subj_name in self.gender_mapper["names"]:
            ind = np.where(self.gender_mapper["names"] == subj_name)[0].item()
            return self.gender_mapper["genders"][ind].lower()
        else:
            return None
        

class StepLR():
    def __init__(self, init_lr=0.001, decrease_every_k_epochs=5, decrease_by=2, **kwargs):
        super(StepLR, self).__init__()

        self.lr = init_lr
        self.decrease_every_k_epochs = decrease_every_k_epochs
        self.decrease_by = decrease_by
        self.last_epoch_changed_lr = 0

        print(f"Initialized scheduler StepLR with params:" +
              f"[INIT LR: {self.lr}," +
              f"DECREASE_EVERY_K_EPOCHS: {self.decrease_every_k_epochs}," +
              f"DECREASE_BY: {self.decrease_by}]")

    def update(self, epoch, **kwargs):
        if (epoch == 0) and (self.decrease_every_k_epochs == 1): # dont want to change in 0th epoch learning rate
            self.last_epoch_changed_lr = epoch + 1
            return False
        if ((epoch +1) - self.last_epoch_changed_lr) == self.decrease_every_k_epochs:
            self.lr = self.lr / self.decrease_by
            self.last_epoch_changed_lr = epoch + 1
            print(f"Update LR - {self.lr}")
            return True
        else:
            return False

class ConstantLR():
    def __init__(self, init_lr=0.001, **kwargs):
        super(ConstantLR, self).__init__()

        self.lr = init_lr

        print(f"Initialized constant learning rate ConstantLR:" +
              f"[INIT LR: {self.lr}")

    def update(self, epoch, **kwargs):
        return False

##############################################################################################################
# STANDING POSE UTILS
##############################################################################################################

POSE_LB = {"left_hip": torch.tensor([0,0,0]),
            "left_ankle": torch.tensor([0,0,-0.5]),
            "right_hip": torch.tensor([0,0,-0.5]),
            "right_ankle": torch.tensor([0,0,-0.2]),

            "left_shoulder": torch.tensor([0,0,-1.3]),
            "left_elbow": torch.tensor([-1,0,0]),
            "left_wrist": torch.tensor([-0.5,0,0]),

            "right_shoulder": torch.tensor([-1,0,0]),
            "right_elbow": torch.tensor([-1,0,0]),
            "right_wrist": torch.tensor([-0.5,0,0]),
            }

POSE_UB = {"left_hip": torch.tensor([0,0,0.5]),
            "left_ankle": torch.tensor([0,0,0.2]),
            "right_hip": torch.tensor([0,0,0]),
            "right_ankle": torch.tensor([0,0,0.5]),

            "left_shoulder": torch.tensor([0,0,1]),
            "left_elbow": torch.tensor([0.2,0,0]),
            "left_wrist": torch.tensor([0.5,0,0]),

            "right_shoulder": torch.tensor([1.3,0,0]),
            "right_elbow": torch.tensor([0.2,0,0]),
            "right_wrist": torch.tensor([0.5,0,0]),
            }


A_POSE_JOINT_AA_BOUNDS = {"LOWER_BOUNDS": POSE_LB,
                          "UPPER_BOUNDS": POSE_UB}

def set_body_pose_aa(current_pose,joint_name,set_pose):
    '''
    Set the pose for the joint_name as set_pose in the 
    pose parameters current_pose

    :param current_pose: (torch.tensor) dim (B,72)
    :param joint_name: (str) name of joint that want to change pose
    :param set_pose: (torch.tensor) (3) rotatin matrix
    '''
    JOINT2IND = {'pelvis': 0,
                 'left_hip': 1,
                 'right_hip': 2,
                 'spine1': 3,
                 'left_knee': 4,
                 'right_knee': 5,
                 'spine2': 6,
                 'left_ankle': 7,
                 'right_ankle': 8,
                 'spine3': 9,
                 'left_foot': 10,
                 'right_foot': 11,
                 'neck': 12,
                 'left_collar': 13,
                 'right_collar': 14,
                 'head': 15,
                 'left_shoulder': 16,
                 'right_shoulder': 17,
                 'left_elbow': 18,
                 'right_elbow': 19,
                 'left_wrist': 20,
                 'right_wrist': 21,
                 'left_hand': 22,
                 'right_hand': 23}
    
    k = JOINT2IND[joint_name]
    
    current_pose[:,(3*k):(3*k + 3)] = set_pose
    
    return current_pose


def generate_standing_poses(N=2000):

    pose_UB = torch.zeros(N,72)
    pose_LB = torch.zeros(N,72)
    
    LOWER_BOUNDS = A_POSE_JOINT_AA_BOUNDS["LOWER_BOUNDS"]
    UPPER_BOUNDS = A_POSE_JOINT_AA_BOUNDS["UPPER_BOUNDS"]

    joint_names = LOWER_BOUNDS.keys()

    for joint_name in joint_names:
        pose_LB = set_body_pose_aa(pose_LB,joint_name,LOWER_BOUNDS[joint_name])
        pose_UB = set_body_pose_aa(pose_UB,joint_name,UPPER_BOUNDS[joint_name])

    # random 3 floats between r1 and r2 
    # (r1 - r2) * torch.rand(1,3) + r2
    # for 3 floats between -1 and 1
    # (-1 - 1) * torch.rand(1, 3)  + 1

    pose_bounds_diff = (pose_LB - pose_UB)
    uniform_on_01 = torch.rand(N,72)
    pose = torch.mul(pose_bounds_diff, uniform_on_01) + pose_UB # N x 72

    return pose



##############################################################################################################
# EVAL UTILS
##############################################################################################################

def find_rotation_between_vectors(v1,v2):
    """
    Find rotation that rotates vector v1 to v2
    """

    v1_magnitude = torch.norm(v1)
    v2_magnitude = torch.norm(v2)

    v1_unit = v1 / v1_magnitude
    v2_unit = v2 / v2_magnitude

    rotation_axis = torch.cross(v1_unit, v2_unit)
    rotation_angle = torch.acos(torch.dot(v1_unit, v2_unit))


def rotation_matrix_from_vectors_numpy(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def rotation_matrix_from_vectors_torch(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    vec1_unit, vec2_unit = (vec1 / torch.norm(vec1)), (vec2 / torch.norm(vec2))
    v = torch.cross(vec1_unit, vec2_unit) # rotation axis
    c = torch.dot(vec1_unit, vec2_unit)
    s = torch.norm(v)
    K = torch.tensor([[0, -v[2], v[1]], 
                      [v[2], 0, -v[0]], 
                      [-v[1], v[0], 0]])
    K2 = torch.matmul(K,K)
    rotation_matrix = torch.eye(3) + K + K2 * ((1 - c) / (s ** 2))
    return rotation_matrix


def pelvis_normalization(
                         landmarks,
                         rt_psis_ind, lt_psis_ind, 
                         rt_asis_ind,lt_asis_ind,
                         nuchale_ind, 
                         return_transformations=False):
    """
    Normalize scan on pelvis. The steps are:
    1. center body on centroid of triangle waist landmark, rt asis landmark and lt asis landmark
    2. find normal to this plane (with direction to half-space where Nuchale landmark is located)
    3. orient normal onto y-ax
    4. orient rt asis towards z-ax

    :param landmarks: (torch.tensor) dim (K,3) of the landmarks
    :param rt_psis_ind: (int) index of right psis landmark
    :param lt_psis_ind: (int) index of left psis landmark
    :param rt_asis_ind: (int) index of right asis landmark
    :param lt_asis_ind: (int) index of left asis landmark
    :param nuchale_ind: (int) index of nuchale landmark
    :param return_transformations: (bool) if True, return the transformations applied to landmarks
    """
    
    # 1. find mid point of psis landmarks
    mid_psis = (landmarks[rt_psis_ind] + landmarks[lt_psis_ind]) / 2 

    # 2. find centroid of triangle mid_psis, rt asis and lt asis
    centroid = (mid_psis + landmarks[rt_asis_ind] + landmarks[lt_asis_ind]) / 3
    # centroid = torch.sum(landmarks[[waist_ind,rt_asis_ind,lt_asis_ind],:],dim=0) / 3

    # 3. center body on centroid
    landmarks = landmarks - centroid
    mid_psis = mid_psis - centroid

    # 4. find normal of plane defined by points mid_psis, rt asis and lt asis
    vec_rt_asis_2_mid_psis = landmarks[rt_asis_ind] - mid_psis
    vec_lt_asis_2_mid_psis = landmarks[lt_asis_ind] - mid_psis
    plane_normal = torch.cross(vec_rt_asis_2_mid_psis,vec_lt_asis_2_mid_psis)

    # 5. check plane_normal orientation by finding distance to Nuchale landmark
    # pick orientation that points towards head
    nuchale_point = landmarks[nuchale_ind]
    dist_pos = torch.norm(plane_normal-nuchale_point,p=2)
    dist_neg = torch.norm(-plane_normal-nuchale_point,p=2)

    # flip plane normal if distance to "negative" plane normal is less than "positive" plane normal
    if dist_neg < dist_pos:
        plane_normal = - plane_normal

    # 6. rotate plane normal so it aligns with y-ax
    if torch.allclose(torch.zeros_like(plane_normal), plane_normal):
        R2y = torch.eye(3)
    else:
        R2y = rotation_matrix_from_vectors_torch(plane_normal,
                                                torch.tensor([0,1,0],dtype=plane_normal.dtype))
        landmarks = torch.matmul(landmarks, R2y.T) # K x 3

    # 6. rotate Rt asis point to face z-ax (Rt asis vector is now on the xz plane)
    if torch.allclose(torch.zeros_like(landmarks[rt_asis_ind]), landmarks[rt_asis_ind]):
        R2z = torch.eye(3)
    else:
        R2z = rotation_matrix_from_vectors_torch(landmarks[rt_asis_ind],
                                                torch.tensor([0,0,1],dtype=landmarks.dtype))
        landmarks = torch.matmul(landmarks, R2z.T) # K x 3

    if return_transformations:
        return landmarks, centroid, R2y.float(), R2z.float()
    else:
        return landmarks
       

CAESAR_BAD_SUBJECTS = [ # ALL SUBJECTS ARE STANDING
                        # FEMALE EXAMPLES
                        "nl_5419a", # a lot of missing landmarks - (0,0,0) + missing back body (partial scan)
                        "nl_6289a", # a lot of missing landmarks - (0,0,0) + missing back body (partial scan)
                        # MALE EXAMPLES
                        "nl_5512a", # missing a lot of landmarks + missing legs on scan
                        ]


##############################################################################################################
# UNPOSING CAESAR UTILS
##############################################################################################################


def unpose_caesar(scan_verts, scan_landmarks, fit_verts, fit_shape, fit_pose, 
                  fit_scale, fit_trans, body_model, batch_size=1, 
                  T=None, J=None, pose_offsets=None, scan2fit_inds=None, lm2fit_inds=None,
                  scan_markers=None, marker2fit_inds=None, unposing_landmark_choice="nn_to_verts",
                  smpl_landmark_inds=None):
    """
    Unpose CAESAR scan by unposing the underlying fitted SMPL body model.
    Each point of the scan is unposed in the same way as its corresponding nearest 
    point on the fitted body model.


    :param scan_verts: (torch.tensor) dim (N,3) vertices of scan
    :param scan_landmarks: (torch.tensor) dim (n_landmarks,3) landmarks of scan
    :param fit_verts: (torch.tensor) dim (N,3) vertices of fitted body model
    :param fit_shape: (torch.tensor) dim (batch_size, 10) shape parameters of fitted mesh
    :param fit_pose: (torch.tensor) dim (batch_size, 72) pose parameters of fitted mesh
    :param fit_scale: (torch.tensor) dim (1) scale parameter of fitted mesh
    :param fit_trans: (torch.tensor) dim (batch_size, 3) translation parameter of fitted mesh
    :param body_model: (smplx.body_models.SMPL) smplx body model (SMPL,SMPLX,..) body model
    :param batch_size: (int) batch size 
    
    #NOTE: not tested for batch_size > 1
    """
    if (isinstance(T,type(None))) or (isinstance(pose_offsets,type(None))):
        v_shaped = body_model.v_template + smplx.lbs.blend_shapes(fit_shape, body_model.shapedirs)
        J = smplx.lbs.vertices2joints(body_model.J_regressor, v_shaped)

        # get pose offsets
        ident = torch.eye(3)
        rot_mats = smplx.lbs.batch_rodrigues(fit_pose.view(-1, 3)).view([batch_size, -1, 3, 3])
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, body_model.posedirs).view(batch_size, -1, 3)

        _, A = smplx.lbs.batch_rigid_transform(rot_mats, J, body_model.parents)

        W = body_model.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1]) 
        num_joints = body_model.J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
        
    # find association from fitted bm to caesar
    # rows are points from scan_verts
    # cols are points from fit_verts
    if not isinstance(scan_verts, type(None)):
        if isinstance(scan2fit_inds,type(None)):
            dists = pairwise_dist(scan_verts.unsqueeze(0).float(),
                                  fit_verts.unsqueeze(0).float())
            scan2fit_inds = torch.argmin(dists,axis=1)[0]
        T_scan = T[:,scan2fit_inds,:,:].float()
        pose_offsets_scan = pose_offsets[:,scan2fit_inds,:]
        
        
        # unpose scan
        N_verts = scan_verts.shape[0]
        verts_scan_unscale_untrans = ((scan_verts / fit_scale) - fit_trans).unsqueeze(0).float()
        verts_scan_unscale_untrans_homo = torch.cat([verts_scan_unscale_untrans,
                                                    torch.ones([batch_size,N_verts,1])],dim=2)
        verts_scan_unscale_untrans_unposed = torch.matmul(torch.inverse(T_scan),
                                                        verts_scan_unscale_untrans_homo.unsqueeze(-1))[:,:,:3,0]
        verts_scan_unscale_untrans_unposed = (verts_scan_unscale_untrans_unposed - pose_offsets_scan).squeeze()
    else:
        verts_scan_unscale_untrans_unposed = None
        scan2fit_inds = None

    # unpose landmarks
    if not isinstance(scan_landmarks, type(None)):
        if unposing_landmark_choice == "nn_to_verts":
            if isinstance(lm2fit_inds,type(None)):
                dists = pairwise_dist(scan_landmarks.unsqueeze(0).float(),
                                    fit_verts.unsqueeze(0).float())
                lm2fit_inds = torch.argmin(dists,axis=1)[0]
        elif unposing_landmark_choice == "nn_to_smpl":
            lm2fit_inds = smpl_landmark_inds

        T_lm = T[:,lm2fit_inds,:,:].float()
        pose_offsets_lm = pose_offsets[:,lm2fit_inds,:]

        N_lm = scan_landmarks.shape[0]
        scan_lm_unscale_untrans = ((scan_landmarks / fit_scale) - fit_trans).unsqueeze(0).float()
       
        scan_lm_unscale_untrans_homo = torch.cat([scan_lm_unscale_untrans,
                                                    torch.ones([batch_size,N_lm,1])],dim=2)
        scan_lm_unscale_untrans_unposed = torch.matmul(torch.inverse(T_lm),
                                                        scan_lm_unscale_untrans_homo.unsqueeze(-1))[:,:,:3,0]
        scan_lm_unscale_untrans_unposed = (scan_lm_unscale_untrans_unposed - pose_offsets_lm).squeeze()
    else:
        scan_lm_unscale_untrans_unposed = None
        lm2fit_inds = None

    # unpose markers
    if not isinstance(scan_markers, type(None)):
        if isinstance(marker2fit_inds,type(None)):
            dists = pairwise_dist(scan_markers.unsqueeze(0).float(),
                                  fit_verts.unsqueeze(0).float())
            marker2fit_inds = torch.argmin(dists,axis=1)[0]
        T_markers = T[:,marker2fit_inds,:,:].float()
        pose_offsets_markers = pose_offsets[:,marker2fit_inds,:]

        N_markers = scan_markers.shape[0]
        scan_markers_unscale_untrans = ((scan_markers / fit_scale) - fit_trans).unsqueeze(0).float()
       
        scan_markers_unscale_untrans_homo = torch.cat([scan_markers_unscale_untrans,
                                                    torch.ones([batch_size,N_markers,1])],dim=2)
        scan_markers_unscale_untrans_unposed = torch.matmul(torch.inverse(T_markers),
                                                        scan_markers_unscale_untrans_homo.unsqueeze(-1))[:,:,:3,0]
        scan_markers_unscale_untrans_unposed = (scan_markers_unscale_untrans_unposed - pose_offsets_markers).squeeze()
    else:
        scan_markers_unscale_untrans_unposed = None
        marker2fit_inds = None

    
    return (verts_scan_unscale_untrans_unposed, scan2fit_inds, 
            scan_lm_unscale_untrans_unposed, lm2fit_inds, 
            scan_markers_unscale_untrans_unposed, marker2fit_inds)


def pairwise_dist(xyz1, xyz2):
    """
    :param xyz1 torch tensor (B,N,1)
    :param xyz1 torch tensor (B,M,1)

    :return dist torch tensor (B,M,N)
    """
    r_xyz1 = torch.sum(xyz1 * xyz1, dim=2, keepdim=True)  # (B,N,1)
    r_xyz2 = torch.sum(xyz2 * xyz2, dim=2, keepdim=True)  # (B,M,1)
    mul = torch.matmul(xyz2, xyz1.permute(0,2,1))         # (B,M,N)
    dist = r_xyz2 - 2 * mul + r_xyz1.permute(0,2,1)       # (B,M,N)
    return dist


def load_scan(scan_path):
    """
    Load 3D scan. 
    Accepted formats .ply, .ply.gz

    :param scan_path (str): path to the scan to load
    """

    ext = scan_path.split(".")[-1]
    ext_extended = f"{scan_path.split('.')[-2]}.{ext}"
    supported_extensions = [".ply",".ply.gz"]

    if ext == "ply":
        scan = o3d.io.read_triangle_mesh(scan_path)
        scan_vertices = np.asarray(scan.vertices)
        scan_faces = np.asarray(scan.triangles)
        scan_faces = scan_faces if scan_faces.shape[0] > 0 else None

    elif ext_extended == "ply.gz":
        with gzip.open(scan_path, 'rb') as gz_file:
            try:
                ply_content = gz_file.read()
            except Exception as _:
                raise ValueError(f"Cannot read .ply.gz file: {scan_path}")

            temp_ply_path = tempfile.mktemp(suffix=".ply")
            with open(temp_ply_path, 'wb') as temp_ply_file:
                temp_ply_file.write(ply_content)

            scan = o3d.io.read_triangle_mesh(temp_ply_path)
            scan_vertices = np.asarray(scan.vertices)
            scan_faces = np.asarray(scan.triangles)
            scan_faces = scan_faces if scan_faces.shape[0] > 0 else None
            os.remove(temp_ply_path)

    else:
        supported_extensions_str = ', '.join(supported_extensions)
        msg = f"Scan extensions supported: {supported_extensions_str}. Got .{ext}."
        raise ValueError(msg)

    return scan_vertices, scan_faces


def repose_caesar(unposed_scan_verts, scan2fit_inds, new_pose, 
                  fitted_shape, fitted_trans, fitted_scale, body_model, batch_size=1,
                  unposed_scan_landmarks=None, lm2fit_inds=None, J=None,
                  unposed_scan_markers=None, marker2fit_inds=None
                  ):
    """
    Repose the unposed CAESAR scan by reposing the underlying fitted SMPL body model.
    Each point of the scan is reposed in the same way as its corresponding nearest
    point on the fitted body model.
    
    :param unposed_scan_verts: (torch.tensor) dim (N,3) vertices of unposed scan
    :param scan2fit_inds: (torch.tensor) dim (N) indices of smpl vertices that correspond to 
                                                  the given scan index
    :param new_pose: (torch.tensor) dim (batch_size, 72) pose parameters with which pose unposed_scan_verts
    :param fitted_shape: (torch.tensor) dim (batch_size, 10) shape parameters of fitted bm to scan
    :param fitted_trans: (torch.tensor) dim (batch_size, 3) translation parameter of fitted bm to scan
    :param fitted_scale: (torch.tensor) dim (1) scale parameter of fitted bm to scan
    :param body_model: (smplx.body_models.SMPL) smplx body model (SMPL,SMPLX,..) body model
    :param batch_size: (int) batch size
    :param unposed_scan_landmarks: (torch.tensor) dim (n_landmarks,3) landmarks of unposed scan
    :param lm2fit_inds: (torch.tensor) dim (n_landmarks) indices of smpl vertices that correspond to
                                                    the given landamrks
    :param J: (torch.tensor) dim (batch_size, 24, 3) joints of fitted bm to scan
    :param unposed_scan_markers: (torch.tensor) dim (n_markers,3) markers of unposed scan
    :param marker2fit_inds: (torch.tensor) dim (n_markers) indices of smpl vertices that correspond to

    :NOTE: not tested for batch_size > 1
    """
    
    # pose offset
    ident = torch.eye(3)
    rot_mats = smplx.lbs.batch_rodrigues(new_pose.view(-1, 3)).view([batch_size, -1, 3, 3])
    pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
    pose_offsets = torch.matmul(pose_feature, body_model.posedirs).view(batch_size, -1, 3)
    
    # create homo transform T
    if isinstance(J, type(None)):
        v_shaped = body_model.v_template + smplx.lbs.blend_shapes(fitted_shape, body_model.shapedirs)
        J = smplx.lbs.vertices2joints(body_model.J_regressor, v_shaped)
    _, A = smplx.lbs.batch_rigid_transform(rot_mats,J,body_model.parents)

    W = body_model.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1]) 
    num_joints = body_model.J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
    
    
    # repose scan
    if not isinstance(unposed_scan_verts, type(None)):
        N_verts = unposed_scan_verts.shape[0]
        pose_offsets_scan = pose_offsets[:,scan2fit_inds,:]
        verts_posed = unposed_scan_verts + pose_offsets_scan # (1,N,3)

        T_scan = T[:,scan2fit_inds,:,:]
        verts_posed_homo = torch.cat([verts_posed,
                                    torch.ones([batch_size,N_verts,1])],dim=2)
        verts_reposed = torch.matmul(T_scan, torch.unsqueeze(verts_posed_homo, dim=-1))[:,:,:3,0]
        verts_reposed = (fitted_scale * (verts_reposed + fitted_trans.unsqueeze(0))).squeeze()
    else:
        verts_reposed = None

    # repose lm
    if not isinstance(unposed_scan_landmarks, type(None)):
        N_lm = unposed_scan_landmarks.shape[0]
        pose_offsets_lm = pose_offsets[:,lm2fit_inds,:]
        lm_posed = unposed_scan_landmarks + pose_offsets_lm # (1,N,3)

        T_lm = T[:,lm2fit_inds,:,:]
        lm_posed_homo = torch.cat([lm_posed,
                                  torch.ones([batch_size,N_lm,1])],dim=2)
        lm_reposed = torch.matmul(T_lm, torch.unsqueeze(lm_posed_homo, dim=-1))[:,:,:3,0]
        lm_reposed = (fitted_scale * (lm_reposed + fitted_trans.unsqueeze(0))).squeeze()
    else:
        lm_reposed = None

    # repose markers
    if not isinstance(unposed_scan_markers, type(None)):
        N_markers = unposed_scan_markers.shape[0]
        pose_offsets_markers = pose_offsets[:,marker2fit_inds,:]
        markers_posed = unposed_scan_markers + pose_offsets_markers # (1,N,3)

        T_markers = T[:,marker2fit_inds,:,:]
        marker_posed_homo = torch.cat([markers_posed,
                                  torch.ones([batch_size,N_markers,1])],dim=2)
        markers_reposed = torch.matmul(T_markers, torch.unsqueeze(marker_posed_homo, dim=-1))[:,:,:3,0]
        markers_reposed = (fitted_scale * (markers_reposed + fitted_trans.unsqueeze(0))).squeeze()
    else:
        markers_reposed = None


    return verts_reposed, lm_reposed, markers_reposed


##############################################################################################################
# DATA UTILS
##############################################################################################################


def get_moyo_poses(data_path,sample_every_kth_pose=1,remove_hands_poses=True):
    """
    Load SMPL poses from the MOYO dataset

    :param data_path (str): path to the MOYO dataset
    :param sample_every_kth_pose (int): sample every k-th pose from the MOYO poses
    :param remove_hands_poses (str): remove hand pose parameters
    """

    # get all .pkl files
    all_files = glob(os.path.join(data_path,"*.pkl"))

    # get all poses
    all_poses = []
    for file in all_files:
        with open(file,'rb') as f:
            data = pickle.load(f)
        body_pose = data["body_pose"]
        global_orient = data["global_orient"]
        pose = np.concatenate([global_orient,body_pose],axis=1)
        if remove_hands_poses:
            pose[:,-6:] = 0 # remove hand pose
        pose = pose[::sample_every_kth_pose,:]
        all_poses.append(pose)
    
    all_poses = np.vstack(all_poses)

    return all_poses


##############################################################################################################
# MESH UTILS
##############################################################################################################


def move_points_along_mesh(mesh_verts: np.ndarray, 
                           mesh_faces: np.ndarray, 
                           points_to_move: np.ndarray, 
                           distance_to_move: List[float]):
    '''
    For each point to move from points_to_move, find a random plane and 
    move the point along the cross section of the mesh (defined with mesh_verts 
    and mesh_faces) and plane, by distance_to_move

    :param mesh_verts: np.ndarray of mesh points (N,3)
    :param mesh_faces: np.ndarray of mesh faces (K,3)
    :param points_to_move: np.ndarray of points to move (M,3)
    :param distance_to_move: list of distances to move each point from points_to_move
                            along mesh
    '''

    moved_points = np.zeros_like(points_to_move)
    N_pts_to_move = points_to_move.shape[0]

    # for each point to move, find a random plane and move the point along the cross section
    # of the mesh and plane
    random_plane_normals_vect = np.random.rand(N_pts_to_move,3) # (N_pts_to_move,3)
    random_plane_normals_vect_norms = np.linalg.norm(random_plane_normals_vect,axis=1).reshape(N_pts_to_move,-1)
    random_plane_normals_vect = random_plane_normals_vect / random_plane_normals_vect_norms # (N_pts_to_move,3)

    mesh = trimesh.Trimesh(vertices=mesh_verts, faces=mesh_faces)

    for i in range(N_pts_to_move):

        pt_to_move = points_to_move[i]
        pt_distance_to_move = distance_to_move[i]
        # find cross section
        slice_segments = trimesh.intersections.mesh_plane(mesh, 
                                                        plane_normal=random_plane_normals_vect[i], 
                                                        plane_origin=pt_to_move, 
                                                        return_faces=False) # (N1, 2, 3)
        unique_slice_points = np.unique(slice_segments.reshape(-1, 3), axis=0)

        current_point = pt_to_move.reshape((1,3))
        slice_points = unique_slice_points
        continue_moving = True
        cumulative_distance = 0.0

        while continue_moving:
            # 1. find closest of current_point and path_points
            dist_to_slice_points = np.linalg.norm(slice_points - current_point,
                                                    axis=1)
            closest_point_ind = np.argmin(dist_to_slice_points)
            closest_point_dist = dist_to_slice_points[closest_point_ind]
            closest_point = slice_points[closest_point_ind]

            # 2. if that point is going to be farther than the desired distance pt_distance_to_move
            # then move along the vector to that closest point but 
            # in a manner that the final distance is exactly pt_distance_to_move
            if (cumulative_distance + closest_point_dist) > pt_distance_to_move:
                remaining_distance = pt_distance_to_move - cumulative_distance
                final_point = current_point + remaining_distance * (closest_point - current_point)
                continue_moving = False
                
                cumulative_distance += remaining_distance
                
            # 3. if that point is not yet at the desired distance pt_distance_to_move
            # continue on the sliced path -> now the current point is the closest_point found
            # and we remove it from the slice_points in order to not walk over it again
            else:
                cumulative_distance += closest_point_dist
                current_point = closest_point
                slice_points = np.delete(slice_points,closest_point_ind,0)    

        moved_points[i] = final_point

    return moved_points


##############################################################################################################
# SAMPLING UTILS
##############################################################################################################

T_names = ['T00','T01','T02','T03',
            'T10','T11','T12','T13',
            'T20','T21','T22','T23',
            'T30','T31','T32','T33']


def pts2homo(pts):
    '''
    input pts: np array dim N x 3
    return pts: np array dim N x 4
    '''
    return np.concatenate((pts, np.ones(pts.shape[0]).reshape(-1,1)), axis=1)

def pts2homo_torch(pts):
    '''
    input pts: np array dim N x 3
    return pts: np array dim N x 4
    '''
    return torch.concat((pts, torch.ones(pts.shape[0]).reshape(-1,1)), axis=1)


def homo_matmul(pts,T): 
    '''
    inputs Nx3 pts and 4x4 transformation matrix
    '''
    pts_T = np.matmul(pts2homo(pts),T.T)
    return (pts_T / pts_T[:,3].reshape(-1,1))[:,:3]

def homo_matmul_torch(pts,T):
    '''
    inputs Nx3 pts and 4x4 transformation matrix
    '''
    pts_T = torch.matmul(pts2homo_torch(pts),T.T)
    return (pts_T / pts_T[:,3].reshape(-1,1))[:,:3]