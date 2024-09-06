
import numpy as np
import torch
import smplx
import os
import pandas as pd
from tqdm import tqdm
from glob import glob
import argparse

import sys
sys.path.append("..")
import importlib
anthropometry_module = importlib.import_module("SMPL-Anthropometry")

from utils import CAESAR_Name2Gender

#################################################################################################################
# PREQUISITES
#################################################################################################################
SIMPLE_MEASUREMENT_NAMES = ["ankle left circumference",
                                "arm length (shoulder to elbow)",
                                "arm right length",
                                "arm length (spine to wrist)",
                                "chest circumference",
                                "crotch height",
                                "head circumference",
                                "Hip circumference max height",
                                "hip circumference",
                                "neck circumference",
                                "height"]

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

CAESAR_MEASUREMENT_NAMES = [SMPL_SIMPLE2CAESAR_MEASUREMENTS_MAP[m_name] 
                            for m_name in SIMPLE_MEASUREMENT_NAMES]

def create_body_model(body_model_path: str, body_model_type: str, gender: str ="neutral", num_betas: int =10):
    '''
    Create SMPL body model
    :param: smpl_path (str): location to SMPL .pkl models
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


SMPL_A_POSE = np.load("../data/poses/a_pose.npy")
SMPL_A_POSE = torch.from_numpy(SMPL_A_POSE)
CM_TO_M = 0.01
M_TO_CM = 100
CM_TO_MM = 10

bms = {
            "MALE": create_body_model("../data/body_models",
                                      "smpl",
                                      "MALE",
                                      10),
            "FEMALE": create_body_model("../data/body_models",
                                      "smpl",
                                      "FEMALE",
                                      10),
            "NEUTRAL": create_body_model("../data/body_models",
                                      "smpl",
                                      "NEUTRAL",
                                      10),
        }


n2g = CAESAR_Name2Gender(gender_mapper_path="../data/gender/CAESAR_GENDER_MAPPER.npz")


#################################################################################################################
# GETTING BASELINE RESULTS
#################################################################################################################


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", 
                        type=str, 
                        default="../data/processed_datasets/dataset_test_unposed.npz",
                        # default="../data/processed_datasets/dataset_test_unposed_noisy",
                        # default="../data/dataset_test_sit_transferred_lm_bm.npz",
                        # default="../data/dataset_test_posed.npz",
                        help="Path to get baseline results.")
    args = parser.parse_args()


    dataset_path = args.dataset_path

    if dataset_path.endswith(".npz"):
        data = np.load(dataset_path)
        names = data["names"]
        measurements_gt = data["measurements"] # in cm, N x 11
        genders = data["genders"]
        fitted_shape = data["fit_shape"]
        N_examples = measurements_gt.shape[0]



        ae = np.zeros((N_examples,len(SIMPLE_MEASUREMENT_NAMES)))
        ae_genders = []

        for i in tqdm(range(N_examples)):
            subj_name = names[i]
            subj_gender = genders[i].upper()
            subj_shape = torch.from_numpy(fitted_shape[i]).unsqueeze(0).float()

            # measure fitted SMPL in A-POSE in cm
            shaped_body = bms[subj_gender](body_pose=SMPL_A_POSE[:,3:], 
                                        betas=subj_shape, 
                                        global_orient=SMPL_A_POSE[:,:3]).vertices[0].detach().cpu() # 1 x N x 3


            complex_measurer = anthropometry_module.MeasureBody("smpl")
            complex_measurer.from_verts(verts=shaped_body) 
            complex_measurer.measure(SIMPLE_MEASUREMENT_NAMES)
            fit_complex_measurements = np.array([complex_measurer.measurements[m_name] 
                                                    for m_name in SIMPLE_MEASUREMENT_NAMES]) # in cm
            
            ae[i,:] = np.abs(fit_complex_measurements - measurements_gt[i])
            ae_genders.append(subj_gender.upper())

        ae_genders = np.array(ae_genders)

        print(f"Evaluated on {ae.shape[0]} subjects")
        results_male = ae[np.where(ae_genders == "MALE")[0]]
        results_male = np.mean(results_male,axis=0) * CM_TO_MM
        results_female = ae[np.where(ae_genders == "FEMALE")[0]]
        results_female = np.mean(results_female,axis=0) * CM_TO_MM
        results_cum = np.mean(ae,axis=0) * CM_TO_MM


        df_res = pd.DataFrame([results_male,results_female,results_cum],
                columns=CAESAR_MEASUREMENT_NAMES,
                index=["MALE AE (mm)", "FEMALE AE (mm)","AE (mm)"]).transpose()
        df_res.loc['Average'] = df_res.mean(axis=0)
        print(df_res)

        
    else:
        all_files = glob(os.path.join(dataset_path,"*.npz"))
        N_examples = len(all_files)



        ae = np.zeros((N_examples,len(SIMPLE_MEASUREMENT_NAMES)))
        ae_genders = []

        for i in tqdm(range(N_examples)):

            subj_data = np.load(all_files[i])
            subj_measurements = subj_data["measurements"]

            subj_name = all_files[i].split("/")[-1].split(".npz")[0]
            subj_gender = n2g.get_gender(subj_name).upper()
            subj_shape = torch.from_numpy(subj_data["fit_shape"]).float()

            # measure fitted SMPL in A-POSE in cm
            shaped_body = bms[subj_gender](body_pose=SMPL_A_POSE[:,3:], 
                                        betas=subj_shape, 
                                        global_orient=SMPL_A_POSE[:,:3]).vertices[0].detach().cpu() # 1 x N x 3


            complex_measurer = anthropometry_module.MeasureBody("smpl")
            complex_measurer.from_verts(verts=shaped_body) 
            complex_measurer.measure(SIMPLE_MEASUREMENT_NAMES)
            fit_complex_measurements = np.array([complex_measurer.measurements[m_name] 
                                                    for m_name in SIMPLE_MEASUREMENT_NAMES]) # in cm
            
            ae[i,:] = np.abs(fit_complex_measurements - subj_measurements)
            ae_genders.append(subj_gender.upper())

        ae_genders = np.array(ae_genders)

        print(f"Evaluated on {ae.shape[0]} subjects")
        results_male = ae[np.where(ae_genders == "MALE")[0]]
        results_male = np.mean(results_male,axis=0) * CM_TO_MM
        results_female = ae[np.where(ae_genders == "FEMALE")[0]]
        results_female = np.mean(results_female,axis=0) * CM_TO_MM
        results_cum = np.mean(ae,axis=0) * CM_TO_MM


        df_res = pd.DataFrame([results_male,results_female,results_cum],
                columns=CAESAR_MEASUREMENT_NAMES,
                index=["MALE AE (mm)", "FEMALE AE (mm)","AE (mm)"]).transpose()
        df_res.loc['Average'] = df_res.mean(axis=0)
        print(df_res)