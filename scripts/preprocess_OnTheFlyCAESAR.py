
import torch
from tqdm import tqdm
import numpy as np
import os
from glob import glob
import pandas as pd
import smplx
import argparse

import sys
sys.path.append("..")
from utils import create_body_model, process_caesar_landmarks, SMPL_INDEX_LANDAMRKS_REVISED, load_scan, pairwise_dist, load_landmarks

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_preprocess_to", 
                        type=str, 
                        default="/data/wear3d_preprocessed/OnTheFlyCaesar/from_2023_10_18_23_32_22_and_2023_09_27_12_21_18_standing",
                        help="Path to save the preprocessed dataset.")
    parser.add_argument("--fitted_bm_path", 
                        type=str, 
                        default="/SMPL-Fitting/results/2023_09_27_12_21_18_standing",
                        help="Path to the fitted SMPL body model to the CAESAR scans.")
    parser.add_argument("--fitted_nrd_path", 
                        type=str, 
                        default="/SMPL-Fitting/results/2023_10_18_23_32_22",
                        help="Path to the fitted SMPL vertices to the CAESAR scans.")
    parser.add_argument("--body_model_path", 
                        type=str, 
                        default="/pose-independent-anthropometry/data/body_models",
                        help="Path to the SMPL body models.")
    parser.add_argument("--caesar_dataset_path", 
                        type=str, 
                        default="/data/wear3d",
                        help="Path to the SMPL body models.")
    parser.add_argument("--transferred_lm_path", 
                        type=str, 
                        default="/pose-independent-anthropometry/data/transferred_landmarks/CAESAR/partial_transfer/only_missing/standing/NRD/from_2023_10_18_23_32_22",
                        help="Path to the transferred landmarks.")
    args = parser.parse_args()


    batch_size = 1


    for CAESAR_SAMPLE in ["CAESAR_TSOLI_TRAIN", "CAESAR_TSOLI_VAL", "CAESAR_TSOLI_TEST"]:


        # CAESAR_SAMPLE = "CAESAR_GOOD_SUBJECTS_TEST"
        # CAESAR_SAMPLE = "CAESAR_TSOLI_TRAIN"
        # CAESAR_SAMPLE = "CAESAR_TSOLI_VAL"
        # CAESAR_SAMPLE = "CAESAR_TSOLI_TEST"
        with open(f"/pose-independent-anthropometry/data/CAESAR_samples/{CAESAR_SAMPLE}.txt", "r") as f:
            train_subjects = f.read().splitlines()

        # remember_preprocessed_scans_path = f"scripts/preprocessed_scans_from_{CAESAR_SAMPLE}.txt"
        # if os.path.exists(remember_preprocessed_scans_path):
        #     with open(remember_preprocessed_scans_path,"r") as f:
        #         remember_preprocessed_scans = f.read().splitlines()
        # else:
        #     remember_preprocessed_scans = []


        # save_preprocess_to = "/data/wear3d_preprocessed/OnTheFlyCaesar/from_2023_10_18_23_32_22_and_2023_09_27_12_21_18_standing"
        # fit_bm_path = "/SMPL-Fitting/results/2023_09_27_12_21_18_standing"
        # fit_nrd_path = "/SMPL-Fitting/results/2023_10_18_23_32_22"
        # body_model_path = "/SMPL-Fitting/data/body_models"
        body_model_name = "smpl"

        # use_transferred_lm_path =  "/pose-independent-anthropometry/data/transferred_landmarks/CAESAR/partial_transfer/only_missing/standing/NRD/from_2023_10_18_23_32_22" # None

        demographics_path = os.path.join(args.caesar_dataset_path, 
                                        "processed_data", 
                                        "demographics.csv")
        demographics = pd.read_csv(demographics_path)

        landmark_names = list(SMPL_INDEX_LANDAMRKS_REVISED.keys())


        body_models = {
            "MALE": create_body_model(args.body_model_path,
                                    body_model_name,
                                    "MALE",
                                    10),
            "FEMALE": create_body_model(args.body_model_path,
                                    body_model_name,
                                    "FEMALE",
                                    10),
            "NEUTRAL": create_body_model(args.body_model_path,
                                    body_model_name,
                                    "NEUTRAL",
                                    10),
        }

        country_scale_2_m = {"Italy": 1, 
                            "The Netherlands": 1000, 
                            "North America": 1} # scale verts to convert from mm to m


        for subject in tqdm(train_subjects):

            if os.path.exists(os.path.join(args.save_preprocess_to,f"{subject}.npz")):
                print("Already preprocessed")
                continue
            
            # get fits
            fit_bm_data = np.load(os.path.join(args.fitted_bm_path,f"{subject}.npz"))
            fit_pose = torch.from_numpy(fit_bm_data["pose"])
            fit_shape = torch.from_numpy(fit_bm_data["shape"])
            fit_trans = torch.from_numpy(fit_bm_data["trans"])
            fit_scale = torch.from_numpy(fit_bm_data["scale"])
            
            fit_nrd_data = np.load(os.path.join(args.fitted_nrd_path,f"{subject}.npz"))
            fit_verts = torch.from_numpy(fit_nrd_data["vertices"])
            
            # get subject vars
            scan_path = glob(f"/data/wear3d/Data AE2000/*/*/{subject}.ply.gz")[0]
            subject_number = int(subject[-5:-1])
            subject_country = scan_path.split("/")[-3]
            subject_scale = country_scale_2_m[subject_country]
            subject_gender = demographics.loc[(demographics["Country"] == subject_country) & 
                                            (demographics["Subject Number"] == subject_number), "Gender"].item().upper()
            bm = body_models[subject_gender]
            scan_verts, scan_faces = load_scan(scan_path)
            scan_verts = torch.from_numpy(scan_verts) / subject_scale
            if isinstance(args.transferred_lm_path,type(None)):
                scan_landmarks_path = scan_path.replace(".ply.gz",".lnd")
            else:
                scan_landmarks_path = os.path.join(args.transferred_lm_path,f"{subject}_landmarks.json")
            # scan_landmarks = process_caesar_landmarks(scan_landmarks_path, 1000)
            scan_landmarks = load_landmarks(landmark_path = scan_landmarks_path, 
                                            landmark_subset=None,
                                            scan_vertices=None,
                                            landmarks_scale=1000,
                                            verbose=False)
            scan_lm = torch.cat([torch.from_numpy(scan_landmarks[lm_name]).unsqueeze(0) 
                                for lm_name in landmark_names
                                if lm_name in scan_landmarks.keys()],axis=0)
            
            # get joints
            v_shaped = bm.v_template + smplx.lbs.blend_shapes(fit_shape, bm.shapedirs)
            J = smplx.lbs.vertices2joints(bm.J_regressor, v_shaped)
            
            # get pose offsets and T
            ident = torch.eye(3)
            rot_mats = smplx.lbs.batch_rodrigues(fit_pose.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
            pose_offsets = torch.matmul(pose_feature, bm.posedirs).view(batch_size, -1, 3)

            _, A = smplx.lbs.batch_rigid_transform(rot_mats, J, bm.parents)

            W = bm.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1]) 
            num_joints = bm.J_regressor.shape[0]
            T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
            
            # get indices
            dists = pairwise_dist(scan_verts.unsqueeze(0).float(),
                                fit_verts.unsqueeze(0).float())
            scan2fit_inds = torch.argmin(dists,axis=1)[0]
            
            dists = pairwise_dist(scan_lm.unsqueeze(0).float(),
                                fit_verts.unsqueeze(0).float())
            lm2fit_inds = torch.argmin(dists,axis=1)[0]
            
            np.savez(os.path.join(args.save_preprocess_to,f"{subject}.npz"),
                    fit_scale=fit_scale,
                    fit_trans=fit_trans,
                    T=T,
                    pose_offsets=pose_offsets,
                    J=J,
                    scan2fit_inds=scan2fit_inds,
                    lm2fit_inds=lm2fit_inds,
                    scan_lm=scan_lm
                    )
