import sys
sys.path.append("..")

from utils import (load_config, 
                   pairwise_dist, 
                   load_landmarks, 
                   SMPL_INDEX_LANDAMRKS_REVISED)

import torch
from tqdm import tqdm
import pandas as pd
import os
from glob import glob
import numpy as np
import argparse


from dataset import OnTheFlyCAESAR




opt = load_config("../configs/config_real.yaml")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", 
                        type=str, 
                        default="csr2895a", # csr4415a
                        help="Subject used to to find pose invariant landmark distances.")
    parser.add_argument("--caesar_dir", 
                        type=str, 
                        default="/data/wear3d",
                        help="Subject used to to find pose invariant landmark distances.")
    args = parser.parse_args()

    SUBJECT = args.subject
    CAESAR_DIR = args.caesar_dir
    LANDMARK_NAMES = list(SMPL_INDEX_LANDAMRKS_REVISED.keys())
    LANDMARK_INDICES = list(SMPL_INDEX_LANDAMRKS_REVISED.values())


    if not os.path.exists("pose_invariant_features_mean.csv"):

        # find a-pose LM distances for subject - the subject is csr2895a
        subject_landmark_path = glob(f"{CAESAR_DIR}/Data AE2000/*/*/{SUBJECT}.lnd")[0]
        print(subject_landmark_path)
        subject_landmarks = load_landmarks(subject_landmark_path)
        subject_landmarks = torch.cat([torch.from_numpy(subject_landmarks[lm_name]).unsqueeze(0) 
                                        for lm_name in LANDMARK_NAMES
                                        if lm_name in subject_landmarks.keys()],axis=0)
        subject_landmarks = subject_landmarks * 100 # scale to cm

        dist_a_pose = pairwise_dist(subject_landmarks.unsqueeze(0).float(),
                                    subject_landmarks.unsqueeze(0).float()).squeeze()
        dist_a_pose = torch.sqrt(dist_a_pose)
        dist_a_pose = dist_a_pose.cuda()

        dataset_train = OnTheFlyCAESAR(
            caesar_dir=opt["paths"]["caesar_dir"],
            fitted_bm_dir=opt["paths"]["fitted_bm_dir"],
            fitted_nrd_dir=opt["paths"]["fitted_nrd_dir"],
            poses_path="../data/poses/smpl_train_poses.npz",
            iterate_over_poses=True,
            iterate_over_subjects=False,
            single_fixed_pose_ind=None,
            n_poses=None,
            dont_pose=False,
            load_countries="All",
            pose_params_from='all',
            body_model_name="smpl",
            body_model_path=opt["paths"]["body_models_path"],
            body_model_num_shape_param=10,
            use_measurements=opt["learning"]["measurements"],
            use_subjects=[SUBJECT],
            use_landmarks=opt["learning"]["landmarks"],
            landmark_normalization=opt["learning"]["landmark_normalization"],
            what_to_return=["landmarks","measurements","gender"],
            augmentation_landmark_jitter_std=0,
            augmentation_landmark_2_origin_prob=0, 
            augmentation_unpose_prob=1,
            augmentation_repose_prob=1,
            preprocessed_path=opt["paths"]["preprocessed_path"],
            subsample_verts=1,
            use_moyo_poses=False,
            moyo_poses_path=None,
            remove_monster_poses_threshold=1000,
            pose_prior_path=opt["paths"]["pose_prior_path"],
            fix_dataset=False,
            mocap_marker_path=None, #"/data/wear3d_preprocessed/mocap/simple+nn/standard/from_2023_10_18_23_32_22"
            )

        N_landmarks = len(LANDMARK_NAMES)
        N_dataset_train = len(dataset_train)

        distance_tracker = torch.zeros((N_dataset_train,N_landmarks,N_landmarks)) # basically AE

        for i in tqdm(range(N_dataset_train)):
            example = dataset_train[i]
            landmarks = example['landmarks'].cuda() # (n_markers, 3)

            dists = pairwise_dist(landmarks.unsqueeze(0).float(),
                                landmarks.unsqueeze(0).float()).squeeze()
            dists = torch.sqrt(dists)
            distance_tracker[i] =  torch.abs(dists - dist_a_pose).detach().cpu()


        distance_tracker_mean = torch.mean(distance_tracker,dim=0).numpy()
        distance_tracker_median = torch.median(distance_tracker,dim=0)[0].numpy()
        distance_tracker_std = torch.std(distance_tracker,dim=0).numpy()
        distance_tracker_max = torch.max(distance_tracker,dim=0)[0].numpy()

        print(distance_tracker_mean.shape)
        print(distance_tracker_median.shape)
        print(distance_tracker_std.shape)
        print(distance_tracker_max.shape)

        packed_stats = [distance_tracker_mean, 
                        distance_tracker_median, 
                        distance_tracker_std, 
                        distance_tracker_max]

        for name, stats in zip(["mean","median","std","max"],packed_stats):
            df = pd.DataFrame(stats, 
                            columns=LANDMARK_NAMES, 
                            index=LANDMARK_NAMES)
            df.to_csv(f"pose_invariant_landmarks_{name}.csv")

    

    df_max = pd.read_csv("pose_invariant_landmarks_max.csv", index_col=0)
    df_median = pd.read_csv("pose_invariant_landmarks_median.csv", index_col=0)
    df_mean = pd.read_csv("pose_invariant_landmarks_mean.csv", index_col=0)
    df_std = pd.read_csv("pose_invariant_landmarks_std.csv", index_col=0)

    columns = df_max.columns

    # 1. features where max dist is less than 1 cm
    feature_names = []
    feature_inds = []
    for c_name1 in columns:
        condition = df_max[c_name1] < 1
        if sum(condition) > 0:
            df_max_sub = df_max[condition]
            for c_name2 in list(df_max_sub.index):
                feature_names.append((c_name1,c_name2))
                feature_inds.append((LANDMARK_NAMES.index(c_name1),
                                    LANDMARK_NAMES.index(c_name2)))

    feature_inds_npy = np.array(feature_inds)
    np.save("/SMPL-Fitting/data/landmarks2features/lm2features_distances_grouped_from_SMPL_INDEX_LANDAMRKS_REVISED_inds_removed_inds_with_max_dist_bigger_than_one_cm_considered_all_features.npy", 
            feature_inds_npy)

    # create features where mean dist is less than 1 cm
    feature_names = []
    feature_inds = []
    for c_name1 in columns:
        condition = df_mean[c_name1] < 1
        if sum(condition) > 0:
            df_mean_sub = df_mean[condition]
            for c_name2 in list(df_mean_sub.index):
                feature_names.append((c_name1,c_name2))
                feature_inds.append((LANDMARK_NAMES.index(c_name1),
                                    LANDMARK_NAMES.index(c_name2)))

    feature_inds_npy = np.array(feature_inds)
    np.save("/SMPL-Fitting/data/landmarks2features/lm2features_distances_grouped_from_SMPL_INDEX_LANDAMRKS_REVISED_inds_removed_inds_with_mean_dist_bigger_than_one_cm_considered_all_features.npy", 
            feature_inds_npy)
    
    # create features where median dist is less than 1 cm
    feature_names = []
    feature_inds = []
    for c_name1 in columns:
        condition = df_median[c_name1] < 1
        if sum(condition) > 0:
            df_median_sub = df_median[condition]
            for c_name2 in list(df_median_sub.index):
                feature_names.append((c_name1,c_name2))
                feature_inds.append((LANDMARK_NAMES.index(c_name1),
                                    LANDMARK_NAMES.index(c_name2)))
                
    feature_inds_npy = np.array(feature_inds)
    np.save("/SMPL-Fitting/data/landmarks2features/lm2features_distances_grouped_from_SMPL_INDEX_LANDAMRKS_REVISED_inds_removed_inds_with_median_dist_bigger_than_one_cm_considered_all_features.npy", 
            feature_inds_npy)