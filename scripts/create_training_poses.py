
import numpy as np
import os
from glob import glob
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_to", 
                        type=str, 
                        default="../data/poses", # csr4415a
                        help="Path to save poses to.")
    parser.add_argument("--poses_path", 
                        type=str, 
                        default="../data/smpl_train_poses.npz",
                        help="Path to all training poses.")
    parser.add_argument("--clustered_poses_path_root", 
                        type=str, 
                        default="../data/pose_similarity_clustering",
                        help="Path to clustered poses.")
    parser.add_argument("--fitted_bm_dir", 
                        type=str, 
                        default="/SMPL-Fitting/results/2023_09_27_12_21_18",
                        help="Path to fitted SMPL body model.")
    parser.add_argument("--subject_names", 
                        type=str, 
                        default="../data/CAESAR_samples/CAESAR_GOOD_SUBJECTS_TRAIN.txt",
                        help="Use these subjects poses.")
    args = parser.parse_args()

    caesar_train_subjects_path = args.subject_names



    cluster_centers_indices = np.load(os.path.join(args.clustered_poses_path_root,
        "CLUSTER_CENTERS_INDICES_FROM_ORIG_DATA_410630POSES_SUM_LM_DIST_MINIBATCH_KMEANS_10000_clusters.npy"))


    with open(caesar_train_subjects_path, "r") as f:
        caesar_train_subjects = f.read().splitlines()

    # 1. get training poses
    training_poses = np.load(args.poses_path)
    training_poses = training_poses["poses"]

    # 2. cluster poses
    training_poses_clustered = training_poses[cluster_centers_indices]

    # 3. get A-fitting poses -> some of them
    a_pose_fitting_files = glob(os.path.join(args.fitted_bm_dir,"*a.npz"))
    standing_poses = []

    for fl in a_pose_fitting_files:
        scan_name = fl.split("/")[-1].split(".")[0]
        if scan_name in caesar_train_subjects:
            standing_poses.append(np.load(fl)["pose"])
    standing_poses = np.array(standing_poses)
    standing_poses = standing_poses[:1000,0,:]

    # 3. get sitting fitted poses -> some of them
    sitting_fitting_files = glob(os.path.join(args.fitted_bm_dir,"*b.npz"))
    sitting_poses = []

    for fl in sitting_fitting_files:
        scan_name = fl.split("/")[-1].split(".")[0]
        scan_name_without_posing = scan_name[:-1] + "a"
        if scan_name_without_posing in caesar_train_subjects:
            sitting_poses.append(np.load(fl)["pose"])
    sitting_poses = np.array(sitting_poses)
    sitting_poses = sitting_poses[:1000,0,:]


    # 4. merge all poses and save
    merged_poses = np.concatenate([training_poses_clustered,standing_poses,sitting_poses])
    np.save(os.path.join(args.save_to,"poses_clusterSumLM10k_caesarFitApose_caesarFitBpose.npy"), merged_poses)