import sys
sys.path.append("..")

import dataset
from utils import (load_config, 
                   SMPL_INDEX_LANDAMRKS_REVISED, 
                  move_points_along_mesh)
from tqdm import tqdm
import numpy as np
import os
from glob import glob
import random
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_to", 
                        type=str, 
                        default="../data/processed_datasets/dataset_test_unposed_noisy",
                        help="Path to save the preprocessed dataset.")
    parser.add_argument("--move_landmarks_lower_bound", 
                        type=float, 
                        default=0.0,
                        help="Lower bound for moving the landmarks on the body.")
    parser.add_argument("--move_landmarks_upper_bound", 
                        type=float, 
                        default=0.56,
                        help="Upper bound for moving the landmarks on the body.")
    parser.add_argument("--use_subjects_path", 
                        type=str, 
                        default="../data/CAESAR_samples/CAESAR_TSOLI_TEST_WITHOUT_BAD.txt",
                        help="Path to the testing subjects to use.")
    parser.add_argument("--preprocessed_path", 
                        type=str, 
                        default=None,
                        help="Preprocessed files for unposing CAESAR.")
    args = parser.parse_args()


    opt = load_config("../configs/config_real.yaml")

    dataset_test = dataset.OnTheFlyCAESAR(caesar_dir=opt["paths"]["caesar_dir"],
                                        fitted_bm_dir=opt["paths"]["fitted_bm_dir"],
                                        fitted_nrd_dir=opt["paths"]["fitted_nrd_dir"],
                                        poses_path="../data/poses/poses_clusterSumLM10k_caesarFitApose_caesarFitBpose.npy", # NOTE: doesnt matter -> using APOSE
                                        iterate_over_poses=False,
                                        iterate_over_subjects=True,
                                        single_fixed_pose_ind=0,
                                        n_poses=None,
                                        dont_pose=True,
                                        load_countries=["Italy","North America"],
                                        pose_params_from='all',
                                        body_model_name="smpl",
                                        body_model_path=opt["paths"]["body_models_path"],
                                        body_model_num_shape_param=10,
                                        use_measurements=opt["learning"]["measurements"],
                                        use_subjects=args.use_subjects_path,
                                        use_landmarks=opt["learning"]["landmarks"],
                                        landmark_normalization="pelvis",
                                        what_to_return=["name","landmarks","measurements","gender","vertices","faces"],
                                        augmentation_landmark_jitter_std=0,
                                        augmentation_landmark_2_origin_prob=0,
                                        augmentation_unpose_prob=0,
                                        augmentation_repose_prob=0,
                                        preprocessed_path=None,
                                        subsample_verts=1,
                                        use_moyo_poses=False,
                                        moyo_poses_path=None,
                                        remove_monster_poses_threshold=None,
                                        pose_prior_path=opt["paths"]["pose_prior_path"],
                                        use_transferred_lm_path=None,
                                        unposing_landmarks_choice="nn_to_verts"
                                    )



    N_subjects = len(dataset_test)
    MOVE_LANDMARKS_LOWER_BOUND = args.move_landmarks_lower_bound
    MOVE_LANDMARKS_UPPER_BOUND = args.move_landmarks_upper_bound

    for i in tqdm(range(N_subjects)):

        # load data
        example = dataset_test[i]
        example_name = example["name"]
        example_lm = example["landmarks"]
        example_meas = example["measurements"]
        example_gender = example["gender"]
        example_vertices = example["vertices"]
        example_faces = example["faces"]
        
        
        N_LM = example_lm.shape[0]
        
        if MOVE_LANDMARKS_LOWER_BOUND == MOVE_LANDMARKS_UPPER_BOUND:
            move_landmarks_by = [MOVE_LANDMARKS_LOWER_BOUND] * N_LM
        else:
            move_landmarks_by = [random.uniform(MOVE_LANDMARKS_LOWER_BOUND, MOVE_LANDMARKS_UPPER_BOUND) 
                                for _ in range(N_LM)]
        
        while True:
            try:
                example_lm_noisy = move_points_along_mesh(mesh_verts=example_vertices.squeeze().detach().cpu().numpy(), 
                                                        mesh_faces=example_faces, 
                                                        points_to_move=example_lm.detach().cpu().numpy(), 
                                                        distance_to_move=move_landmarks_by)
                break
            except:
                print("Error moving landmarks. Retrying...")
                continue

        np.savez(os.path.join(args.save_to, f"{example_name}.npz"),
                    lm_displacement=np.array(move_landmarks_by),
                    landmarks=example_lm,
                    landmarks_noisy=example_lm_noisy,
                    measurements=example_meas.numpy()
                    )

    
    