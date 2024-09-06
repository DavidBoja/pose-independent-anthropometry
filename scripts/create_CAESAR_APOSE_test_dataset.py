
import sys
sys.path.append("..")

import dataset
from utils import load_config
from tqdm import tqdm
import torch
import numpy as np
import argparse



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_to", 
                        type=str, 
                        default="../data/processed_datasets/dataset_test_unposed.npz",
                        help="Path to save the preprocessed dataset.")
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
                                        poses_path="../data/poses/poses_clusterSumLM10k_caesarFitApose_caesarFitBpose.npy", #NOTE: this wont be actually used so it doesnt matter
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
                                        what_to_return=["name","landmarks","measurements","gender"],
                                        augmentation_landmark_jitter_std=0,
                                        augmentation_landmark_2_origin_prob=0,
                                        augmentation_unpose_prob=0,
                                        augmentation_repose_prob=0,
                                        preprocessed_path=args.preprocessed_path,
                                        subsample_verts=1,
                                        use_moyo_poses=False,
                                        moyo_poses_path=None,
                                        remove_monster_poses_threshold=None,
                                        pose_prior_path=opt["paths"]["pose_prior_path"],
                                        use_transferred_lm_path=None,
                                        unposing_landmarks_choice="nn_to_verts"
                                      )
    
    N_subjects = len(dataset_test)

    dataset_names = []
    dataset_landmarks = []
    dataset_measurements = []
    dataset_genders = []

    for i in tqdm(range(N_subjects)):
        example = dataset_test[i]
        name = example["name"]
        landmarks = example['landmarks'] # (n_landm, 3)
        measurements_gt = example['measurements'] # (n_measurements) 
        gender = example["gender"]
        
        dataset_names.append(name)
        dataset_landmarks.append(landmarks)
        dataset_measurements.append(measurements_gt)
        dataset_genders.append(gender)

    np.savez(args.save_to, 
            names=np.array(dataset_names), 
            landmarks=torch.stack(dataset_landmarks).numpy(),
            measurements=torch.stack(dataset_measurements).numpy(),
            genders=np.array(dataset_genders))