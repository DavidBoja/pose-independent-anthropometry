
import sys
sys.path.append("..")

import dataset
from utils import (load_config, SMPL_INDEX_LANDAMRKS_REVISED)
from tqdm import tqdm
import numpy as np
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_to", 
                        type=str, 
                        default="../data/processed_datasets/dataset_test_posed.npz",
                        help="Path to save the preprocessed dataset.")
    parser.add_argument("--use_subjects_path", 
                        type=str, 
                        default="../data/CAESAR_samples/CAESAR_TSOLI_TEST_WITHOUT_BAD.txt",
                        help="Path to the testing subjects to use.")
    args = parser.parse_args()


    opt = load_config("../configs/config_real.yaml")


    dataset_test = dataset.OnTheFlyCAESAR(caesar_dir=opt["paths"]["caesar_dir"],
                                    fitted_bm_dir=opt["paths"]["fitted_bm_dir"],
                                    fitted_nrd_dir=opt["paths"]["fitted_nrd_dir"],
                                    poses_path="data/aist_plus_plus_poses_test_400.npz",
                                    iterate_over_poses=False,
                                    iterate_over_subjects=True,
                                    single_fixed_pose_ind=None,
                                    n_poses=None,
                                    dont_pose=False,
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
                                    augmentation_unpose_prob=1,
                                    augmentation_repose_prob=1,
                                    preprocessed_path=None,#opt["paths"]["preprocessed_path"],
                                    subsample_verts=1,
                                    use_moyo_poses=False,
                                    moyo_poses_path=None,
                                    gather_augmentation_stats=False,
                                    remove_monster_poses_threshold=None,
                                    pose_prior_path=opt["paths"]["pose_prior_path"],
                                    use_transferred_lm_path=None
                                  )


    N_subjects = len(dataset_test)

    names = []
    landmarks = []
    measurements = []
    genders = []


    for i in tqdm(range(N_subjects)):
        
        # load data
        example = dataset_test[i]
        example_name = example["name"]
        example_lm = example["landmarks"].numpy()
        example_meas = example["measurements"].numpy()
        example_gender = example["gender"]
        
        # save data
        names.append(example_name)
        landmarks.append(example_lm)
        measurements.append(example_meas)
        genders.append(example_gender)


    np.savez(args.save_to,
                names=np.array(names),
                genders=np.array(genders),
                landmarks=np.array(landmarks),
                measurements=np.array(measurements)
                )
    