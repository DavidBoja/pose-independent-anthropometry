
import sys
sys.path.append("..")

import dataset
from utils import (load_config, frontal_normalization,
                  SMPL_INDEX_LANDAMRKS_REVISED
                  )
from tqdm import tqdm
import torch
import numpy as np
import os
import open3d as o3d
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_to", 
                        type=str, 
                        default="/data/wear3d_preprocessed/data_train_frontal_APOSE", 
                        help="Path to save the preprocessed dataset.")
    parser.add_argument("--use_subjects_path", 
                        type=str, 
                        default="../data/CAESAR_samples/CAESAR_TSOLI_VAL_WITHOUT_BAD.txt",
                        help="Path to the validation subjects to use.")
    parser.add_argument("--visible_lm_threshold", 
                        type=float, 
                        default=1.0,
                        help="If LM closer than threshold to points of partial scan, then LM visible. IN CM.")
    parser.add_argument("--viewpoint_coord", 
                        nargs="+", 
                        default=[-100, -100, 0],
                        help="Viewpoint camera location. Only first three items taken")
    args = parser.parse_args()


    # setup
    VISIBLE_LM_THRESHOLD = args.visible_lm_threshold # 1cm?
    VIEWPOINT_COORD = np.array([float(x) for x in args.viewpoint_coord][:3])

    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)

    opt = load_config("../configs/config_real.yaml")


    # load data
    dataset_val = dataset.OnTheFlyCAESAR(caesar_dir=opt["paths"]["caesar_dir"],
                                    fitted_bm_dir=opt["paths"]["fitted_bm_dir"],
                                    fitted_nrd_dir=opt["paths"]["fitted_nrd_dir"],
                                    poses_path="/data/aist_plus_plus/poses_test_400.npz", #NOTE: not posing so not important
                                    iterate_over_poses=False,
                                    iterate_over_subjects=True,
                                    single_fixed_pose_ind=0,
                                    n_poses=None,
                                    dont_pose=True,
                                    fix_dataset=None,
                                    load_countries=["Italy","North America"],
                                    pose_params_from='all',
                                    body_model_name="smpl",
                                    body_models_path=opt["paths"]["body_models_path"],
                                    body_model_num_shape_param=10,
                                    use_measurements=opt["learning"]["measurements"],
                                    use_subjects=args.use_subjects_path,
                                    use_landmarks=opt["learning"]["landmarks"],
                                    landmark_normalization=None,
                                    what_to_return=["name","landmarks","measurements","gender","vertices"],
                                    augmentation_landmark_jitter_std=0,
                                    augmentation_landmark_2_origin_prob=0,
                                    augmentation_unpose_prob=0,
                                    augmentation_repose_prob=0,
                                    preprocessed_path=opt["paths"]["preprocessed_path"],
                                    subsample_verts=1,
                                    use_moyo_poses=False,
                                    moyo_poses_path=opt["paths"]["moyo_poses_path"],
                                    remove_monster_poses_threshold=None,
                                    pose_prior_path=opt["paths"]["pose_prior_path"],
                                    use_transferred_lm_path=None,
                                    unposing_landmarks_choice="nn_to_verts", # "nn_to_smpl"
                                    mocap_marker_path=None
                                  )
    
    LANDMARK_NAMES = dataset_val.landmark_names
    LANDMARK_SUBSTERNALE_IND = LANDMARK_NAMES.index("Substernale")
    LANDMARK_SUPRAMENTON_IND = LANDMARK_NAMES.index("Supramenton")
    LANDMARK_THELION_RT_IND = LANDMARK_NAMES.index("Rt. Thelion/Bustpoint")
    LANDMARK_THELION_LT_IND = LANDMARK_NAMES.index("Lt. Thelion/Bustpoint")
    N_subjects = len(dataset_val)


    for i in tqdm(range(N_subjects)):
        
    
        # load data
        example = dataset_val[i]
        example_name = example["name"]
        example_lm = example["landmarks"].numpy()
        example_meas = example["measurements"]
        example_gender = example["gender"]
        example_vertices = example["vertices"]
        N_vertices = example_vertices.shape[0]
        # example_faces = example["faces"]
        
        example_o3d = o3d.geometry.PointCloud()
        example_o3d.points = o3d.utility.Vector3dVector(example_vertices)

        # variables
        diameter = np.linalg.norm(
            np.asarray(example_o3d.get_max_bound()) - np.asarray(example_o3d.get_min_bound()))
        radius = diameter * 1000

        
        # get partial vertices
        _, pt_map = example_o3d.hidden_point_removal(VIEWPOINT_COORD, radius)
        partial_vertices = np.asarray(example_o3d.points)[pt_map,:]
        N_partial_vertices = partial_vertices.shape[0]
        partiality_percentage_verts = (N_partial_vertices / N_vertices)
        
        
        # get partial landmarks
        indices_lm = []
        for lm_ind,lm in enumerate(example_lm):
            dist = np.sqrt(np.sum((partial_vertices - lm)**2, axis=1))
            if any(dist < VISIBLE_LM_THRESHOLD):
                indices_lm.append(lm_ind)    
        partiality_percentage_lm = len(indices_lm) / example_lm.shape[0]


        # normalize frontal partial scan
        example_lm, centroid, R2y, R2x = frontal_normalization(
                                                            torch.from_numpy(example_lm),
                                                            LANDMARK_SUBSTERNALE_IND, 
                                                            LANDMARK_SUPRAMENTON_IND,
                                                            LANDMARK_THELION_RT_IND,
                                                            LANDMARK_THELION_LT_IND,
                                                            return_transformations=True)
        

        np.savez(os.path.join(args.save_to, f"{example_name}.npz"),
                camera=VIEWPOINT_COORD,
                indices_verts=np.array(pt_map),
                partiality_verts=np.array(partiality_percentage_verts),
                indices_lm=np.array(indices_lm),
                partiality_lm=np.array(partiality_percentage_lm),
                landmarks=example_lm,
                measurements=example_meas.numpy()
                )
        