
import argparse
import sys
import os
from tqdm import tqdm
from glob import glob
import open3d as o3d
import torch
import numpy as np

sys.path.append("..")
from utils import create_body_model, SMPL_INDEX_LANDAMRKS_REVISED


LANDMARK_ORDER = list(SMPL_INDEX_LANDAMRKS_REVISED.keys())
LANDMARK_INDICES = list(SMPL_INDEX_LANDAMRKS_REVISED.values())


BMS = {
            "MALE": create_body_model("../data/body_models",
                                      "smpl",
                                      "MALE",
                                      8),#.cuda(),
            "FEMALE": create_body_model("../data/body_models",
                                      "smpl",
                                      "FEMALE",
                                      8),#.cuda(),
            "NEUTRAL": create_body_model("../data/body_models",
                                      "smpl",
                                      "NEUTRAL",
                                      8)#.cuda(),
        }

def annotate_4DHuman_landmarks(scan_paths: str,
                               fit_paths:str,
                               save_to: str = None,
                               transfer_method: str = "simple",
                               nn_threshold: float = 0.01): # this should be in meters, so 1cm
    """
    Annotate 4DHumanOutfit landmarks for all subjects and actions in fit_paths.
    The landmarks are saved in fit_paths

    :param scan_paths (str): Path to the 4DHumanOutfit scans
    :param fit_paths (str): Path to the 4DHumanOutfit SMPL fits
    :param save_to (str): Path to save the annotated landmarks
    :param transfer_method (str): Method to transfer landmarks. 
                                Options: "simple", "nn", "nn_threshold"
                                -> simple: Use SMPL landmarks directly
                                -> nn: Transfer landmarks to nearest neighbor point on scan
                                -> nn_threshold: Transfer landmarks to nearest neighbor point on scan,
                                                    but if nn distance above threshold, use SMPL LM.
    :param nn_threshold (float): Threshold in meters for nearest neighbor (nn_threshold) transfer method. 
                                 -> If nn distance above that, use SMPL LM.
    
    """
    
    all_male_subjects = ["ben","bob","jon","leo","mat","pat","ray","ted","tom"]
    all_female_subjects = ["ada","bea","deb","gia","joy","kim","mia","sue","zoe"]
    all_subjects_names = all_male_subjects + all_female_subjects
    
    # create gender mapper
    all_genders = ["male"] * len(all_male_subjects) + ["female"] * len(all_female_subjects)
    gender_mapper = dict(zip(all_subjects_names,all_genders))

    for subj_name in tqdm(all_subjects_names):

        if not os.path.exists(os.path.join(fit_paths,subj_name)):
            continue

        all_subj_action_paths = glob(os.path.join(fit_paths,subj_name,f"{subj_name}-tig-*"))
        for subj_action_path in all_subj_action_paths:
            action_name = os.path.basename(subj_action_path).split("-")[-1]
            sequence_name = f"{subj_name}-tig-{action_name}"

            print(f"Annotating {sequence_name}")

            seq_scan_paths = os.path.join(scan_paths,subj_name,sequence_name,"OBJ_4DCVT_15k","*.obj")
            seq_scan_paths = sorted(glob(seq_scan_paths))

            fit_pose = torch.load(os.path.join(fit_paths,subj_name,sequence_name,"poses.pt"), map_location="cpu")
            fit_shape = torch.load(os.path.join(fit_paths,subj_name,sequence_name,"betas.pt"), map_location="cpu")
            fit_trans = torch.load(os.path.join(fit_paths,subj_name,sequence_name,"trans.pt"), map_location="cpu")

            subj_gender = gender_mapper[subj_name].upper()
            N_frames = fit_pose.shape[0]
            transferred_landmarks = np.zeros((N_frames,len(LANDMARK_INDICES),3))

            if transfer_method == "simple":
                lm_save_name = "landmarks_simple.pt"
            elif transfer_method == "nn":
                lm_save_name = "landmarks_nn.pt"
            elif transfer_method == "nn_threshold":
                thr_as_str = str(nn_threshold).replace(".","_")
                lm_save_name = f"landmarks_nn_thresholded_{thr_as_str}m.pt"

            save_lm_to = os.path.join(save_to,subj_name,sequence_name,lm_save_name)
            print(f"Saving to {save_lm_to}")

            if os.path.exists(save_lm_to):
                continue

            for i in range(N_frames):

                fit_verts = BMS[subj_gender](body_pose=fit_pose[i,3:].unsqueeze(0), 
                                            betas=fit_shape[i].unsqueeze(0), 
                                            global_orient=fit_pose[i,:3].unsqueeze(0),
                                            transl=fit_trans[i].unsqueeze(0)).vertices[0].detach().cpu().numpy() # 1 x N x 3

                # transfer landmarks to scan
                if transfer_method == "simple":
                    fit_lm = fit_verts[LANDMARK_INDICES, :]
                    transferred_landmarks[i] = fit_lm
                elif transfer_method == "nn":
                    try:
                        scan = o3d.io.read_triangle_mesh(seq_scan_paths[i])
                    except Exception as e:
                        print(f"Error reading {seq_scan_paths[i]}")
                        continue
                    scan_verts = np.asarray(scan.vertices)

                    for iter_ind, lm_ind in enumerate(LANDMARK_INDICES):
                        lm_coord = fit_verts[lm_ind,:]


                        # find closest scan point
                        dists = np.sqrt(np.sum((scan_verts - lm_coord)**2,axis=1))
                        closest_point_ind = np.argmin(dists)
                        scan_lm = scan_verts[closest_point_ind,:]
                        
                        transferred_landmarks[i,iter_ind,:] = scan_lm
                elif transfer_method == "nn_threshold":
                    scan = o3d.io.read_triangle_mesh(seq_scan_paths[i])
                    scan_verts = np.asarray(scan.vertices)

                    for iter_ind, lm_ind in enumerate(LANDMARK_INDICES):
                        lm_coord = fit_verts[lm_ind,:]

                        # find closest scan point
                        dists = np.sqrt(np.sum((scan_verts - lm_coord)**2,axis=1))
                        if np.min(dists) < nn_threshold:
                            closest_point_ind = np.argmin(dists)
                            scan_lm = scan_verts[closest_point_ind,:]
                        else:
                            scan_lm = lm_coord
                        
                        transferred_landmarks[i,iter_ind,:] = scan_lm

            # save_landmarks
            torch.save(torch.from_numpy(transferred_landmarks),
                       os.path.join(save_to,subj_name,sequence_name,lm_save_name))




    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_paths", type=str, default="/data/FourDHumanOutfit/SCANS")
    parser.add_argument("--fit_paths", type=str, default="/FourDHumanOutfit-FITS")
    # parser.add_argument("--save_to", type=str, default="/FourDHumanOutfit-FITS")
    parser.add_argument("--transfer_method", choices=["simple","nn", "nn_threshold"], type=str, default="simple")
    parser.add_argument("--nn_threshold", type=float, default=0.01,
                        help="Threshold in meters for nearest neighbor transfer method. If nn above that, use SMPL LM.")
    args = parser.parse_args()

    save_to = args.fit_paths

    annotate_4DHuman_landmarks(scan_paths=args.scan_paths,
                               fit_paths=args.fit_paths,
                               save_to=save_to,
                               transfer_method=args.transfer_method,
                               nn_threshold=args.nn_threshold)