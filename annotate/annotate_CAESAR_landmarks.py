
import argparse
import sys
import os
from glob import glob
import numpy as np
import json
from tqdm import tqdm

sys.path.append("..")
from utils import (load_config,
                   SMPL_INDEX_LANDAMRKS_REVISED, 
                   process_caesar_landmarks)


def annotate_caesar_landmarks(caesar_path: str,
                             fitting_path: str,
                             save_to: str = None,
                             bm_landmarks: str = None,
                             transfer_method: int = 1):
    """
    Annotate the caesar dataset using the landmarks defined on the
    fitted body model.

    :param caesar_path (str): path to the CAESAR dataset
    :param fitting_path (str): path to fitted BM to CAESAR dataset
    :param save_to (str) path to save landamrks
    :param bm_landmarks (str): landmarks defined on the BM to transfer to scan
    :param transfer_method (int): transfer landmarks by:    
                            1) just use fitted BM landmarks
                            2) only transfer rib landmarks, rest keep as is
                            3) only transfer LMs that originally are (0,0,0) or None
    """

    caesar_path = os.path.join(caesar_path, "Data AE2000")

    # create saving dir
    fitting_path_dir_name = fitting_path.split("/")[-1]
    save_to = os.path.join(save_to, f"from_{fitting_path_dir_name}")
    os.makedirs(save_to, exist_ok=True)
    print("Saving to: ", save_to)

    # load params
    cfg = load_config(os.path.join(fitting_path,"config.yaml"))
    if isinstance(bm_landmarks, type(None)):
        body_model_type = cfg["body_model"]
        bm_landmark_dict_name = f"{body_model_type.upper()}_INDEX_LANDMARKS_REVISED"
    else:
        bm_landmark_dict_name = bm_landmarks
    # bm_landmark_dict = getattr(landmarks,bm_landmark_dict_name)
    bm_landmark_dict = eval(bm_landmark_dict_name)
    bm_landmark_dict_names = list(bm_landmark_dict.keys())
    bm_landmark_dict_inds = list(bm_landmark_dict.values())


    # load fittings
    fitted_npz_paths = glob(f"{fitting_path}/*.npz")
    for fitted_npz_path in tqdm(fitted_npz_paths):
        # print(fitted_npz_path)
        scan_name = fitted_npz_path.split("/")[-1].split(".")[0]

        try:
            fitted_params = np.load(fitted_npz_path)
            bm_verts = fitted_params["vertices"]
            bm_lm = bm_verts[bm_landmark_dict_inds,:]
        except Exception as e:
            print(f"Error loading {fitted_npz_path}. Skipping.")
            continue

        if transfer_method == 1:

            transferred_lm = dict(zip(bm_landmark_dict_names,bm_lm.tolist()))

        elif transfer_method == 2:
            scan_lm_path = glob(f"{caesar_path}/*/*/{scan_name}.lnd")[0]
            if not os.path.exists(scan_lm_path):
                continue
            scan_lm = process_caesar_landmarks(scan_lm_path, 1000)
            transferred_lm = scan_lm

            # transfer rib landmarks
            for lm_name in ["10th Rib Midspine", "Lt. 10th Rib", "Rt. 10th Rib"]:
                lm_ind = bm_landmark_dict_names.index(lm_name)
                transferred_lm[lm_name] = bm_lm[lm_ind,:]

            for lm_name, lm_coord in transferred_lm.items():
                transferred_lm[lm_name] = lm_coord.tolist()
        elif transfer_method == 3:

            # load scan landmarks
            scan_lm_path = glob(f"{caesar_path}/*/*/{scan_name}.lnd")[0]
            if not os.path.exists(scan_lm_path):
                continue
            scan_lm = process_caesar_landmarks(scan_lm_path, 1000)
            transferred_lm = scan_lm

            # load fit landmarks
            # scan_name = fitted_npz_path.split("/")[-1].split(".")[0]
           
            # transfer missing landmarks
            for lm_name in bm_landmark_dict_names:
                # if there is no landmark, 
                # or landmark is [0,0,0] 
                # or landmark is None
                if (lm_name not in transferred_lm) or \
                    isinstance(transferred_lm[lm_name],type(None)) or \
                    np.all(np.isclose(transferred_lm[lm_name],0.0)):

                    bm_lm_ind = bm_landmark_dict[lm_name]
                    transferred_lm[lm_name] = bm_verts[bm_lm_ind,:]

            for lm_name, lm_coord in transferred_lm.items():
                transferred_lm[lm_name] = lm_coord.tolist()

            # current_keys = list(transferred_lm.keys()).copy()
            # for lm_name in current_keys:
            #     if lm_name not in bm_landmark_dict_names:
            #         print(f"POP {lm_name}")
            #         transferred_lm.pop(lm_name, None)
        else:
            raise ValueError("transfer_method must be 1, 2 or 3. Check help.")
        

        # save landmarks
        save_landmarks_as = os.path.join(save_to,
                                    f"{scan_name}_landmarks.json")
        with open(save_landmarks_as,"w") as f:
            json.dump(transferred_lm, f)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caesar_path", type=str, default="/data/wear3d")
    parser.add_argument("--fitting_path", type=str, required=True, 
                        help="Path to the BM / vertex fitting to the CAESAR dataset.")
    parser.add_argument("--save_to", type=str, default="../data/transferred_landmarks/CAESAR/partial_transfer/only_missing/sitting/BM", required=True,
                        help="Path to save landamrks. Saved as scanname_landmarks.json.")
    parser.add_argument("--bm_landmarks", type=str, default="SMPL_INDEX_LANDAMRKS_REVISED", 
                        required=False,
                        help="Landmarks defined on the BM to transfer to scan.")
    parser.add_argument("--transfer_method", type=int, default=3, choices=[1,2,3],
                        help="Transfer landmarks by:\n \
                            1) (simple) just use fitted BM landmarks \n \
                            2) partial transfer - only transfer the 3 rib landamrks, rest keep as is (partial/only_ribs) \n \
                            3) partial transfer - only transfer LMs that originally have (0,0,0) or None into (partial/only_missing)"
                        )
    args = parser.parse_args()

    annotate_caesar_landmarks(args.caesar_path, 
                              args.fitting_path, 
                              args.save_to,
                              args.bm_landmarks,
                              args.transfer_method)