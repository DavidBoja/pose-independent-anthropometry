
import argparse
from glob import glob
import os
import numpy as np
import pandas as pd
# from utils import SMPL_INDEX_LANDAMRKS_REVISED
import utils
from tqdm import tqdm

def NoisyCaesar_stats(dataset_path: str, 
                      use_landmarks: str = "SMPL_INDEX_LANDAMRKS_REVISED",
                      **kwargs):

    all_files = sorted(glob(os.path.join(dataset_path,"*.npz")))
    N_files = len(all_files)

    # process first example
    data = np.load(all_files[0])
    N_lm = data["lm_displacement"].shape[0]

    noise_per_lm = np.zeros((N_files,N_lm))
    noise_per_lm[0,:] = data["lm_displacement"]

    for i, file in tqdm(enumerate(all_files[1:])):
        data = np.load(file)
        noise_per_lm[i+1,:] = data["lm_displacement"]

    # compute stats
    landmarks_dict = getattr(utils, use_landmarks.upper(), None)
    landmark_names = list(landmarks_dict.keys())
    df = pd.DataFrame(noise_per_lm, columns=landmark_names)

    dataset_name = dataset_path.split("/")[-1]
    print(f"Landmark movement in cm for {dataset_name}")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.describe().T[["min","50%","mean","max"]])


def caesar_sitting_stats(dataset_path, use_landmarks, **kwargs):

    all_landmark_paths = glob(os.path.join(dataset_path,"Data AE2000/*/*/*b.lnd"))
    N_files = len(all_landmark_paths)

    use_landmarks = getattr(utils, use_landmarks, None)
    landmark_names = list(use_landmarks.keys())
    N_lm = len(landmark_names)

    counter = dict(zip(landmark_names,[0]*N_lm))
    counter_per_subj = {}

    for pth in tqdm(all_landmark_paths):
        landmarks = utils.load_landmarks(pth,landmarks_scale=1000, verbose=False)
        counter_per_subj[pth] = 0
        for lm_name, lm_coord in landmarks.items():
            if lm_name not in landmark_names:
                continue
            elif isinstance(lm_coord,type(None)):
                continue
            elif np.all(np.isclose(lm_coord,0.0)):
                continue
            else:
                counter[lm_name] = counter[lm_name] + 1
                counter_per_subj[pth] = counter_per_subj[pth] + 1

    subj_with_all_lm = [pth.split("/")[-1] for pth, val in counter_per_subj.items() if val >= N_lm]
    print(f"Subjects with all landmarks {len(subj_with_all_lm)} / {N_files}")
    print("Landmark counts")
    for lm_name, lm_count in counter.items():
        print(f"{lm_name}: {lm_count} / {N_files}")

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="subparsers", dest="subparser_name")


    parser_noisycaesar_stats = subparsers.add_parser('NoisyCaesar')
    parser_noisycaesar_stats.add_argument("--dataset_path", type=str,
                                          default="data/processed_datasets/dataset_test_unposed_noisy")
    parser_noisycaesar_stats.add_argument("--use_landmarks", type=str, default="SMPL_INDEX_LANDAMRKS_REVISED")
    parser_noisycaesar_stats.set_defaults(func=NoisyCaesar_stats)


    parser_caesar_sitting_stats = subparsers.add_parser('CAESAR_SITTING')
    parser_caesar_sitting_stats.add_argument("--dataset_path", type=str,
                                          default="/data/wear3d")
    parser_caesar_sitting_stats.add_argument("--use_landmarks", type=str, default="SMPL_INDEX_LANDAMRKS_REVISED")
    parser_caesar_sitting_stats.set_defaults(func=caesar_sitting_stats)
    
    args = parser.parse_args()

    args.func(**vars(args))