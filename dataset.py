
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import os
import pandas as pd
from glob import glob
import gzip
import tempfile
import open3d as o3d
from typing import List
import argparse
from sklearn.cluster import MiniBatchKMeans
import pickle
from copy import deepcopy
from tqdm import tqdm
import h5py
import sys


from utils import (create_body_model, 
                   SMPL_SIMPLE_LANDMARKS, 
                   SMPL_SIMPLE_MEASUREMENTS, 
                   SMPL_SIMPLE_MEASUREMENTS_REVISED,
                   get_normalizing_landmark,
                   process_caesar_landmarks,
                   pelvis_normalization,
                   unpose_caesar,
                   repose_caesar,
                   load_scan,
                   get_moyo_poses,
                   load_landmarks)
import utils
from losses import MaxMixturePrior


class DYNA(Dataset):
    '''
    DYNA dataset
    y-axis is height
    orig scale is meters --> convert to cm
    measurements are in cm
    '''
    def __init__(self, 
                 dataset_path: str = "/data/DYNA",
                 subject_id: str = "50009",
                 subject_action: str = "jumping_jacks",
                 use_landmarks: str = "SMPL_INDEX_LANDAMRKS_REVISED",
                 measurement_type: str = None,
                 pelvis_normalization: bool = False,
                 **kwargs):
        
        """
        :param dataset_path: (str) path to DYNA dataset
        :param subject_id: (str) subject id
        :param subject_action: (str) action performed by subject
        :param use_landmarks: (str) landmarks defined on the SMPL model from utils.py
        :param measurement_type: (str) type of measurements to load
                                    -> npy measurement file per subject
                                    with potentially multiple measurements
                                    not necessary for evaluation on DYNA
        :param pelvis_normalization: (bool) whether to normalize landmarks to pelvis
        """

        self.subject_id = subject_id

        # pelvis normalization
        self.pelvis_normalization = pelvis_normalization
        landmarks_dict = getattr(utils, use_landmarks.upper(), None)
        self.landmark_names = list(landmarks_dict.keys())

        self.rt_psis_ind = self.landmark_names.index("Rt. PSIS")
        self.lt_psis_ind = self.landmark_names.index("Lt. PSIS")
        self.rt_asis_ind = self.landmark_names.index("Rt. ASIS")
        self.lt_asis_ind = self.landmark_names.index("Lt. ASIS")
        self.nuchale_ind = self.landmark_names.index("Nuchale")

        
        
        genders = {'50002': 'male',
                '50007': 'male',
                '50009': 'male',
                '50026': 'male',
                '50027': 'male',
                '50004': 'female',
                '50020': 'female',
                '50021': 'female',
                '50022': 'female',
                '50025': 'female'}

        self.subject_gender = genders[subject_id]
        data_all_subjects = h5py.File(os.path.join(dataset_path,
                                f"dyna_{self.subject_gender.lower()}.h5"), 'r')
        seq_name = f"{subject_id}_{subject_action}"
        self.subject_sequence_data = data_all_subjects[seq_name][()].transpose(2,0,1)

        self.scan_faces = data_all_subjects["faces"][()]


        if not isinstance(measurement_type, type(None)):
            measurements_dir_path = os.path.join(dataset_path, "measurements")
            self.subject_measurements = np.load(os.path.join(measurements_dir_path, 
                                                    f"{subject_id}.npz"))[measurement_type]
            self.subject_measurements = torch.from_numpy(self.subject_measurements)
        else:
            self.subject_measurements = None


        landmark_dict = getattr(utils, use_landmarks.upper(), None)
        landmark_names = list(landmark_dict.keys())
        self.landmark_indices = list(landmark_dict.values())

        self.SCALE_TO_CM = 100

        

    def __getitem__(self, index):
        """
        :return (dict): dictionary with keys:
            "name": name of scan
            "vertices": (N,3) np.array
            "faces": (N,3) np.array or None if no faces
            "landmarks": dict with landmark names as keys and landmark coords as values
                         landmark coords are (1,3) np.array or None if no landmarks
        """

        # load scan
        scan_vertices = self.subject_sequence_data[index]
        scan_vertices = torch.from_numpy(scan_vertices)

        # load landmarks
        landmarks = scan_vertices[self.landmark_indices,:]

        # load measurements
        measurements = self.subject_measurements


        # pelvis normalization
        if self.pelvis_normalization:
            landmarks, centroid, R2y, R2z = pelvis_normalization(landmarks,
                                                            self.rt_psis_ind,
                                                            self.lt_psis_ind,
                                                            self.rt_asis_ind,
                                                            self.lt_asis_ind,
                                                            self.nuchale_ind,
                                                            return_transformations=True)

            scan_vertices = scan_vertices - centroid
            scan_vertices = torch.matmul(scan_vertices, R2y.T) # K x 3
            scan_vertices = torch.matmul(scan_vertices, R2z.T)
            

        return_dict = {"name": f"{self.subject_id}_t{index}",
                        "vertices": scan_vertices * self.SCALE_TO_CM,
                        "faces": self.scan_faces,
                        "landmarks": landmarks * self.SCALE_TO_CM,
                        "measurements": measurements,
                        "gender": self.subject_gender
                        }
        
        
        return return_dict

    def __len__(self):
        return self.subject_sequence_data.shape[0]

class FourDHumanOutfit(Dataset):
    '''
    4DHumanOutfit dataset - loading only tight clothing type.
    '''
    def __init__(self, 
                 dataset_path: str,
                 parameters_path: str,
                 sequence_list: List[str] = None,
                 clothing_type: str = "tig",
                 pelvis_normalization: bool = False,
                 use_landmarks: str = "SMPL_INDEX_LANDAMRKS_REVISED",
                 transferred_landmarks_name: str = "simple",
                 body_models_path: str = "data/body_models",
                 **kwargs):
        
        """
        :param dataset_path: (str) path to 4DHumanOutfit scans
        :param parameters_path: (str) path to the directory where the already
                                 fitted SMPL parameters are stored
        :param sequence_list: (List[str]) list of sequences to load. 
                                If "All", all sequences are loaded
        :param clothing_type: (str) type of clothing to load
        :param pelvis_normalization: (bool) whether to normalize landmarks to pelvis
        :param use_landmarks: (str) landmarks defined on the SMPL model from utils.py
        :param transferred_landmarks_name: (str) name of the transferred landmarks
                                            provided by the dataset
                                            -> simple, nn or nn_threshold
        :param body_models_path: (str) path to the SMPL body models
        """
        all_male_subjects = ["ben","bob","jon","leo","mat","pat","ray","ted","tom"]
        all_female_subjects = ["ada","bea","deb","gia","joy","kim","mia","sue","zoe"]
        all_subjects_names = all_male_subjects + all_female_subjects
        
        # create gender mapper
        all_genders = ["male"] * len(all_male_subjects) + ["female"] * len(all_female_subjects)
        self.gender_mapper = dict(zip(all_subjects_names,all_genders))

        self.load_scans = True if not isinstance(dataset_path, type(None)) else False

        if not isinstance(sequence_list, type(None)):
            use_subjects = [seq.split("-")[0] for seq in sequence_list]
        else:
            use_subjects = all_subjects_names

        self.scan_paths = []
        self.subject_names = []
        self.action_names = []
        self.sequence_names = []
        self.poses = []
        self.shapes = []
        self.trans = []
        self.genders = []
        self.landmarks = []

        for subj_name in all_subjects_names:
            if subj_name in use_subjects:
                all_subj_action_paths = glob(os.path.join(parameters_path,subj_name,f"{subj_name}-{clothing_type}-*"))
                for subj_action_path in all_subj_action_paths:
                    action_name = os.path.basename(subj_action_path).split("-")[-1]
                    sequence_name = f"{subj_name}-{clothing_type}-{action_name}"

                    if not isinstance(sequence_list, type(None)):
                        if sequence_name not in sequence_list:
                            continue

                    all_seq_poses = torch.load(os.path.join(subj_action_path,"poses.pt"),
                                                      map_location=torch.device("cpu")).detach().cpu()
                    all_seq_shapes = torch.load(os.path.join(subj_action_path,"betas.pt"),
                                                    map_location=torch.device("cpu")).detach().cpu()
                    all_seq_trans = torch.load(os.path.join(subj_action_path,"trans.pt"),
                                                    map_location=torch.device("cpu")).detach().cpu()
                    all_seq_gender = self.gender_mapper[subj_name]

                    all_seq_landmarks_paths = torch.load(os.path.join(subj_action_path,f"landmarks_{transferred_landmarks_name}.pt"),
                                                           map_location=torch.device("cpu"))

                    N_frames = all_seq_poses.shape[0]

                    if self.load_scans:
                        all_seq_scan_paths = sorted(glob(os.path.join(dataset_path,subj_name,sequence_name,"*/*.obj")))
                    else:
                        all_seq_scan_paths = [f"{i}.obj" for i in range(N_frames)] # NOTE: just placeholders

                    for frame_ind in range(N_frames):
                        self.scan_paths.append(all_seq_scan_paths[frame_ind])
                        self.subject_names.append(subj_name)
                        self.action_names.append(action_name)
                        self.sequence_names.append(sequence_name)
                        self.poses.append(all_seq_poses[frame_ind])#.unsqueeze(0).detach().cpu())
                        self.shapes.append(all_seq_shapes[frame_ind])#.unsqueeze(0).detach().cpu())
                        self.trans.append(all_seq_trans[frame_ind])#.unsqueeze(0).detach().cpu())
                        self.genders.append(all_seq_gender)
                        self.landmarks.append(all_seq_landmarks_paths[frame_ind])#.unsqueeze(0).detach().cpu())
                

        self.dataset_size = len(self.poses)

        # pelvis normalization
        self.pelvis_normalization = pelvis_normalization
        landmarks_dict = getattr(utils, use_landmarks.upper(), None)
        self.landmark_names = list(landmarks_dict.keys())

        self.rt_psis_ind = self.landmark_names.index("Rt. PSIS")
        self.lt_psis_ind = self.landmark_names.index("Lt. PSIS")
        self.rt_asis_ind = self.landmark_names.index("Rt. ASIS")
        self.lt_asis_ind = self.landmark_names.index("Lt. ASIS")
        self.nuchale_ind = self.landmark_names.index("Nuchale")

        # create body models
        self.bms = {
            "MALE": create_body_model(body_models_path,
                                      "smpl",
                                      "MALE",
                                      8),
            "FEMALE": create_body_model(body_models_path,
                                      "smpl",
                                      "FEMALE",
                                      8),
            "NEUTRAL": create_body_model(body_models_path,
                                      "smpl",
                                      "NEUTRAL",
                                      8),
        }


    def __getitem__(self, index):
        """
        :return (dict): dictionary with keys:
        """

        # load scan
        scan_path = self.scan_paths[index]
        sequence_name = self.sequence_names[index] #scan_path.split("/")[-2]
        scan_name = os.path.basename(scan_path).split(".obj")[0]

        if self.load_scans:
            scan = o3d.io.read_triangle_mesh(scan_path)
            scan_vertices = torch.from_numpy(np.asarray(scan.vertices))
            scan_faces = torch.from_numpy(np.asarray(scan.triangles))
            scan_faces = scan_faces if scan_faces.shape[0] > 0 else None
        else:
            scan_vertices = None
            scan_faces = None
        scan_landmarks = self.landmarks[index]
        scan_gender = self.genders[index].upper()

        # create fitting
        fit_pose = self.poses[index].unsqueeze(0)
        fit_shape = self.shapes[index].unsqueeze(0)
        fit_trans = self.trans[index].unsqueeze(0)
        fit = self.bms[scan_gender](body_pose=fit_pose[:,3:],
                                    betas=fit_shape,
                                    global_orient=fit_pose[:,:3],
                                    transl=fit_trans,
                                    pose2rot=True).vertices[0].detach().cpu()
        # fit = fit + fit_trans

        # normalize
        if self.pelvis_normalization:
            scan_landmarks, centroid, R2y, R2z = pelvis_normalization(scan_landmarks,
                                                                    self.rt_psis_ind,
                                                                    self.lt_psis_ind,
                                                                    self.rt_asis_ind,
                                                                    self.lt_asis_ind,
                                                                    self.nuchale_ind,
                                                                    return_transformations=True)
            
            if self.load_scans:
                scan_vertices = scan_vertices - centroid
                scan_vertices = torch.matmul(scan_vertices.float(), R2y.T) # K x 3
                scan_vertices = torch.matmul(scan_vertices, R2z.T)

            fit = fit - centroid
            fit = torch.matmul(fit.float(), R2y.T) # K x 3
            fit = torch.matmul(fit, R2z.T)


        return {"name": f"{sequence_name}-{scan_name}",
                "sequence_name": sequence_name,
                "vertices": scan_vertices,
                "faces": scan_faces,
                "landmarks": scan_landmarks,
                "pose": self.poses[index],
                "shape": self.shapes[index],
                "trans": self.trans[index],
                "gender": self.genders[index],
                "fit":fit}


    def __len__(self):
        return self.dataset_size

class NPZDataset(Dataset):
    def __init__(self, 
                 dataset_path: str,
                 what_to_return: List[str] =["landmarks","measurements"],
                 **kwargs
                 ):
        """
        Dataset defined with a .npz file or folder with .npz files
        The .npz files are data points of subjects with landmarks and measurements

        :param dataset_path: (str) path to the .npz file or folder with .npz files
        :param what_to_return: (List[str]) list of keys to return from the .npz
        """
        
        if dataset_path.endswith(".npz"):
            self.data_orig = np.load(dataset_path)
            self.data = {name: (torch.from_numpy(self.data_orig[name])
                                                 if self.data_orig[name].dtype.type not in [np.str_] else self.data_orig[name]) 
                         for name in self.data_orig.keys()
                         }
        # if path is a flder with .npz files
        else:
            all_files = sorted(glob(os.path.join(dataset_path,"*.npz")))
            first_example = np.load(all_files[0])
            self.data = {name : [] for name in first_example.keys()}
            # self.possible_returns = list(first_example.keys())

            for fl in all_files:
                d = np.load(fl)
                for name in d.keys():
                    self.data[name].append(d[name])

            for name in self.data.keys():
                if np.array(self.data[name]).dtype.type not in [np.str_]:
                    if name not in ["indices_verts","indices_lm"]:
                        self.data[name] = torch.from_numpy(np.array(self.data[name]))
                    else:
                        self.data[name] = np.array(self.data[name])
                else:
                    self.data[name] = self.data[name]

        self.possible_returns = list(self.data.keys())

        self.what_to_return = what_to_return

        if "use_measurements" in kwargs:
            self.use_measurements = kwargs["use_measurements"]

    def __len__(self):
        if "landmarks" in self.data:
            return self.data["landmarks"].shape[0]
        elif "markers" in self.data:
            return self.data["markers"].shape[0]

    def __getitem__(self, index):
        
        return {name: self.data[name][index]
                for name in self.what_to_return
                if name in self.possible_returns}

class NPZDatasetTrainRobustness(Dataset):
    def __init__(self, 
                 dataset_path: str,
                 what_to_return: List[str] =["landmarks","measurements"],
                 pelvis_normalization: bool = True,
                 augmentation_landmark_2_origin_prob: float = 0,
                 use_landmarks: str = "SMPL_INDEX_LANDAMRKS_REVISED",
                 use_preprocessed_partial_landmarks: bool = False,
                 **kwargs
                 ):
        """
        Extension of the NPZ dataset that allows for augmentation of the landmarks

        :param dataset_path: (str) path to the .npz file or folder with .npz files
        :param what_to_return: (List[str]) list of keys to return from the .npz
        :param pelvis_normalization: (bool) whether to normalize landmarks to pelvis
        :param augmentation_landmark_2_origin_prob: (float) probability of moving a random landmark to the origin
        :param use_landmarks: (str) landmarks defined on the SMPL model from utils.py
        :param use_preprocessed_partial_landmarks: (bool) whether to use preprocessed partial landmarks instead of
                                                    randomly moving landmarks to the origin
        """

        if dataset_path.endswith(".npz"):
            self.data = np.load(dataset_path)
            self.data = {name: (torch.from_numpy(self.data[name])
                            if self.data[name].dtype.type not in [np.str_] else self.data[name]) 
                        for name in self.data.keys()}
        else:
            all_files = sorted(glob(os.path.join(dataset_path,"*.npz")))
            first_example = np.load(all_files[0])
            self.data = {name : [] for name in first_example.keys()}
            self.data["names"] = [] 

            for fl in all_files:
                self.data["names"].append(fl.split("/")[-1].split(".npz")[0])
                d = np.load(fl)
                for name in d.keys():
                    self.data[name].append(d[name])

            for name in self.data.keys():
                if np.array(self.data[name]).dtype.type not in [np.str_]:
                    if name not in ["indices_verts","indices_lm", "names"]:
                        self.data[name] = torch.from_numpy(np.array(self.data[name]))
                    else:
                        self.data[name] = np.array(self.data[name])
                else:
                    self.data[name] = self.data[name]

        self.possible_returns = list(self.data.keys())
        self.what_to_return = what_to_return

        # augmentation to partial landmarks
        self.augmentation_landmark_2_origin_prob = augmentation_landmark_2_origin_prob
        self.use_preprocessed_partial_landmarks = use_preprocessed_partial_landmarks

        self.save_augmentation = False
        if "save_path" in kwargs:
            self.save_augmentation = True
            self.augmentation_tracker_path = os.path.join(kwargs["save_path"], 
                                                        "augmentation_tracker.txt")

        # pelvis normalization
        self.pelvis_normalization = pelvis_normalization
        landmarks_dict = getattr(utils, use_landmarks.upper(), None)
        self.landmark_names = list(landmarks_dict.keys())

        self.rt_psis_ind = self.landmark_names.index("Rt. PSIS")
        self.lt_psis_ind = self.landmark_names.index("Lt. PSIS")
        self.rt_asis_ind = self.landmark_names.index("Rt. ASIS")
        self.lt_asis_ind = self.landmark_names.index("Lt. ASIS")
        self.nuchale_ind = self.landmark_names.index("Nuchale")

        if "use_measurements" in kwargs:
            self.use_measurements = kwargs["use_measurements"]


    def __len__(self):
        return self.data["landmarks"].shape[0]

    def __getitem__(self, index):

        current_data = {name: self.data[name][index]
                        for name in self.what_to_return
                        if name in self.possible_returns}
        lm = current_data["landmarks"].clone()

        # move random LM to origin
        if self.augmentation_landmark_2_origin_prob > 0:
            if np.random.choice([0,1],p=[0.5,0.5]):
                augment_lms = np.random.choice([0,1], 
                                            size=lm.shape[0],
                                            p=[1-self.augmentation_landmark_2_origin_prob,
                                                self.augmentation_landmark_2_origin_prob])
                # NOTE: HARDCODED WAY FOR NEVER REMOVING NORMALIZATION LANDMARKS
                augment_lms[[self.rt_psis_ind,
                            self.lt_psis_ind,
                            self.rt_asis_ind,
                            self.lt_asis_ind,
                            self.nuchale_ind]] = 0
                N_lm_to_origin = np.sum(augment_lms)
                # self.augmentation_tracker.append(N_lm_to_origin)
                if self.save_augmentation:
                    with open(self.augmentation_tracker_path, "a") as f:
                        f.write(f"{N_lm_to_origin}\n")
                augment_lms = torch.from_numpy(augment_lms) == 1
                lm[augment_lms] = torch.zeros((N_lm_to_origin,3),
                                                dtype=lm.dtype)
                
        if self.use_preprocessed_partial_landmarks:
            if np.random.choice([0,1],p=[0.5,0.5]):
                # NOTE: Hardcoded to always use pelvis landmarks
                current_data["indices_lm"][self.rt_psis_ind] = 1
                current_data["indices_lm"][self.lt_psis_ind] = 1
                current_data["indices_lm"][self.rt_asis_ind] = 1
                current_data["indices_lm"][self.lt_asis_ind] = 1
                current_data["indices_lm"][self.nuchale_ind] = 1
                partial_inds = current_data["indices_lm"]
                lm = lm[partial_inds]

        # normalize
        if self.pelvis_normalization:
            lm, centroid, R2y, R2z = pelvis_normalization(lm,
                                                            self.rt_psis_ind,
                                                            self.lt_psis_ind,
                                                            self.rt_asis_ind,
                                                            self.lt_asis_ind,
                                                            self.nuchale_ind,
                                                            return_transformations=True)


        current_data["landmarks"] = lm
        current_data["name"] = self.data["names"][index]
        
        return current_data
    
class CAESAR(Dataset):
    '''
    CAESAR dataset
    z-ax is the height
    returning vertices and landmarks in m and measurements in mm!
    '''
    def __init__(self, cfg: dict): 
        
        """
        cfg: config dictionary with
        :param data_dir (str): path to caesar dataset
        :param load_countries (str or list): countries to load. If "All", all countries are loaded
        :param use_landmarks (str or list): landmarks to use. If "All", all landmarks are loaded,
                                             if list of landmark names, only those are loaded
        :param load_measurements (bool): whether to load measurements or not
        """
        data_dir = cfg["data_dir"]
        load_countries = cfg["load_countries"]
        self.landmark_subset = cfg.get("use_landmarks",None)
        self.load_measurements = cfg.get("load_measurements",None)
        self.load_only_standing_pose = cfg.get("load_only_standing_pose",False)
        self.load_only_sitting_pose = cfg.get("load_only_sitting_pose",False)
        
        # set loading countries
        all_countries = ["Italy","The Netherlands","North America"]
        if load_countries == "All":
            load_countries = all_countries

        for country in load_countries:
            if country not in all_countries:
                msg = f"Country {country} not found. Available countries are: {all_countries}"
                raise ValueError(msg)

        # set paths
        scans_and_landmark_dir = os.path.join(data_dir, "Data AE2000")
        scans_and_landmark_paths = {
            "Italy": os.path.join(scans_and_landmark_dir, "Italy","PLY and LND Italy"),
            "The Netherlands": os.path.join(scans_and_landmark_dir, "The Netherlands","PLY and LND TN"),
            "North America": os.path.join(scans_and_landmark_dir, "North America","PLY and LND NA")
            }
        self.scans_and_landmark_paths = scans_and_landmark_paths

        if self.load_measurements:
            measurements_path = os.path.join(data_dir, 
                                            "processed_data", 
                                            "measurements.csv")
            self.measurements = pd.read_csv(measurements_path)
            measurements_extr_seat_path = os.path.join(data_dir, 
                                                    "processed_data", 
                                                    "measurements_extracted_seated.csv")
            self.meas_extr_seated = pd.read_csv(measurements_extr_seat_path)
            measurements_extr_stand_path = os.path.join(data_dir, 
                                            "processed_data", 
                                            "measurements_extracted_standing.csv")
            self.meas_extr_stand = pd.read_csv(measurements_extr_stand_path)
            demographics_path = os.path.join(data_dir, 
                                            "processed_data", 
                                            "demographics.csv")
            self.demographics = pd.read_csv(demographics_path)

        self.scan_paths = []
        self.landmark_paths = []
        self.countries = []

        for country, path in scans_and_landmark_paths.items():
            for scan_path in glob(f"{path}/*.ply.gz"):

                scan_pose = scan_path.split("/")[-1].split(".ply.gz")[0][-1]

                if self.load_only_standing_pose:
                    if scan_pose != "a":
                        continue

                if self.load_only_sitting_pose:
                    if scan_pose != "b":
                        continue

                # set scan path
                self.scan_paths.append(scan_path)

                # set landmark path
                landmark_path = scan_path.replace(".ply.gz", ".lnd")
                if os.path.exists(landmark_path):
                    self.landmark_paths.append(landmark_path)
                else:
                    self.landmark_paths.append(None)

                # set country 
                self.countries.append(country)


            
        self.dataset_size = len(self.scan_paths)
        self.LANDMARK_SCALE = 1000 # landmark coordinates are in mm, we want them in m
        self.country_scales = {"Italy": 1, "The Netherlands": 1000, "North America": 1} # scale to convert from mm to m
        

    def __getitem__(self, index):
        """
        :return (dict): dictionary with keys:
            "name": name of scan
            "vertices": (N,3) np.array
            "faces": (N,3) np.array or None if no faces
            "landmarks": dict with landmark names as keys and landmark coords as values
                         landmark coords are (1,3) np.array or None if no landmarks
            "country": string
        """

        # load country
        scan_country = self.countries[index]
        scan_scale = self.country_scales[scan_country]

        # load scan
        scan_path = self.scan_paths[index]
        scan_name = os.path.basename(scan_path).split(".ply.gz")[0]
        scan_number = int(scan_name[-5:-1])

        with gzip.open(scan_path, 'rb') as gz_file:
            try:
                ply_content = gz_file.read()
            except Exception as _:
                return {"name": scan_name,
                        "vertices": None,
                        "faces": None,
                        "landmarks": None,
                        "country": None,
                        }

            # TRIMESH APPROACH THAT WORKS
            # ply_fileobj = io.BytesIO(ply_content)
            # scan = trimesh.load_mesh(file_obj=ply_fileobj, file_type='ply')
            # scan_vertices = scan.vertices
            # scan_faces = scan.faces

            # OPEN3D APPROACH
            # complicated way to save ply to disk and load again -.-
            temp_ply_path = tempfile.mktemp(suffix=".ply")
            with open(temp_ply_path, 'wb') as temp_ply_file:
                temp_ply_file.write(ply_content)

            scan = o3d.io.read_triangle_mesh(temp_ply_path)
            # scan_center = scan.get_center()
            scan_vertices = np.asarray(scan.vertices) / scan_scale
            scan_faces = np.asarray(scan.triangles)
            scan_faces = scan_faces if scan_faces.shape[0] > 0 else None
            os.remove(temp_ply_path)

        # load landmarks
        landmark_path = self.landmark_paths[index]
        if landmark_path is not None:
            landmarks = process_caesar_landmarks(landmark_path, 
                                                 self.LANDMARK_SCALE)
                
            if isinstance(self.landmark_subset, list):
                landmarks = {lm_name: landmarks[lm_name] 
                             for lm_name in self.landmark_subset 
                             if lm_name in landmarks.keys()}
        else:
            landmarks = None

        # load measurements
        if self.load_measurements:
            measurements = self.measurements.loc[
                    (self.measurements["Country"] == scan_country) & 
                    (self.measurements["Subject Number"] == scan_number)
                    ].to_dict("records")
            measurements = None if measurements == [] else measurements[0]
            
            measurements_seat = self.meas_extr_seated.loc[
                    (self.meas_extr_seated["Country"] == scan_country) & 
                    (self.meas_extr_seated["Subject Number"] == scan_number)
                    ].to_dict("records")
            measurements_seat = None if measurements_seat == [] else measurements_seat[0]
            
            measurements_stand = self.meas_extr_stand.loc[
                    (self.meas_extr_stand["Country"] == scan_country) & 
                    (self.meas_extr_stand["Subject Number"] == scan_number)
                    ].to_dict("records")
            measurements_stand = None if measurements_stand == [] else measurements_stand[0]
            
            demographics = self.demographics.loc[
                    (self.demographics["Country"] == scan_country) & 
                    (self.demographics["Subject Number"] == scan_number)
                    ].to_dict("records")
            demographics = None if demographics == [] else demographics[0]
        else:
            measurements = None
            measurements_seat = None
            measurements_stand = None
            demographics = None
        

        return {"name": scan_name,
                "vertices": scan_vertices,
                "faces": scan_faces,
                "landmarks": landmarks,
                "country": self.countries[index],
                "measurements": measurements,
                "measurements_seat": measurements_seat,
                "measurements_stand": measurements_stand,
                "demographics": demographics
                }

    def __len__(self):
        return self.dataset_size

class OnTheFlyCAESAR(Dataset):
    '''
    reposed CAESAR dataset
    y-ax is the height
    vertices, landmarks, markers and measurements are returned in cm

    The idea is that for each pose from a SMPL pose dataset
    choose a random CAESAR subject and repose them into that pose
    with a certain probability.
    '''
    def __init__(self,
                 caesar_dir,
                 fitted_bm_dir=None,
                 fitted_nrd_dir=None,
                 poses_path=None,
                 n_poses=None,
                 dont_pose=False,
                 iterate_over_poses=True,
                 load_countries="All",
                 pose_params_from='all',
                 body_model_name="smpl",
                 body_models_path="data/body_models",
                 body_model_num_shape_param=10,
                 use_measurements=None,
                 use_subjects=None,
                 iterate_over_subjects=False,
                 use_landmarks="SMPL_INDEX_LANDMARKS",
                 landmark_normalization="pelvis",
                 what_to_return=["landmarks","measurements"],
                 augmentation_landmark_jitter_std=0,
                 augmentation_landmark_2_origin_prob=0,
                 augmentation_unpose_prob=0,
                 augmentation_repose_prob=0,
                 preprocessed_path=None,
                 subsample_verts=None,
                 use_moyo_poses=False,
                 moyo_poses_path=None,
                 single_fixed_pose_ind=None,
                 fix_dataset=None,
                 remove_monster_poses_threshold=None,
                 pose_prior_path="data/prior",
                 mocap_marker_path="/data/wear3d_preprocessed/mocap/simple+nn/standard/from_2023_10_18_23_32_22",
                 use_transferred_lm_path=None,
                 unposing_landmarks_choice="nn_to_verts", # "nn_to_smpl"
                 **kwargs
                 ): 
        
        """
        :param caesar_dir: (str) path to caesar dataset
        :param fitted_bm_dir: (str) path to SMPL param fits to CAESAR dataset
        :param fitted_nrd_dir: (str) path to SMPL vertex fits to CAESAR dataset
        :param poses_path: (str) SMPL poses dataset to use
        :param n_poses: (int) number of poses to use from the poses dataset
        :param dont_pose: (bool) whether to use the poses or not
        :param iterate_over_poses: (bool) whether to iterate over poses or subjects
                                -> if iterating over subjects the poses can be omitted or used
        :param load_countries: (str or list) countries to load from CAESAR. If "All", all countries are loaded
        :param pose_params_from: (str) which pose parameters to use from the SMPL poses dataset
        :param body_model_name: (str) name of the body model to use (smpl only)
        :param body_models_path: (str) path to the body models
        :param body_model_num_shape_param: (int) number of shape parameters of the body model
        :param use_measurements: (List[str]) measurements to load
        :param use_subjects: (str or List[str]) subjects to use from the CAESAR dataset
                            -> path to a .txt file with subjects to use
                               or list of subjects
        :param use_landmarks: (str) landmarks defined on the SMPL model from utils.py
        :param landmark_normalization: (str) normalization method for landmarks
        :param what_to_return: (List[str]) what to return from the dataset
        :param augmentation_landmark_jitter_std: (float) std of the landmark jitter
        :param augmentation_landmark_2_origin_prob: (float) probability of moving a random landmark to the origin
        :param augmentation_unpose_prob: (float) probability of unposing the subject
        :param augmentation_repose_prob: (float) probability of reposing the subject
        :param preprocessed_path: (str) path to preprocessed data used for unposing and reposing (not necessary)
        :param subsample_verts: (int) number of vertices to subsample to
        :param use_moyo_poses: (bool) whether to use MOYO poses
        :param moyo_poses_path: (str) path to MOYO poses
        :param single_fixed_pose_ind: (int) index if you want to use a single pose for the whole dataset
        :param fix_dataset: (str) fix a set of subjects to use from the start instead of randomly sampling in each iteration
        :param remove_monster_poses_threshold: (int) threshold for removing poses that are not plausible
        :param pose_prior_path: (str) path to the pose prior from SMPLify
        :param mocap_marker_path: (str) path to the mocap markers for the CAESAR dataset
        :param use_transferred_lm_path: (str) path to the transferred landmarks
        :param unposing_landmarks_choice: (str) choice of how to unpose the landmarks
                                            -> currently only "nn_to_verts"
        """

        self.caesar_dir = caesar_dir
        self.fitted_bm_dir = fitted_bm_dir
        self.fitted_nrd_dir = fitted_nrd_dir
        self.poses_path = poses_path
        self.n_poses = n_poses
        self.dont_pose = dont_pose
        self.load_countries = load_countries
        self.pose_params_from = pose_params_from
        self.body_model_name = body_model_name
        self.body_models_path = body_models_path
        self.body_model_num_shape_param = body_model_num_shape_param
        self.use_measurements = use_measurements
        self.use_subjects = use_subjects
        self.use_landmarks = use_landmarks
        self.landmark_normalization = landmark_normalization
        self.what_to_return = what_to_return
        self.augmentation_landmark_jitter_std = augmentation_landmark_jitter_std
        self.augmentation_landmark_2_origin_prob = augmentation_landmark_2_origin_prob
        self.augmentation_unpose_prob = augmentation_unpose_prob
        self.augmentation_dont_unpose_prob = 1 - augmentation_unpose_prob
        self.augmentation_repose_prob = augmentation_repose_prob
        self.augmentation_dont_repose_prob = 1 - augmentation_repose_prob
        self.preprocessed_path = preprocessed_path
        self.use_preprocessed_data = True if not isinstance(preprocessed_path, type(None)) else False
        # NOTE: if subsampling verts, the faces wont correspond anymore!
        self.subsample_verts = 1 if isinstance(subsample_verts, type(None)) else subsample_verts
        self.use_moyo_poses = use_moyo_poses
        self.moyo_poses_path = moyo_poses_path
        self.remove_monster_poses_threshold = remove_monster_poses_threshold
        self.pose_prior_path = pose_prior_path
        self.use_transferred_lm_path = use_transferred_lm_path
        self.unposing_lm_choice = unposing_landmarks_choice

        self.iterate_over_subjects = iterate_over_subjects
        self.single_fixed_pose_ind = single_fixed_pose_ind
        self.iterate_over_poses = iterate_over_poses
        self.fix_dataset = fix_dataset
        if self.iterate_over_subjects and self.iterate_over_poses:
            raise ValueError("Cannot iterate over subjects AND over poses \
                             - choose one bruv")


        # set CAESAR subjects to use
        if isinstance(self.use_subjects,str):
            with open(self.use_subjects, "r") as f:
                self.subjects_subset = f.read().splitlines()
        elif isinstance(self.use_subjects,list):
            self.subjects_subset = self.use_subjects
        else:
            self.subjects_subset = None
        
        # set CAESAR countries to load
        all_countries = ["Italy","The Netherlands","North America"]
        if load_countries == "All":
            load_countries = all_countries

        assert all([(country in all_countries) for country in load_countries]), \
            f"Available countries are: {all_countries}"

        # set paths
        scans_and_landmark_dir = os.path.join(caesar_dir, "Data AE2000")
        scans_and_landmark_paths = {
            "Italy": os.path.join(scans_and_landmark_dir, "Italy","PLY and LND Italy"),
            "The Netherlands": os.path.join(scans_and_landmark_dir, "The Netherlands","PLY and LND TN"),
            "North America": os.path.join(scans_and_landmark_dir, "North America","PLY and LND NA")
            }
        scans_and_landmark_paths = {country:path for country, path in scans_and_landmark_paths.items() if country in load_countries}
        self.scans_and_landmark_paths = scans_and_landmark_paths


        # load measurements
        if "measurements" in self.what_to_return:
            assert self.use_measurements is not None, "use_measurements must be set if measurements are returned"
            measurements_path = os.path.join(caesar_dir, 
                                            "processed_data", 
                                            "measurements.csv")
            self.measurements = pd.read_csv(measurements_path)
            measurements_extr_seat_path = os.path.join(caesar_dir, 
                                                    "processed_data", 
                                                    "measurements_extracted_seated.csv")
            self.meas_extr_seated = pd.read_csv(measurements_extr_seat_path)
            measurements_extr_stand_path = os.path.join(caesar_dir, 
                                            "processed_data", 
                                            "measurements_extracted_standing.csv")
            self.meas_extr_stand = pd.read_csv(measurements_extr_stand_path)
            demographics_path = os.path.join(caesar_dir, 
                                            "processed_data", 
                                            "demographics.csv")
            self.demographics = pd.read_csv(demographics_path)


        # gather paths for each subject
        self.scan_paths = []
        self.fitted_bm_paths = []
        self.fitted_nrd_paths = []
        self.landmark_paths = []
        self.countries = []
        self.scan_marker_paths = []

        for country, path in scans_and_landmark_paths.items():
            for scan_path in glob(f"{path}/*.ply.gz"):

                scan_name_with_extension = scan_path.split("/")[-1]
                # scan_pose = scan_name_with_extension.split(".ply.gz")[0][-1]
                scan_name = scan_name_with_extension.split(".ply.gz")[0]

                if not isinstance(self.subjects_subset,type(None)):
                    if scan_name not in self.subjects_subset:
                        continue

                # set fitted bm path
                if self.dont_pose:
                    pass
                else:
                    bm_path = os.path.join(self.fitted_bm_dir, f"{scan_name}.npz")
                    if not os.path.exists(bm_path):
                        continue
                    nrd_path = os.path.join(self.fitted_nrd_dir, f"{scan_name}.npz")
                    if not os.path.exists(nrd_path):
                        continue
                    self.fitted_bm_paths.append(bm_path)
                    self.fitted_nrd_paths.append(nrd_path)

                # set scan path
                self.scan_paths.append(scan_path)

                # set landmark path
                if isinstance(self.use_transferred_lm_path,type(None)):
                    landmark_path = scan_path.replace(".ply.gz", ".lnd")
                else:
                    landmark_path = os.path.join(use_transferred_lm_path, f"{scan_name}_landmarks.json")
                if os.path.exists(landmark_path):
                    self.landmark_paths.append(landmark_path)
                else:
                    self.landmark_paths.append(None)

                # set country 
                self.countries.append(country)

                # markers
                if not isinstance(mocap_marker_path, type(None)):
                    marker_path = os.path.join(mocap_marker_path, f"{scan_name}_markers.npy")
                    self.scan_marker_paths.append(marker_path)


            
        # set scaling
        self.N_scans = len(self.scan_paths)
        # scale everyhing to m - unpose and unscale works on m (divide by this scale)
        self.LANDMARK_SCALE_M = 1000 # landmark coordinates from mm to m
        self.country_scales_m = {"Italy": 1, "The Netherlands": 1000, "North America": 1} # scale to convert from mm to m
        # after unposing and scaling, scale everyhing to cm (multiply with this scale)
        self.LANDMARK_SCALE_CM = 100 # landmark coordinates from m to cm
        self.SCAN_SCALE_CM = 100 # scan vertices from m to cm
        self.MEASUREMENT_SCALE_CM = 10 # measurements from mm to cm
        self.MOCAP_MARKER_SCALE_CM = 100 # mocap markers from m to cm


        # get poses
        assert self.pose_params_from in ['all', 'h36m', 'up3d', '3dpw', 'amass', 'not_amass']

        all_poses = np.load(self.poses_path)
        if self.poses_path.endswith(".npz"):
            self.fnames = all_poses['fnames']
            self.poses = all_poses['poses']
        else:
            self.poses = all_poses
            self.fnames = None

        if self.pose_params_from != 'all':
            if self.pose_params_from == 'not_amass':
                indices = [i for i, x in enumerate(self.fnames)
                           if (x.startswith('h36m') or x.startswith('up3d') or x.startswith('3dpw'))]
                self.fnames = [self.fnames[i] for i in indices]
                self.poses = [self.poses[i] for i in indices]
            elif self.pose_params_from == 'amass':
                indices = [i for i, x in enumerate(self.fnames)
                           if not (x.startswith('h36m') or x.startswith('up3d') or x.startswith('3dpw'))]
                self.fnames = [self.fnames[i] for i in indices]
                self.poses = [self.poses[i] for i in indices]
            else:
                indices = [i for i, x in enumerate(self.fnames) if x.startswith(self.pose_params_from)]
                self.fnames = [self.fnames[i] for i in indices]
                self.poses = [self.poses[i] for i in indices]

        self.poses = np.stack(self.poses, axis=0)


        if self.use_moyo_poses:
            moyo_poses = get_moyo_poses(self.moyo_poses_path,
                                        sample_every_kth_pose=1,
                                        remove_hands_poses=True)
            
            self.poses = np.vstack([self.poses, moyo_poses])
        self.poses = torch.from_numpy(self.poses.astype(np.float32))

        # remove unplausible poses
        if not isinstance(self.remove_monster_poses_threshold, type(None)):
            pose_prior_loss_model = MaxMixturePrior(prior_folder=self.pose_prior_path,
                                                     num_gaussians=8)
            pose_prior_loss = pose_prior_loss_model.forward(self.poses[:, 3:].float(), None).numpy()
            good_poses = np.where(pose_prior_loss < self.remove_monster_poses_threshold)[0]
            self.poses = self.poses[good_poses,:]
            
        
        if not isinstance(self.n_poses, type(None)):
            # shuffle and take n_poses
            self.poses[torch.randperm(self.poses.size()[0])]
            self.poses = self.poses[:self.n_poses,:]
            print(f"subsampled poses to {self.poses.shape}")

        if self.dont_pose:
            self.poses = torch.zeros((1,72))
            self.single_fixed_pose_ind = 0
            self.iterate_over_subjects = True
            self.augmentation_unpose_prob = 0
            self.augmentation_dont_unpose_prob = 1
            self.augmentation_repose_prob = 0
            self.augmentation_dont_repose_prob = 1

        # fix subjects in each epoch instead of randomly sampling them
        if not isinstance(self.fix_dataset,type(None)):
            if isinstance(self.fix_dataset, list):
                if isinstance(self.fix_dataset[0],str):
                    # ovo su imena bez zadnjeg slova tako da mozes i sitting ljude imat
                    self.scan_paths_names = [scan_path.split("/")[-1].split(".ply.gz")[0][:-1] for scan_path in self.scan_paths]
                    self.fixed_subject_list = [self.scan_paths_names.index(name[:-1]) for name in self.fix_dataset]
                elif isinstance(self.fix_dataset[0],int):
                    self.fixed_subject_list = self.fix_dataset
            else:
                self.fixed_subject_list = np.random.choice(range(self.N_scans),
                                                        size=len(self.poses))
            self.fix_dataset = True

        # create body model
        self.body_models = {
            "MALE": create_body_model(self.body_models_path,
                                      self.body_model_name,
                                      "MALE",
                                      self.body_model_num_shape_param),
            "FEMALE": create_body_model(self.body_models_path,
                                      self.body_model_name,
                                      "FEMALE",
                                      self.body_model_num_shape_param),
            "NEUTRAL": create_body_model(self.body_models_path,
                                      self.body_model_name,
                                      "NEUTRAL",
                                      self.body_model_num_shape_param),
        }

        # set landmarks to use
        if isinstance(self.use_landmarks,str):
            landmarks_dict = getattr(utils, self.use_landmarks.upper(), None)
            self.landmark_names = list(landmarks_dict.keys())
            self.landmark_smpl_inds = list(landmarks_dict.values())

        assert all(isinstance(item, int) for item in self.landmark_smpl_inds), \
                "Landmark indices must be List[int]"

        # landmark normalization
        self.point_normalization = False
        self.pelvis_normalization = False

        if self.landmark_normalization:
            # point normalization
            if self.landmark_normalization in ["Substernale","Nose","BELLY_BUTTON"]:
                self.point_normalization = True
                self.landmark_normalizing_name, self.landmark_normalizing_ind = get_normalizing_landmark(self.landmark_names)
            # pelvis normalization
            elif self.landmark_normalization in ["pelvis"]:
                self.pelvis_normalization = True
                self.rt_psis_ind = self.landmark_names.index("Rt. PSIS")
                self.lt_psis_ind = self.landmark_names.index("Lt. PSIS")
                self.rt_asis_ind = self.landmark_names.index("Rt. ASIS")
                self.lt_asis_ind = self.landmark_names.index("Lt. ASIS")
                self.nuchale_ind = self.landmark_names.index("Nuchale")
            else:
                landmark_normalization_options = ["Substernale","Nose","BELLY_BUTTON","pelvis"]
                raise ValueError(f"{self.landmark_normalization} not in landmark \
                                 normalization options {landmark_normalization_options}")

        self.gender_encoder = {"MALE": 0,
                               "FEMALE": 1}

        # mocap markers
        if not isinstance(mocap_marker_path, type(None)):
            # path is like /data/wear3d_preprocessed/mocap/simple+nn/standard/from_2023_10_18_23_32_22
            mocap_marker_mapper = getattr(utils, "MOCAP_MARKER_MAPPER", None)
            mocap_marker_type = mocap_marker_mapper[mocap_marker_path.split("/")[-2]]

            self.mocap_markers = getattr(utils, mocap_marker_type.upper(), None)
            self.mocap_marker_order = list(self.mocap_markers.keys())
            self.mocap_marker_inds = [self.mocap_markers[m_name] for m_name in self.mocap_marker_order ]
        

    def __getitem__(self, index):

        return_dict={}

        # get pose
        if isinstance(self.single_fixed_pose_ind,type(None)):
            pose = self.poses[index].reshape(1,-1)
        else:
            pose = self.poses[self.single_fixed_pose_ind].reshape(1,-1)

        # get random caesar subject and load scan
        if self.iterate_over_subjects:
            scan_index = index
        elif self.fix_dataset:
            scan_index = self.fixed_subject_list[index]
        else:
            scan_index = np.random.choice(range(self.N_scans))
        scan_path = self.scan_paths[scan_index]
        scan_name = os.path.basename(scan_path).split("/")[-1].split(".ply.gz")[0]
        scan_number = int(scan_name[-5:-1])
        scan_country = self.countries[scan_index]
        if "vertices" in self.what_to_return:
            scan_verts, scan_faces = load_scan(scan_path)
            scan_scale = self.country_scales_m[scan_country]
            scan_verts = torch.from_numpy(scan_verts / scan_scale)
            scan_verts = scan_verts[::self.subsample_verts,:]
        else:
            scan_verts = None
            scan_faces = None


        # get measurements + gender
        if "measurements" in self.what_to_return:
            measurements = self.measurements.loc[
                    (self.measurements["Country"] == scan_country) & 
                    (self.measurements["Subject Number"] == scan_number)
                    ].to_dict("records")
            measurements = None if measurements == [] else measurements[0]
            
            # measurements_seat = self.meas_extr_seated.loc[
            #         (self.meas_extr_seated["Country"] == scan_country) & 
            #         (self.meas_extr_seated["Subject Number"] == scan_number)
            #         ].to_dict("records")
            # measurements_seat = None if measurements_seat == [] else measurements_seat[0]
            
            # measurements_stand = self.meas_extr_stand.loc[
            #         (self.meas_extr_stand["Country"] == scan_country) & 
            #         (self.meas_extr_stand["Subject Number"] == scan_number)
            #         ].to_dict("records")
            # measurements_stand = None if measurements_stand == [] else measurements_stand[0]
            
            demographics = self.demographics.loc[
                    (self.demographics["Country"] == scan_country) & 
                    (self.demographics["Subject Number"] == scan_number)
                    ].to_dict("records")
            demographics = None if demographics == [] else demographics[0]

            # process measurements
            measurements = torch.tensor([measurements[m_name] 
                                        for m_name in self.use_measurements])
            measurements = measurements / self.MEASUREMENT_SCALE_CM # CAESAR defined in mm, convert to cm
        else:
            measurements = None
            # measurements_seat = None
            # measurements_stand = None
            demographics = None

        scan_gender = "NEUTRAL"
        if demographics:
            scan_gender = demographics["Gender"].upper()

        # load landmarks
        landmark_path = self.landmark_paths[scan_index]
        if landmark_path is not None:
            landmarks = load_landmarks(landmark_path = landmark_path, 
                                        landmark_subset=None,
                                        scan_vertices=None,
                                        landmarks_scale=self.LANDMARK_SCALE_M,
                                        verbose=False)
            
            # landmarks = process_caesar_landmarks(landmark_path, 
            #                                      self.LANDMARK_SCALE_M)
                
            scan_lm = torch.cat([torch.from_numpy(landmarks[lm_name]).unsqueeze(0) 
                                for lm_name in self.landmark_names
                                if lm_name in landmarks.keys()],axis=0)
            scan_lm_names = [lm_name for lm_name in self.landmark_names if lm_name in landmarks.keys()]
            
        else:
            landmarks = None
            scan_lm = None


        if "markers" in self.what_to_return:
            marker_path = self.scan_marker_paths[scan_index]
            scan_markers = np.load(marker_path)
            scan_markers = torch.from_numpy(scan_markers)

            if isinstance(fit_verts, type(None)):
                fit_nrd_path = self.fitted_nrd_paths[scan_index]
                fit_nrd_data = np.load(fit_nrd_path)
                fit_verts = torch.from_numpy(fit_nrd_data["vertices"])
        else:
            scan_markers = None


        # get fitted body model
        if self.dont_pose:
            pass
        else:
            if self.use_preprocessed_data:
                preprocessed_data = np.load(os.path.join(self.preprocessed_path,
                                                        f"{scan_name}.npz"))

                # input(preprocessed_data.files)
                fit_verts = None
                fit_pose = None
                if "fit_shape" in preprocessed_data.files:
                    fit_shape = torch.from_numpy(preprocessed_data["fit_shape"])
                else:
                    fit_shape = None
                fit_trans = torch.from_numpy(preprocessed_data["fit_trans"])
                fit_scale = torch.from_numpy(preprocessed_data["fit_scale"])
                T = torch.from_numpy(preprocessed_data["T"])
                J = torch.from_numpy(preprocessed_data["J"])
                pose_offsets = torch.from_numpy(preprocessed_data["pose_offsets"])
                scan2fit_inds = torch.from_numpy(preprocessed_data["scan2fit_inds"])
                lm2fit_inds = torch.from_numpy(preprocessed_data["lm2fit_inds"])

                
            else:
                fit_bm_path = self.fitted_bm_paths[scan_index]
                fit_bm_data = np.load(fit_bm_path)
                fit_pose = torch.from_numpy(fit_bm_data["pose"])
                fit_shape = torch.from_numpy(fit_bm_data["shape"])
                fit_trans = torch.from_numpy(fit_bm_data["trans"])
                fit_scale = torch.from_numpy(fit_bm_data["scale"])

                fit_nrd_path = self.fitted_nrd_paths[scan_index]
                fit_nrd_data = np.load(fit_nrd_path)
                fit_verts = torch.from_numpy(fit_nrd_data["vertices"])
                
                T = None
                J = None
                pose_offsets = None
                scan2fit_inds = None
                lm2fit_inds = None

            return_dict["fit_shape"] = fit_shape

        
        # unpose
        unpose_scan_choice = np.random.choice([0,1], size=1,
                                              p=[self.augmentation_dont_unpose_prob,
                                                 self.augmentation_unpose_prob]).item()

        if unpose_scan_choice:
            unposed_data = unpose_caesar(scan_verts=scan_verts,
                                        scan_landmarks=scan_lm,
                                        scan_markers=scan_markers,
                                        fit_verts=fit_verts, 
                                        fit_shape=fit_shape, 
                                        fit_pose=fit_pose, 
                                        fit_scale=fit_scale, 
                                        fit_trans=fit_trans, 
                                        body_model=self.body_models[scan_gender], 
                                        batch_size=1,
                                        T=T, 
                                        J=J,
                                        pose_offsets=pose_offsets, 
                                        scan2fit_inds=scan2fit_inds, 
                                        lm2fit_inds=lm2fit_inds,
                                        marker2fit_inds=None,
                                        unposing_landmark_choice=self.unposing_lm_choice, #"nn_to_verts",
                                        smpl_landmark_inds=self.landmark_smpl_inds)
            scan_verts, scan2fit_inds, scan_lm, lm2fit_inds, scan_markers, marker2fit_inds = unposed_data

    
        # repose
        if unpose_scan_choice == 0:
            repose_scan_choice = 0
        else:
            repose_scan_choice = np.random.choice([0,1], size=1,
                                                  p=[self.augmentation_dont_repose_prob,
                                                     self.augmentation_repose_prob]).item()

        if repose_scan_choice:
            reposed_data = repose_caesar(unposed_scan_verts=scan_verts, 
                                         scan2fit_inds=scan2fit_inds, 
                                         new_pose=pose, 
                                         fitted_shape=fit_shape, 
                                         fitted_trans=fit_trans, 
                                         fitted_scale=fit_scale, 
                                         body_model=self.body_models[scan_gender], 
                                         unposed_scan_landmarks=scan_lm,
                                         lm2fit_inds=lm2fit_inds,
                                         batch_size=1,
                                         J=J,
                                         unposed_scan_markers=scan_markers, 
                                         marker2fit_inds=marker2fit_inds)
            scan_verts, scan_lm, scan_markers = reposed_data

        # landmarks + verts normalization
        # get landmarks + normalize + delete normalizing landmark (for point normalization) + scale
        if self.point_normalization:
            scan_lm = scan_lm - scan_lm[self.landmark_normalizing_ind,:] # K x 3
            scan_lm = torch.cat([scan_lm[:self.landmark_normalizing_ind],
                                 scan_lm[self.landmark_normalizing_ind+1:]]) # remove normalizing landmark
        elif self.pelvis_normalization:
            scan_lm, centroid, R2y, R2z = pelvis_normalization(scan_lm,
                                                            self.rt_psis_ind,
                                                            self.lt_psis_ind,
                                                            self.rt_asis_ind,
                                                            self.lt_asis_ind,
                                                            self.nuchale_ind,
                                                            return_transformations=True)

            if "vertices" in self.what_to_return:
                scan_verts = scan_verts - centroid
                scan_verts = torch.matmul(scan_verts.float(), R2y.T) # K x 3
                scan_verts = torch.matmul(scan_verts, R2z.T)

            if "markers" in self.what_to_return:
                scan_markers = scan_markers - centroid
                scan_markers = torch.matmul(scan_markers, R2y.T) # K x 3
                scan_markers = torch.matmul(scan_markers, R2z.T)

        
            
        # augment landmarks
        if self.augmentation_landmark_jitter_std: 
            scan_lm += torch.empty_like(scan_lm).normal_(0,self.augmentation_landmark_jitter_std)

        if self.augmentation_landmark_2_origin_prob:
            augment_lms = np.random.choice([0,1], 
                                           size=scan_lm.shape[0],
                                           p=[1-self.augmentation_landmark_2_origin_prob,
                                              self.augmentation_landmark_2_origin_prob])
            augment_lms = torch.from_numpy(augment_lms) == 1
            scan_lm[augment_lms] = torch.zeros((torch.sum(augment_lms).item(),3),
                                               dtype=scan_lm.dtype)



        # scale everyhing to cm
        scan_verts = scan_verts * self.SCAN_SCALE_CM if not isinstance(scan_verts,type(None)) else None
        scan_lm = scan_lm * self.LANDMARK_SCALE_CM
        scan_markers = scan_markers * self.MOCAP_MARKER_SCALE_CM if not isinstance(scan_markers,type(None)) else None

        # return what is needed
        return_dict.update({"name": scan_name,
                        "pose_param": pose,
                        "vertices": scan_verts,
                        "faces": scan_faces,
                        "landmarks": scan_lm,
                        "landmark_names": scan_lm_names,
                        "country": scan_country,
                        "measurements": measurements,
                        "unposed_scan_bool": unpose_scan_choice,
                        "reposed_scan_bool": repose_scan_choice,
                        "gender": scan_gender,
                        "gender_encoded": self.gender_encoder[scan_gender],
                        "markers": scan_markers
                        })
                        
        return {name: return_dict[name] for name in self.what_to_return}

    def __len__(self):
        if self.iterate_over_subjects:
            return len(self.scan_paths)
        return len(self.poses)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="subparsers", dest="subparser_name")

    parser_cluster_dataset = subparsers.add_parser('cluster_dataset')
    parser_cluster_dataset.add_argument('--poses_path', 
                                        type=str,
                                        default="data/poses/smpl_train_poses.npz",
                                        help='Path to smpl poses in npz format.', 
                                        )
    parser_cluster_dataset.add_argument('--on_what', 
                                        choices=["landmarks","pose_angles"], 
                                        default="landmarks",
                                        help='Cluster dataset on what.', 
                                        )
    parser_cluster_dataset.add_argument('--n_clusters', 
                                        type=int, 
                                        default=10000,
                                        help='Number of final clusters.', 
                                        )
    parser_cluster_dataset.add_argument('--random_state', 
                                        default=42,
                                        help='Randomness init for clustering.', 
                                        )
    parser_cluster_dataset.add_argument('--preprocessed_landmarks', 
                                        type=str,
                                        required=True,
                                        default=None,#"/SMPL-Fitting/aistats24/data/pose_similarity_clustering/dataset_landmarks_complete.npy",
                                        help='If clustering on LM, preprocessed LM \
                                            corresponding to poses.', 
                                        )
    parser_cluster_dataset.add_argument("--save_dataset_to",
                                        type=str,
                                        default="data/clustering_data",
                                        help="Save clustered dataset path.")
    parser_cluster_dataset.add_argument("--save_cluster_model_to",
                                        type=str,
                                        default="data/pose_similarity_clustering",
                                        help="Save clustering model and clusterindices path.")
    parser_cluster_dataset.add_argument("--cluster_pose_angles_func",
                                        type=str,
                                        default="np.sum",
                                        help="Sum or max of RRE for 23 joints to cluster poses.")

    args = parser.parse_args()
    
    if args.subparser_name == "cluster_dataset":

        from utils import SMPL_INDEX_LANDAMRKS_REVISED,create_body_model
        
        # load poses
        if not os.path.exists(args.poses_path):
            raise FileNotFoundError()
        
        data = np.load(args.poses_path)
        if args.poses_path.endswith(".npy"):
            poses = data
        else:
            poses = data["poses"]


        # ignore root joints rotation -- want to compare pose
        body_poses = poses[:,3:]
        N_poses = body_poses.shape[0]


        if args.on_what == "landmarks":

            name = f"SUM_LM_DIST"
            
            # get dataset landmarks
            landmarks_order = list(SMPL_INDEX_LANDAMRKS_REVISED.keys())
            landmark_inds = list(SMPL_INDEX_LANDAMRKS_REVISED.values())

            if isinstance(args.preprocessed_landmarks,type(None)) or \
                (not os.path.exists(args.preprocessed_landmarks)):

                root = "data/pose_similarity_clustering"
                args.preprocessed_landmarks = os.path.join(root,"dataset_landmarks.npy")
                
                response = input(f"""Creating landmarks from poses will be saved at
                                 {args.preprocessed_landmarks}. Continue? (y/n)""")
                
                if response.lower() in ["n","no"]:
                    sys.exit("Stopping...")

                dataset_landmarks = np.empty((body_poses.shape[0],len(landmark_inds),3))

                body_model_name="smpl"
                body_model_path="data/body_models"
                bm = create_body_model(body_model_path,
                                      body_model_name,
                                      "MALE",
                                      10)

                for i in tqdm(range(body_poses.shape[0])):


                    current_pose = body_poses[i]
                    bm_lm = bm(body_pose=torch.from_numpy(current_pose).unsqueeze(0).float(), 
                                                betas=torch.zeros((1,10)).float(), 
                                                global_orient=torch.zeros((1,3)).float(),
                                                pose2rot=True)["vertices"][0,landmark_inds,:].detach().cpu().numpy() # N_LM x 3
                    
                    dataset_landmarks[i,:,:] = bm_lm
                    
                np.save(args.preprocessed_landmarks,dataset_landmarks)
                print(f"Landmarks (obatained from posed smpls) saved in {args.preprocessed_landmarks}")

        
            # define inputs for clustering
            dataset_landmarks = np.load(args.preprocessed_landmarks)
            num_lm = dataset_landmarks.shape[1] #70
            single_cluster_shape = (1,num_lm,3)

            clustering_data = dataset_landmarks.reshape(dataset_landmarks.shape[0], -1)
            def custom_distance(x, y):
                lm_x = x.reshape((num_lm, 3))
                lm_y = y.reshape((num_lm, 3))
                dist_sum = np.sum(np.sqrt(np.sum((lm_x - lm_y)**2,axis=1)))
            #     dist_max = np.max(np.sqrt(np.sum((lm_x - lm_y)**2,axis=1)))
                return dist_sum

            def find_new_cluster_centers(clustering_data,clustering_model,single_cluster_shape,**kwargs):

                data_shape = deepcopy(single_cluster_shape)
                data_shape = [s for s in data_shape]
                data_shape[0] = -1
                data_shape = tuple(data_shape)
                clustering_data = clustering_data.reshape(data_shape)

                new_cluster_center_inds = []
                clustering_data_cuda = torch.from_numpy(clustering_data).cuda()

                for i in tqdm(range(args.n_clusters)):
                    cc = torch.from_numpy(clustering_model.cluster_centers_[i].reshape(single_cluster_shape)).cuda()
                    dists_from_cc = torch.sum(torch.sqrt(torch.sum(((clustering_data_cuda - cc)**2),dim=2)),dim=1)
                    new_cluster_center_inds.append(torch.argmin(dists_from_cc).detach().cpu().item())
                new_cluster_center_inds = np.array(new_cluster_center_inds)

                return new_cluster_center_inds

        elif args.on_what == "pose_angles":

            from scipy.spatial.transform import Rotation as sciRot

            def RRE(R_gt,R_estim):
                '''
                R_gt: numpy array dim (3,3)
                R_estim: np array dim (3,3)
                Returns: angle measurement in degrees
                '''

                # tnp = np.matmul(R_estim.T,R_gt)
                tnp = np.matmul(np.linalg.inv(R_estim),R_gt)
                tnp = (np.trace(tnp) -1) /2
                tnp = np.clip(tnp, -1, 1)
                tnp = np.arccos(tnp) * (180/np.pi)
                return tnp

            def RRE_batch(R_batch_estim,R_gt):
                '''
                R_batch_gt: numpy array dim (N,3,3)
                R_estim: numpy array dim (3,3)
                Returns: angle measurement in degrees
                '''

                tnp = np.matmul(np.transpose(R_batch_estim,(0,2,1)),R_gt)
                tnp = (np.trace(tnp,axis1=1,axis2=2) -1) /2
                tnp = np.clip(tnp, -1, 1)
                tnp = np.arccos(tnp) * (180/np.pi)

                return np.min(tnp), np.argmin(tnp)
            

            def RRE_batch_torch(R_batch_estim,R_gt):
                '''
                R_batch_gt: torch tensor dim (N,3,3)
                R_estim: torch tensor dim (3,3)
                Returns: angle measurement in degrees
                '''

                tnp = torch.matmul(R_batch_estim.transpose(1,2),R_gt)
                tnp = (torch.diagonal(tnp, dim1=1, dim2=2).sum(dim=1) - 1) / 2
                tnp = torch.clamp(tnp, -1, 1)
                tnp = torch.acos(tnp) * (180 / torch.pi)
                return torch.min(tnp).item(), torch.argmin(tnp).item()
            
            
            name = f"{args.cluster_pose_angles_func.split('.')[1].upper()}_RRE"
            
            # define inputs for clustering
            rotmat_body_poses = sciRot.from_rotvec(body_poses.reshape(-1,3), # (B*69,3)
                                  degrees=False)
            rotmat_body_poses = rotmat_body_poses.as_matrix().reshape(N_poses,-1,3,3)
            num_joints = rotmat_body_poses.shape[1]
            normalizer = num_joints * 180
            single_cluster_shape = (1,num_joints,3,3)
            clustering_data = rotmat_body_poses.reshape(rotmat_body_poses.shape[0], -1)

            args.cluster_pose_angles_func = eval(args.cluster_pose_angles_func)


            def custom_distance(x, y):
                pose_x = x.reshape((num_joints, 3, 3))
                pose_y = y.reshape((num_joints, 3, 3))
                sim_per_joint = [RRE(pose_x[i], pose_y[i]) for i in num_joints]

                sim = args.cluster_pose_angles_func(sim_per_joint) / normalizer
                return sim


            def find_new_cluster_centers(clustering_data,clustering_model,single_cluster_shape,**kwargs):

                data_shape = deepcopy(single_cluster_shape)
                data_shape = [s for s in data_shape]
                data_shape[0] = -1
                data_shape = tuple(data_shape)
                clustering_data = clustering_data.reshape(data_shape)

                new_cluster_center_inds = []
                clustering_data_cuda = torch.from_numpy(clustering_data).cuda()

                for i in tqdm(range(args.n_clusters)):
                    cc = torch.from_numpy(clustering_model.cluster_centers_[i].reshape(single_cluster_shape)).cuda()
                    rot_similarities_per_joint = {}

                    for i in range(num_joints):
                        rre_angle, _ = RRE_batch_torch(clustering_data_cuda[:,i,:,:],
                                              cc[0,i,:,:])
                        
                        rot_similarities_per_joint[i] = rre_angle#.detach().cpu().numpy()

                    similarities_for_cc = np.vstack([rot_similarities_per_joint[i] for i in range(num_joints)]).T
                    similarities_for_cc = args.cluster_pose_angles_func(similarities_for_cc,axis=1)
                    most_sim_example_to_cc = np.argmin(similarities_for_cc)

                    new_cluster_center_inds.append(most_sim_example_to_cc)
                new_cluster_center_inds = np.array(new_cluster_center_inds)

                return new_cluster_center_inds


        # cluster
        clustering_model = MiniBatchKMeans(n_clusters=args.n_clusters, 
                                           random_state=args.random_state,
                                           reassignment_ratio=0)
        clustering_model.pairwise_distances_ = custom_distance 
        clusters = clustering_model.fit_predict(clustering_data)

        clustering_model_name = f"MODEL_{N_poses}POSES_{name}_MINIBATCH_KMEANS_{args.n_clusters}_clusters"
        pickle.dump(clustering_model, 
                    open(os.path.join(args.save_cluster_model_to,clustering_model_name), "wb"))

        # because the actual clusters do not need to be from the data
        # redefine the cluster centers to be data points nearest to the found cluster center
        new_cluster_center_inds = find_new_cluster_centers(clustering_data,
                                                           clustering_model,
                                                           single_cluster_shape)
        clustering_model_centers_name = f"CLUSTER_CENTERS_INDICES_FROM_ORIG_DATA_{N_poses}POSES_{name}_MINIBATCH_KMEANS_{args.n_clusters}_clusters"
        np.save(os.path.join(args.save_cluster_model_to,clustering_model_centers_name),
                new_cluster_center_inds)