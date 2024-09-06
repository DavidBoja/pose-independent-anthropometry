
import sys
sys.path.append("..")

import os
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
import smplx
import argparse

from utils import create_body_model, SMPL_INDEX_LANDAMRKS_REVISED, CAESAR_Name2Gender
from losses import MaxMixturePrior

class OptimizationSMPL(torch.nn.Module):
    def __init__(self, init_pose=None, init_beta=None, init_trans=None):
        super(OptimizationSMPL, self).__init__()

        if not isinstance(init_pose,type(None)):
            self.pose = torch.nn.Parameter(init_pose.cuda())
        else:
            self.pose = torch.nn.Parameter(torch.zeros(1, 72).cuda())
        if not isinstance(init_beta,type(None)):
            self.beta = torch.nn.Parameter(init_beta.cuda())
        else:
            self.beta = torch.nn.Parameter((torch.zeros(1, 10).cuda()))
        if not isinstance(init_trans,type(None)):
            self.trans = torch.nn.Parameter(init_trans.cuda())
        else:
            self.trans = torch.nn.Parameter(torch.zeros(1, 3).cuda())

    def forward(self):
        return self.pose, self.beta, self.trans
    
class SMPLBodyModel():
    
    def __init__(self, gender, bm_path="../data/body_models/smpl"):
        
        body_model_path = os.path.join(bm_path, f"SMPL_{gender.upper()}.pkl")
        self.body_model = smplx.create(body_model_path, 
                                        model_type="SMPL",
                                        gender=gender.upper(), 
                                        use_face_contour=False,
                                        ext='pkl')
        
        self.current_pose = None
        self.current_global_orient = None
        self.current_shape = None
        self.current_trans = None
        self.body_model_name = "smpl"

    @property
    def N_verts(self):
        return 6890

    @property
    def verts_t_pose(self):
        return self.body_model.v_template

    @property
    def verts(self):
        return self.body_model(body_pose=self.current_pose, 
                               betas=self.current_shape, 
                               global_orient=self.current_global_orient).vertices[0]

    @property
    def joints(self):
        return self.body_model(body_pose=self.current_pose, 
                               betas=self.current_shape, 
                               global_orient=self.current_global_orient).joints[0]

    @property
    def faces(self):
        return self.body_model.faces
    
#     def landmark_indices(self,landmarks_order):
#         return [self.all_landmark_indices[k] for k in landmarks_order]


    def cuda(self):
        self.body_model.cuda()

    def __call__(self, pose, betas, **kwargs):

        self.current_pose = pose[:,3:]
        self.current_global_orient = pose[:,:3]
        self.current_shape = betas

        body_pose = pose[:,3:]
        global_orient = pose[:,:3]
        return self.body_model(body_pose=body_pose, betas=betas, global_orient=global_orient)
    
    def deform_verts(self, 
                     pose: torch.tensor,
                     betas: torch.tensor,
                     trans: torch.tensor):
        
        self.current_pose = pose[:,3:]
        self.current_global_orient = pose[:,:3]
        self.current_shape = betas
        self.current_trans = trans

        body_pose = pose[:,3:]
        global_orient = pose[:,:3]
        deformed_verts = self.body_model(body_pose=body_pose, 
                                         betas=betas, 
                                         global_orient=global_orient).vertices[0]

        return deformed_verts + trans
    
bms = {
            "MALE": create_body_model("../data/body_models",
                                      "smpl",
                                      "MALE",
                                      10),
            "FEMALE": create_body_model("../data/body_models",
                                      "smpl",
                                      "FEMALE",
                                      10),
            "NEUTRAL": create_body_model("../data/body_models",
                                      "smpl",
                                      "NEUTRAL",
                                      10),
        }


def fit_landmarks(subj_landmarks, init_trans=None, init_pose=None, init_beta=None):
    
    # get fitting optimizer params
    body_model_params = OptimizationSMPL(init_trans=init_trans,
                                         init_beta=init_beta,
                                         init_pose=init_pose).cuda()
    body_optimizer = torch.optim.Adam(body_model_params.parameters(), lr=INIT_LR)
    body_model = SMPLBodyModel(subj_gender)
    body_model.cuda()
    prior = MaxMixturePrior(prior_folder="../data/prior", num_gaussians=8) 
    prior = prior.cuda()

    # fit
    iterator = tqdm(range(FIT_ITERATIONS))
    for ind in iterator:

        if ind in WEIGHT_CONFIG:
            LM_WEIGHT = WEIGHT_CONFIG[ind]["landmark"]
            PRIOR_WEIGHT = WEIGHT_CONFIG[ind]["prior"]
            BETA_WEIGHT = WEIGHT_CONFIG[ind]["beta"]

        pose, beta, trans = body_model_params.forward()
        body_model_verts = body_model.deform_verts(pose,
                                                beta,
                                                trans)

        body_model_lm = body_model_verts[LANDMARKS_INDICES,:]

        # compute losses
        lm_loss = torch.sum(torch.norm((subj_landmarks - body_model_lm), dim=1))
        prior_loss = prior.forward(pose[:, 3:], beta)
        beta_loss = (beta**2).mean()
        loss = LM_WEIGHT * lm_loss + PRIOR_WEIGHT * prior_loss + BETA_WEIGHT * beta_loss
        iterator.set_description(f"Loss {loss.item():.4f}")

        # optimize
        body_optimizer.zero_grad()
        loss.backward()
        body_optimizer.step()

    # get final fit
    with torch.no_grad():
        pose, beta, trans = body_model_params.forward()
        body_model_verts = body_model.deform_verts(pose, beta, trans)
        fitted_body_model_verts = body_model_verts.detach().cpu().data.numpy()
        fitted_pose = pose.detach().cpu()#.numpy()
        fitted_shape = beta.detach().cpu().numpy()
        fitted_trans = trans.detach().cpu()#.numpy()

    return fitted_body_model_verts, fitted_pose, fitted_shape, fitted_trans

WEIGHT_CONFIG =  {0: {
                    "landmark": 5,
                    "prior": 0.001,
                    "beta": 0.01},

                # 70: {
                #     "landmark": 2.5,
                #     "prior": 0.1,
                #     "beta": 0.01}
                  

            }

LANDMARKS_ORDER = list(SMPL_INDEX_LANDAMRKS_REVISED.keys())
LANDMARKS_INDICES = list(SMPL_INDEX_LANDAMRKS_REVISED.values())

INIT_LR = 0.1
FIT_ITERATIONS = 100
BATCH_SIZE = 1
CM_TO_M = 0.01
M_TO_CM = 100

n2g = CAESAR_Name2Gender(gender_mapper_path="../data/gender/CAESAR_GENDER_MAPPER.npz")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", 
                        type=str, 
                        default="../data/processed_datasets/dataset_test_unposed.npz",
                        # default="../data/processed_datasets/dataset_test_unposed_noisy",
                        # default="../data/dataset_test_sit_transferred_lm_bm.npz",
                        # default="../data/dataset_test_posed.npz",
                        help="Path to get baseline results.")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    if "noisy" in dataset_path.lower():
        NOISY_DATASET = True


    if dataset_path.endswith(".npz"):

        data = np.load(dataset_path)
        all_landmarks = torch.from_numpy(data["landmarks"] * CM_TO_M).cuda() # N x 70 x 3
        N_examples = all_landmarks.shape[0]

        all_shapes = np.zeros((N_examples,10))

        for i in range(N_examples):
            subj_name = data["names"][i]
            print(f"Fitting {subj_name}")
            subj_gender = data["genders"][i].upper()
            subj_landmarks = all_landmarks[i,:,:] # 70 x 3

            if subj_landmarks.shape[0] != len(SMPL_INDEX_LANDAMRKS_REVISED):
                print(f"Subj {subj_name} does not have {len(SMPL_INDEX_LANDAMRKS_REVISED)} LM but {subj_landmarks.shape[0]}")

            # fit bm to 
            # init trans with substernale movement
            SUBSTERNALE_LM_INDEX = LANDMARKS_ORDER.index("Substernale")
            lm_bm_substeranle = bms[subj_gender].v_template[SUBSTERNALE_LM_INDEX].unsqueeze(0).cuda()
            lm_gt_substernale = subj_landmarks[SUBSTERNALE_LM_INDEX,:].unsqueeze(0)
            init_trans = (- lm_bm_substeranle + lm_gt_substernale).float()


            fitted_body_model_verts, fitted_pose, fitted_shape, fitted_trans = fit_landmarks(subj_landmarks,
                                                                                            init_trans=init_trans)            
            all_shapes[i,:] = fitted_shape

        np.savez(dataset_path,
                names=data["names"],
                landmarks=data["landmarks"],
                measurements=data["measurements"],
                genders=data["genders"],
                fit_shape=all_shapes)

        
    else:
        all_files = glob(os.path.join(dataset_path,"*.npz"))

        for pth in all_files:
            data = np.load(pth)
            
            if "name" in data:
                subj_name = data["name"]
            else:
                subj_name = pth.split("/")[-1].split(".npz")[0]
            print(f"Fitting {subj_name}")
            if NOISY_DATASET:
                subj_landmarks = torch.from_numpy(data["landmarks_noisy"] * CM_TO_M).cuda()
            else:
                subj_landmarks = torch.from_numpy(data["landmarks"] * CM_TO_M).cuda()
            subj_measurements = data["measurements"]
            if "gender" in data:
                subj_gender = data["gender"].item()
            else:
                subj_gender = n2g.get_gender(subj_name).upper()
            
            if subj_landmarks.shape[0] != len(SMPL_INDEX_LANDAMRKS_REVISED):
                print(f"Subj {subj_name} does not have {len(SMPL_INDEX_LANDAMRKS_REVISED)} LM but {subj_landmarks.shape[0]}")
            
            # fit bm to 
            # init trans with substernale movement
            SUBSTERNALE_LM_INDEX = LANDMARKS_ORDER.index("Substernale")
            lm_bm_substeranle = bms[subj_gender].v_template[SUBSTERNALE_LM_INDEX].unsqueeze(0).cuda()
            lm_gt_substernale = subj_landmarks[SUBSTERNALE_LM_INDEX,:].unsqueeze(0)
            init_trans = (- lm_bm_substeranle + lm_gt_substernale).float()

            fitted_body_model_verts, fitted_pose, fitted_shape, fitted_trans = fit_landmarks(subj_landmarks,
                                                                                            init_trans=init_trans)
            
            if NOISY_DATASET:
                np.savez(pth,
                    lm_displacement=data["lm_displacement"],
                    landmarks=data["landmarks"],
                    landmarks_noisy=data["landmarks_noisy"],
                    measurements=data["measurements"],
                    fit_shape=fitted_shape
                    )
            else:
                np.savez(pth,
                        name=data["name"] if "name" in data else subj_name,
                        landmarks=data["landmarks"],
                        measurements=data["measurements"],
                        gender=data["gender"] if "gender" in data else subj_gender,
                        fit_shape=fitted_shape)


