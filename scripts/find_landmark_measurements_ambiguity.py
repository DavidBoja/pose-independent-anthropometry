

import sys
sys.path.append("..")

import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import argparse
import plotly.graph_objects as go

import importlib
anthropometry_module = importlib.import_module("SMPL-Anthropometry")

from utils import (create_body_model, 
                   SMPL_INDEX_LANDAMRKS_REVISED, 
                   get_simple_measurements, 
                   SMPL_SIMPLE_LANDMARKS, 
                   SMPL_SIMPLE_MEASUREMENTS_REVISED2)



simple_measurement_names = ["ankle left circumference",
                                    "arm length (shoulder to elbow)",
                                    "arm right length",
                                    "arm length (spine to wrist)",
                                    "chest circumference",
                                    "crotch height",
                                    "head circumference",
                                    "Hip circumference max height",
                                    "hip circumference",
                                    "neck circumference",
                                    "height"]


def get_subj(bmi_level, measurements_df):
    
    if bmi_level.lower() == "underweight":
        filtered_subj = measurements_df[measurements_df["BMI"] <= 18.4]
    elif bmi_level.lower() == "healthy":
        filtered_subj = measurements_df[(measurements_df["BMI"] > 18.4) & (measurements_df["BMI"] <= 24.9)]
    elif bmi_level.lower() == "overweight":
        filtered_subj = measurements_df[(measurements_df["BMI"] > 25.0) & (measurements_df["BMI"] <= 29.9)]
    elif bmi_level.lower() == "obese":
        filtered_subj = measurements_df[measurements_df["BMI"] > 30.0]
        
    subj = filtered_subj[["Subject Number","Country", "Gender"]].reset_index().iloc[0]

    return subj["Subject Number"], subj["Country"], subj["Gender"]


class Loss(nn.Module):

    def __init__(self, 
                 reference_shape, reference_lm, 
                 lm_inds,  
                 lm_dist_weight, displacement_unit_weight):
        
        super(Loss, self).__init__()
        
        self.reference_shape = reference_shape
        self.reference_lm = reference_lm
        
        self.bm = create_body_model(body_model_path="../data/body_models",
                                     body_model_type="smpl",
                                     gender="neutral",
                                     num_betas=10).cuda()
        self.lm_inds = lm_inds
        
        self.zero_pose = torch.zeros((1,69)).cuda()
        self.zero_global_orient = torch.zeros((1,3)).cuda()

        self.lm_dist_weight = lm_dist_weight
        self.displacement_unit_weight = displacement_unit_weight
        
        
    def forward(self, displacement_vector,**kwargs):
        
        displaced_shape = self.reference_shape + displacement_vector
        
        displaced_lm = self.bm(body_pose=self.zero_pose,
                                betas=displaced_shape,
                                global_orient=self.zero_global_orient,
                                pose2rot=True).vertices[0,self.lm_inds]
        
        # landmark distance loss
        sum_of_lm_distances = torch.sum(torch.norm((self.reference_lm - displaced_lm),
                                                   dim=1))
        
        # || displacement vector || = 1 loss
        distance_from_unit_norm = torch.abs((torch.norm(displacement_vector) - 1))        

        return self.lm_dist_weight * sum_of_lm_distances + self.displacement_unit_weight * distance_from_unit_norm
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--caesar_dir", 
                        type=str, 
                        default="/data/wear3d",
                        help="Subject name.")
    parser.add_argument("--fitted_bm_dir",
                        type=str,
                        default="/SMPL-Fitting/results/2023_09_27_12_21_18",
                        help="Path to fitted SMPL body model to the CAESAR dataset.")
    parser.add_argument("--preload_displacement_vector",
                        type=bool,
                        default=True,
                        help="Instead of optimizing for the displacement vector, preload it.")
    args = parser.parse_args()



    LANDMARK_NAMES = list(SMPL_INDEX_LANDAMRKS_REVISED.keys())
    LANDMARK_INDICES = list(SMPL_INDEX_LANDAMRKS_REVISED.values())


    measurements_path = os.path.join(args.caesar_dir,"processed_data/measurements.csv")
    measurements = pd.read_csv(measurements_path)
    # bmi is weight in kg / height in m **2
    measurements["BMI"] = (measurements["Weight (kg)"]) / ((measurements["Stature (mm)"] / 1000)**2  )



    ########### CREATE REFERENCE SHAPE, LANDMARKS, AND MEASUREMENTS ########################################

    # get random healthy subject to visualize the landmark-measurement ambiguity
    subj_number, subj_country, subj_gender = get_subj("healthy", measurements)

    subj_name = f"csr{subj_number}a.npz" if subj_country != "The Netherlands" else f"nl_{subj_number}a.npz"
    subj_path = os.path.join(args.fitted_bm_dir,subj_name)
    subject_fit = np.load(subj_path)
    reference_shape = subject_fit["shape"]
    reference_shape = torch.from_numpy(reference_shape)

    # create and measure reference body model
    reference_bm = create_body_model(body_model_path="../data/body_models",
                                    body_model_type="smpl",
                                    gender="neutral",
                                    num_betas=10)

    reference_bm_verts = reference_bm(body_pose=torch.zeros((1,69)),
                                        betas=reference_shape,
                                        global_orient=torch.zeros((1,3)),
                                        pose2rot=True).vertices[0]

    reference_lm = reference_bm_verts[LANDMARK_INDICES]

    # reference_measurements_simple = get_simple_measurements(reference_bm_verts.detach().cpu(), 
    #                                                         SMPL_SIMPLE_LANDMARKS,
    #                                                         SMPL_SIMPLE_MEASUREMENTS_REVISED2,
    #                                                         simple_measurement_names)
    # reference_measurements_simple = reference_measurements_simple * 0.01 # scale to meters

    reference_measurements_complex_measurer = anthropometry_module.MeasureBody("smpl")
    reference_measurements_complex_measurer.from_verts(verts=reference_bm_verts.detach().cpu()) 
    reference_measurements_complex_measurer.measure(simple_measurement_names)
    reference_measurements_complex = torch.tensor([reference_measurements_complex_measurer.measurements[m_name] 
                                                for m_name in simple_measurement_names])
    reference_measurements_complex = reference_measurements_complex * 0.01 # scale to meters


    ########### OPTIMIZE FOR SHAPE WITH MOST SIMILAR LANDMARK LOCATIONS ####################################

    if not args.preload_displacement_vector:
        NUM_ITER = 1000
        LM_DIST_WEIGHT = 1
        DISPLACEMENT_UNIT_WEIGHT = 1
        CONVERT_TO_CM = 100
        UPPER_BOUND_DISPLACEMENT = 11

        # random normalized displacement in shape space from reference shape
        displacement_vector = torch.randn((1,10), requires_grad=True)
        displacement_vector = displacement_vector / torch.norm(displacement_vector)
        displacement_vector = torch.nn.Parameter(displacement_vector.cuda())

        optimizer = optim.Adam([displacement_vector], lr=0.01)

        loss_func = Loss(reference_shape.cuda(), 
                        reference_lm.cuda(), 
                        LANDMARK_INDICES, 
                        LM_DIST_WEIGHT,
                        DISPLACEMENT_UNIT_WEIGHT)
        
        iterator = tqdm(range(NUM_ITER))
        for i in iterator:
            optimizer.zero_grad()
            
            loss = loss_func(displacement_vector)
            loss.backward(retain_graph=True)
            
            optimizer.step()
            iterator.set_description(f"Loss {loss.item():.4f}")

        final_displacement = displacement_vector.data.detach().cpu()
        print("Displacement vector:")
        print(final_displacement.shape)
        print("Norm of displacement vector:", torch.norm(final_displacement))

        avg_displacement = (loss.item() - abs(1-torch.norm(final_displacement)) / 70)
        print(f"On average, the LM are displaced for {avg_displacement} meters")

    else:
        final_displacement = torch.load("final_displacement.pt")


    ########### FOR EACH STEP ALONG THE DISPACEMENT VECTOR FIND DIFFERENCE IN LM AND MEAS ###############
    optim_bm = create_body_model(body_model_path="../data/body_models",
                                body_model_type="smpl",
                                gender="neutral",
                                num_betas=10)
    optim_bm_template = optim_bm.v_template.clone()
    optim_bm_faces = optim_bm.faces.copy()

    errors = {}
    errors["dist_from_reference_shape"] = []
    errors["median_dist_lm"] = []
    errors["max_dist_lm"] = []

    for m_name in simple_measurement_names:
        errors[f"ae_{m_name}"] = []

    for k in np.arange(0,UPPER_BOUND_DISPLACEMENT,0.2):
    
        optim_bm_verts = optim_bm(body_pose=torch.zeros((1,69)),
                                    betas=reference_shape + k * final_displacement,
                                    global_orient=torch.zeros((1,3)),
                                    pose2rot=True).vertices[0]
        
        pve = torch.norm(reference_bm_verts - optim_bm_verts,dim=1).detach().cpu()
        
        errors["dist_from_reference_shape"].append(k)
        errors["median_dist_lm"].append(torch.median(pve[LANDMARK_INDICES]).item())
        errors["max_dist_lm"].append(torch.max(pve[LANDMARK_INDICES]).item())
        

        optim_measurements_simple_complex_measurer = anthropometry_module.MeasureBody("smpl")
        optim_measurements_simple_complex_measurer.from_verts(verts=optim_bm_verts.detach().cpu()) 
        optim_measurements_simple_complex_measurer.measure(simple_measurement_names)
        optim_measurements_complex = torch.tensor([optim_measurements_simple_complex_measurer.measurements[m_name] 
                                                    for m_name in simple_measurement_names])
        optim_measurements_complex = optim_measurements_complex * 0.01 # scale to meters
        
        ae_measurements_complex = torch.abs(optim_measurements_complex-reference_measurements_complex)
        
        for m_ind, m_name in enumerate(simple_measurement_names):
            errors[f"ae_{m_name}"].append(ae_measurements_complex[m_ind].item())


    ########### CREATE FIGURE ###########################################################################
    fig = go.Figure()

    graph_map = {"arm right length": "Shoulder to wrist L.",
                    "chest circumference": "Chest C.",
                    "hip circumference": "Hip C.",
                    "height": "Stature",
                    "head circumference": "Head C."}
    
    for m_ind, m_name in enumerate(list(graph_map.keys())):
        fig.add_trace(go.Scatter(x=[x*CONVERT_TO_CM for x in errors["max_dist_lm"]], 
                                y=[x*CONVERT_TO_CM for x in errors[f"ae_{m_name}"]],
                                mode='lines',
                                name=graph_map[m_name]
                                )
                    )
        
        
    fig.update_layout(hovermode="x",
                    xaxis_title="Maximum landmark distance (cm)",
                    yaxis_title="Measurement absolute error (cm)",
                    yaxis=dict(tickvals=list(np.arange(0,7.5,0.5)),
                                tickformat=".2f",
                                tickfont = dict(size=10)),
                    xaxis_range=[0,2.5],
                    yaxis_range=[0,7],
                    width=900,
                    height=700,
                    legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            )
                    )

    fig.write_image("ambiguity_max_landmarks_wrt_measurements.pdf")
    fig.show()