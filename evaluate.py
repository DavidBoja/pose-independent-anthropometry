
import torch
import os
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import models
from utils import (load_config, 
                   SMPL_SIMPLE2CAESAR_MEASUREMENTS_MAP,
                   SMPL_INDEX_LANDAMRKS_REVISED,
                   pelvis_normalization,
                   SMPL_SIMPLE_LANDMARKS,
                   SMPL_SIMPLE_MEASUREMENTS,
                   LM2Features,
                   CAESAR_Name2Gender
                   )
import dataset
    

class Tester(object):
    """
    Evaluation is in cm
    """
    def __init__(self, opt):
        super(Tester, self).__init__()
        self.opt = opt
        self.process_opt(opt)
        self.build_loss_tracker()


    def process_opt(self, opt):

        # general params
        self.trained_network_path = os.path.join(opt["results_path"],"network_best_val.pth")

        # results params
        self.final_results = None
        self.save_results = opt["save_results"]
        self.visualize_results = opt["visualize_results"]

        self.show_stats = opt["show_stats"]
        if isinstance(self.show_stats, str):
            self.show_stats = [self.show_stats]

        # subset of subjects to evaluate on
        self.evaluate_on_subset = False
        subject_subset_path = getattr(self.opt["dataset"], "subject_subset_path", None)
        if not isinstance(subject_subset_path, type(None)):
            self.evaluate_on_subset = True
            with open(subject_subset_path,"r") as f:
                subject_subset = f.read()
            self.subject_subset = subject_subset.split("\n")

        self.results_unit = self.opt["results_unit"]


        # learning params
        self.batch_size = 1
        self.model_name = self.opt["learning"]["model_name"]
        self.model_configs = self.opt["model_configs"][self.model_name]

        # assertions
        msg = "Output dim of model must match the number of measurements"
        assert self.model_configs["output_dim"] == len(self.opt["learning"]["measurements"]), msg

        # landmarks and measurements
        self.landmarks_dict = eval(self.opt["learning"]["landmarks"])
        self.landmarks_order = list(self.landmarks_dict.keys())
        self.measurements_order = self.opt["learning"]["measurements"]

        self.evaluate_on_measurements = ["ankle left circumference",
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
        assert (self.measurements_order[:11] == self.evaluate_on_measurements) or \
                (self.measurements_order[:11] == [SMPL_SIMPLE2CAESAR_MEASUREMENTS_MAP[mm_name] 
                                                  for mm_name in self.evaluate_on_measurements])
            
        # landmark normalization
        self.pelvis_normalization = False

        self.landmark_normalization = self.opt["learning"].get("landmark_normalization")
        if self.landmark_normalization:
            landmark_normalization_options = ["pelvis"]
            assert self.landmark_normalization in landmark_normalization_options, f"Landmark normalization {self.landmark_normalization} not \
                                                                                  found in normalization options: {landmark_normalization_options}"
            
            if self.landmark_normalization in ["pelvis"]:
                self.pelvis_normalization = True
                print(f"Using pelvis normalization")
                self.rt_psis_ind = self.landmarks_order.index("Rt. PSIS")
                self.lt_psis_ind = self.landmarks_order.index("Lt. PSIS")
                self.rt_asis_ind = self.landmarks_order.index("Rt. ASIS")
                self.lt_asis_ind = self.landmarks_order.index("Lt. ASIS")
                self.nuchale_ind = self.landmarks_order.index("Nuchale")


        # landmark features
        self.transform_landmarks = False
        landmark_transformer_names = self.opt["learning"]["transform_landmarks"]
        if not (isinstance(landmark_transformer_names, type(None)) or 
                landmark_transformer_names == []):
            
            self.transform_landmarks = True
            if isinstance(landmark_transformer_names, type(str)):
                landmark_transformer_names = [landmark_transformer_names]
            
            self.lm2feats = []
            lm2feats_dim = 0
            for lm2feat_name in landmark_transformer_names:
                lm2feat_config = self.opt["feature_transformers"][lm2feat_name]
                lm2feat_config.update(opt["learning"])
                lm2feat_config["transform_landmarks"] = lm2feat_name
                lm2feat_class = LM2Features(**lm2feat_config)
                lm2feat_func = getattr(lm2feat_class,lm2feat_name)
                self.lm2feats.append(lm2feat_func)
                lm2feats_dim += lm2feat_class.out_dim

            self.model_configs["encoder_input_dim"] = lm2feats_dim # model input dim

        # gender
        self.name2gender = CAESAR_Name2Gender(self.opt["paths"]["caesar_gender_mapper"])

        # dataset return variables
        self.what_to_return = self.opt["learning"]["what_to_return"]
        for feature_name in self.what_to_return:
            # landmarks are processed in the transform_landmarks above
            if feature_name == "landmarks":
                continue
            elif feature_name == "gender_encoded":
                self.model_configs["encoder_input_dim"] = self.model_configs["encoder_input_dim"] + 1
            elif feature_name == "fit_shape":
                self.model_configs["encoder_input_dim"] = self.model_configs["encoder_input_dim"] + 10

        self.not_input_data = ["measurements", "name", "pose_param",
                               "unposed_scan_bool", "reposed_scan_bool"]

    def build_network(self):

        try:
            network = getattr(models, self.model_name, None)(**self.model_configs)
            network.load_state_dict(torch.load(self.trained_network_path))
            print(" Previous network weights loaded! From ", self.trained_network_path)
        except Exception as e:
            print(e)
            print(f"Network {self.model_name} not found or config not defined properly.")
        network.cuda()  # put network on GPU

        network = network.eval()
        self.network = network

    def evaluate(self):
        if self.opt["subaparser_name"] in ["VALIDATION_POSED",
                                           "CAESAR_STAND",
                                           "CAESAR_SIT",
                                           "CAESAR_SIT_TRANS_BM",
                                           "CAESAR_SIT_TRANS_NRD",
                                           "CAESAR_POSED",
                                           "FAUST_POSED",
                                           "DYNA_POSED"
                                           ]:
            self.evaluate_npz()
        elif self.opt["subaparser_name"] == "CAESAR_NOISY":
            self.evaluate_NoisyCaesar()
        elif self.opt["subaparser_name"] == "4DHumanOutfit":
            self.evaluate_4DHumanOutfit() 

    def build_dataset_test(self):
        """
        Create testing dataset
        """
        self.dataset_val = getattr(dataset, self.opt["dataset"]["dataset_name"])(**self.opt["dataset"])

    def evaluate_npz(self):

        self.network.eval()
        self.len_dataset = 0

        iterator = tqdm(self.dataset_val)
        for example_data in iterator:

            # get data
            landmarks = example_data["landmarks"].unsqueeze(0)
            measurements_gt = example_data["measurements"].unsqueeze(0) if not isinstance(example_data["measurements"],type(None)) else None
            if "names" in example_data:
                name = example_data.get("names", None)
            elif "name" in example_data:
                name = example_data.get("name", None)
            else:
                name = None
            if "gender" in example_data:
                gender = example_data.get("gender",None)
            elif "genders" in example_data:
                gender = example_data.get("genders",None)

            if self.transform_landmarks:
                landmark_features = [lm2feat(landmarks) for lm2feat in self.lm2feats] # each tensor (B, kfeatures)
                if len(landmark_features[0].shape) > 2:
                    example_data["landmarks"] = torch.cat(landmark_features,dim=2).float() # (B, N_lm, sum of K_i)
                else:
                    example_data["landmarks"] = torch.cat(landmark_features,dim=1).float() # (B, sum of K_i)

            inputs = tuple(example_data[name].view(example_data[name].shape[0],-1) 
                           if len(example_data[name].shape) > 1 else example_data[name].unsqueeze(0)
                           for name in self.what_to_return 
                           if name not in self.not_input_data
                           )
            inputs = torch.cat(inputs,1)

            inputs = inputs.cuda().float()
            measurements_gt = measurements_gt.cuda.float() if not isinstance(measurements_gt, type(None)) else None
            pred_measurements = self.network(inputs)
            self.track_loss(pred_measurements, measurements_gt, name, gender)

            self.len_dataset += 1

        self.print_stats()
        self.save_stats()

    def evaluate_NoisyCaesar(self):

        self.network.eval()
        self.len_dataset = 0

        # if normalize LM (doing this because NPZ dataset normalizes only landmarks and not landmarks_noisy)
        pelvis_normalize = self.opt["dataset"]["pelvis_normalization"]

        iterator = tqdm(self.dataset_val)
        for example_data in iterator:

            # keys are: ['lm_displacement', 'landmarks', 'landmarks_noisy', 'measurements']

            subj_name = example_data["name"]
            gender = self.name2gender.get_gender(subj_name)

            lm = example_data["landmarks_noisy"]#.unsqueeze(0)
            if pelvis_normalize:
                lm, centroid, R2y, R2z = pelvis_normalization(lm,
                                                            self.rt_psis_ind,
                                                            self.lt_psis_ind,
                                                            self.rt_asis_ind,
                                                            self.lt_asis_ind,
                                                            self.nuchale_ind,
                                                            return_transformations=True)
            

            # get data
            landmarks = lm.unsqueeze(0)
            measurements_gt = example_data["measurements"].unsqueeze(0)

            if self.transform_landmarks:
                landmark_features = [lm2feat(landmarks) for lm2feat in self.lm2feats] # each tensor (B, kfeatures)
                if len(landmark_features[0].shape) > 2:
                    example_data["landmarks"] = torch.cat(landmark_features,dim=2).float() # (B, N_lm, sum of K_i)
                else:
                    example_data["landmarks"] = torch.cat(landmark_features,dim=1).float() # (B, sum of K_i)

            inputs = tuple(example_data[name].view(example_data[name].shape[0],-1) 
                           for name in self.what_to_return 
                           if name not in self.not_input_data
                           )
            inputs = torch.cat(inputs,1)

            inputs, measurements_gt = inputs.cuda().float(), measurements_gt.cuda().float()
            pred_measurements = self.network(inputs)
            self.track_loss(pred_measurements, measurements_gt, gender=gender)

            self.len_dataset += 1

        self.print_stats()
        self.save_stats()

    def evaluate_4DHumanOutfit(self):

        self.network.eval()
        self.len_dataset = 0
        M_TO_CM = 100

        iterator = tqdm(self.dataset_val)
        for example_data in iterator:

            landmarks = example_data["landmarks"].unsqueeze(0) * M_TO_CM
            name = example_data.get("name", None)
            gender = example_data.get("gender",None)

            if self.transform_landmarks:
                landmark_features = [lm2feat(landmarks) for lm2feat in self.lm2feats] # each tensor (B, kfeatures)
                if len(landmark_features[0].shape) > 2:
                    example_data["landmarks"] = torch.cat(landmark_features,dim=2).float() # (B, N_lm, sum of K_i)
                else:
                    example_data["landmarks"] = torch.cat(landmark_features,dim=1).float() # (B, sum of K_i)

            inputs = tuple(example_data[name].view(example_data[name].shape[0],-1) 
                           if len(example_data[name].shape) > 1 else example_data[name].unsqueeze(0)
                           for name in self.what_to_return 
                           if name not in self.not_input_data
                           )
            inputs = torch.cat(inputs,1)

            inputs = inputs.cuda().float()
            pred_measurements = self.network(inputs)
            self.track_loss(pred_measurements, None, name, gender)

            self.len_dataset += 1

        self.print_stats()
        self.save_stats()

    def build_loss_tracker(self):
        self.tracked_loss = {}
        self.tracked_loss_per_example = {}
        self.tracked_names = []
        self.tracked_gender = []
        self.pred_meas_per_example = {}
        self.N_examples = 0

        for m_name in self.evaluate_on_measurements:
            self.tracked_loss[m_name] = torch.zeros(1)
            self.tracked_loss_per_example[m_name] = []
            self.pred_meas_per_example[m_name] = []

    def track_loss(self, pred_m, gt_m=None, name=None, gender=None):
        if isinstance(gt_m, type(None)):
            diff = pred_m.detach().cpu()
        else:
            diff = (pred_m - gt_m).detach().cpu()
        diff_abs = torch.abs(diff)
        self.N_examples += pred_m.shape[0]
        for i, m_name in enumerate(self.evaluate_on_measurements):
            self.tracked_loss[m_name] = self.tracked_loss[m_name] + diff_abs[0,i]
            self.tracked_loss_per_example[m_name].append(diff[0,i].item())
            self.pred_meas_per_example[m_name].append(pred_m[0,i].item())
        self.tracked_names.append(name)
        self.tracked_gender.append(gender)

    def print_stats(self):
        print(f"Number of examples evaluated:{self.N_examples}/{self.len_dataset}")

        if "aMAE" in self.show_stats:
            data_df = {}
            data_df["all"] = [(self.tracked_loss[m_name] / self.len_dataset).item() for m_name in self.evaluate_on_measurements]
            rows = [f"ALL ({self.results_unit})"]
            cols = self.evaluate_on_measurements

            if not isinstance(self.tracked_gender[0],type(None)):
                tracked_genders = np.array([x.lower() for x in self.tracked_gender])
                for gender in ["female","male"]:

                    if gender in tracked_genders:
                        data_df[gender] = []

                        data_filter = np.where(tracked_genders == gender)[0]
                        data_N = data_filter.shape[0]
                        for m_name in self.evaluate_on_measurements:
                            data_gendered = np.array(self.tracked_loss_per_example[m_name])[data_filter]
                            data_gendered_mae = np.sum(np.abs(data_gendered)) / data_N
                            data_df[gender].append(data_gendered_mae)

                        rows.insert(0,f"{gender.upper()} ({self.results_unit})")
    
            data_df = [data_df[name] for name in ["male","female","all"] if name in data_df]

            results = pd.DataFrame(data_df, 
                                    columns=cols, 
                                    index=rows).transpose()
            results.loc['Average'] = results.mean()
            pd.options.display.float_format = "{:,.4f}".format
            if self.results_unit == "mm":
                results = results * 10
            self.final_results = results
            print(results)

        
        # AE OVER TIME W.R.T. FIRST FRAME PRED
        if "STD_DYNA" in self.show_stats:
            devs = []

            for i, m_name in enumerate(self.evaluate_on_measurements):

                Y = np.array(self.pred_meas_per_example[m_name])
                Y = Y - Y[0]

                dev = np.std(Y)
                print(f"{m_name} - std {dev:.4f}cm")
                devs.append(dev)

            self.final_results = pd.DataFrame(devs,index=self.evaluate_on_measurements)

        
        if "STD_4DHumanOutfit" in self.show_stats:

            stats = np.zeros((len(self.evaluate_on_measurements),len(self.tracked_names)))

            all_full_names = self.tracked_names
            all_sequence_names = np.array([f"{name.split('-')[0]}-{name.split('-')[1]}-{name.split('-')[2]}" 
                                   for name in self.tracked_names])
            unique_sequence_names = np.unique(all_sequence_names)

            for m_ind, m_name in enumerate(self.evaluate_on_measurements):

                all_meas_m = np.array(self.pred_meas_per_example[m_name])
                for current_seq_name in unique_sequence_names:

                    seq_inds = np.where(all_sequence_names == current_seq_name)[0]

                    seq_meas = all_meas_m[seq_inds]
                    seq_meas = seq_meas - seq_meas[0]

                    stats[m_ind,seq_inds] = seq_meas

            df = pd.DataFrame(stats,
                              columns=all_full_names,
                              index=self.evaluate_on_measurements)
            self.final_results = df

            print("STD per measurement")
            print(df.std(axis=1))

    def save_stats(self):
        if self.save_results:
            if not isinstance(self.final_results,type(None)):
                self.final_results.to_csv(f"{self.opt['subaparser_name']}_results.csv")
            else:
                print("No results to save.")

    def visualize_stats(self):
        if not self.visualize_results:
            return
        
        if self.opt["subaparser_name"] == "DYNA_POSED":

            # AE OVER TIME W.R.T. FIRST FRAME PRED
            fig = go.Figure()

            N_frames = self.N_examples
            X = np.arange(N_frames)

            n_colors = len(self.evaluate_on_measurements)
            colorscale = px.colors.sample_colorscale("turbo", 
                                    [n/(n_colors -1) for n in range(n_colors)])

            for i, m_name in enumerate(self.evaluate_on_measurements):

                Y = np.array(self.pred_meas_per_example[m_name])
                Y = Y - Y[0]

                dev = np.std(Y)
                # print(f"{m_name} - std {dev:.4f}cm")

                fig.add_trace(go.Scatter(x=X, 
                                     y=Y,
                                    mode='lines',
                                    name=f"{m_name} - std {dev:.4f}cm",
                                    line=dict(color=colorscale[i])))

            fig.update_layout(title=f"DYNA - AE over time w.r.t. first frame prediction",
                              xaxis_title="time frames",
                              yaxis_title="AE")
            fig.show()
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="subparsers", dest="subparser_name")

    parser_valid_posed = subparsers.add_parser('VALIDATION_POSED')
    parser_valid_posed.add_argument("-R","--results_path", type=str, required=True)
    parser_valid_posed.add_argument("--dataset_name", type=str, default="NPZDataset")
    parser_valid_posed.add_argument("--dataset_path", type=str, 
                    default="data/processed_datasets/dataset_val.npz")
    parser_valid_posed.add_argument("--show_stats", default=["aMAE"],
                                help="Eval stats to show.")
    parser_valid_posed.add_argument("--save_results", action="store_true",
                        help="Save evaluation results.")


    parser_caesar_stand = subparsers.add_parser('CAESAR_STAND')
    parser_caesar_stand.add_argument("-R","--results_path", type=str, required=True)
    parser_caesar_stand.add_argument("--dataset_name", type=str, default="NPZDataset")
    parser_caesar_stand.add_argument("--dataset_path", type=str, 
                    default="data/processed_datasets/dataset_test_unposed.npz")
    parser_caesar_stand.add_argument("--show_stats", default=["aMAE"],
                                help="Eval stats to show.")
    parser_caesar_stand.add_argument("--save_results", action="store_true",
                        help="Save evaluation results.")
    

    parser_noisy_caesar = subparsers.add_parser('CAESAR_NOISY')
    parser_noisy_caesar.add_argument("-R","--results_path", type=str, required=True)
    parser_noisy_caesar.add_argument("--dataset_name", type=str, default="NPZDatasetTrainRobustness")
    parser_noisy_caesar.add_argument("--dataset_path", type=str, 
                    default="data/processed_datasets/dataset_test_unposed_noisy")
    parser_noisy_caesar.add_argument("--pelvis_normalization", action="store_true",
                        help="Use pelvis normalization for landmarks.")
    parser_noisy_caesar.add_argument("--use_landmarks", default="SMPL_INDEX_LANDAMRKS_REVISED",
                        help="Which landmarks are you using - necessary to do pelvis normalization.")
    parser_noisy_caesar.add_argument("--show_stats", default=["aMAE"],
                                help="Eval stats to show.")
    parser_noisy_caesar.add_argument("--save_results", action="store_true",
                        help="Save evaluation results.")


    # parser_caesar_sit = subparsers.add_parser('CAESAR_SIT')
    # parser_caesar_sit.add_argument("-R","--results_path", type=str, required=True)
    # parser_caesar_sit.add_argument("--dataset_name", type=str, default="NPZDataset")
    # parser_caesar_sit.add_argument("--dataset_path", type=str, 
    #                 default="/pose-independent-anthropometry/data/train_simple_models/data_test_sit.npz")
    # parser_caesar_sit.add_argument("--save_results", action="store_true",
    #                     help="Save evaluation results.")


    parser_caesar_sit_trans_bm = subparsers.add_parser('CAESAR_SIT_TRANS_BM')
    parser_caesar_sit_trans_bm.add_argument("-R","--results_path", type=str, required=True)
    parser_caesar_sit_trans_bm.add_argument("--dataset_name", type=str, default="NPZDataset")
    parser_caesar_sit_trans_bm.add_argument("--dataset_path", type=str, 
            default="data/dataset_test_sit_transferred_lm_bm.npz")
    parser_caesar_sit_trans_bm.add_argument("--show_stats", default=["aMAE"],
                                help="Eval stats to show.")
    parser_caesar_sit_trans_bm.add_argument("--save_results", action="store_true",
                        help="Save evaluation results.")
    

    # parser_caesar_sit_trans_nrd = subparsers.add_parser('CAESAR_SIT_TRANS_NRD')
    # parser_caesar_sit_trans_nrd.add_argument("-R","--results_path", type=str, required=True)
    # parser_caesar_sit_trans_nrd.add_argument("--dataset_name", type=str, default="NPZDataset")
    # parser_caesar_sit_trans_nrd.add_argument("--dataset_path", type=str, 
    #                 default="/pose-independent-anthropometry/data/train_simple_models/data_test_sit_transf_lm_nrd.npz")
    # parser_caesar_sit_trans_nrd.add_argument("--save_results", action="store_true",
    #                     help="Save evaluation results.")


    parser_caesar_posed = subparsers.add_parser('CAESAR_POSED')
    parser_caesar_posed.add_argument("-R","--results_path", type=str, required=True)
    parser_caesar_posed.add_argument("--dataset_name", type=str, default="NPZDataset")
    parser_caesar_posed.add_argument("--dataset_path", type=str, 
        default="data/dataset_test_posed.npz")
    parser_caesar_posed.add_argument("--show_stats", default=["aMAE"],
                                help="Eval stats to show.")
    parser_caesar_posed.add_argument("--save_results", action="store_true",
                        help="Save evaluation results.")
    

    parser_dyna = subparsers.add_parser('DYNA_POSED')
    parser_dyna.add_argument("-R","--results_path", type=str, required=True)
    parser_dyna.add_argument("--dataset_name", type=str, default="DYNA")
    parser_dyna.add_argument("--dataset_path", type=str, default="/data/DYNA")
    parser_dyna.add_argument("--subject_id", type=str, default="50009")
    parser_dyna.add_argument("--subject_action", type=str, default="jumping_jacks")
    parser_dyna.add_argument("--use_landmarks", default="SMPL_INDEX_LANDAMRKS_REVISED",
                        help="Which landmarks are you using - necessary to do pelvis normalization.")
    parser_dyna.add_argument("--show_stats", default=["STD_DYNA"],
                                help="Eval stats to show.")
    parser_dyna.add_argument("--measurement_type", type=str, default=None,
                             help="How is the DYNA SMPL measured.")
    parser_dyna.add_argument("--pelvis_normalization", action="store_true",
                        help="Use pelvis normalization for landmarks.")
    parser_dyna.add_argument("--save_results", action="store_true",
                        help="Save evaluation results.")
    parser_dyna.add_argument("--visualize_results", action="store_true",
                        help="Visualize evaluation results.")
    

    parser_4dhumanoutfit = subparsers.add_parser('4DHumanOutfit')
    parser_4dhumanoutfit.add_argument("-R","--results_path", type=str, required=True)
    parser_4dhumanoutfit.add_argument("--dataset_name", type=str, default="FourDHumanOutfit")
    parser_4dhumanoutfit.add_argument("--dataset_path", type=str, default=None) #"/data/FourDHumanOutfit/SCANS")
    parser_4dhumanoutfit.add_argument("--parameters_path", type=str, default="/FourDHumanOutfit-FITS",
                                help="Path to SMPL fitted params and landmarks to the 4DHumanOutfit dataset.")
    parser_4dhumanoutfit.add_argument("--use_landmarks", default="SMPL_INDEX_LANDAMRKS_REVISED",
                        help="Which landmarks are you using - necessary to do pelvis normalization.")
    parser_4dhumanoutfit.add_argument("--pelvis_normalization", action="store_true",
                        help="Use pelvis normalization for landmarks.")
    parser_4dhumanoutfit.add_argument("--transferred_landmarks_name", default="simple",
                                choices=["simple","nn"],
                                help="Method name to transfer landmarks.")
    parser_4dhumanoutfit.add_argument("--show_stats", default="STD_4DHumanOutfit",
                                help="Eval stats to show.")
    parser_4dhumanoutfit.add_argument("--sequence_list", 
                                default=["ben-tig-avoid","ben-tig-run","ben-tig-dance",
                                         "leo-tig-avoid","leo-tig-run","leo-tig-dance",
                                         "mat-tig-avoid","mat-tig-run","mat-tig-dance",
                                         "kim-tig-avoid","kim-tig-run","kim-tig-dance",
                                         "mia-tig-avoid","mia-tig-run","mia-tig-dance",
                                         "sue-tig-avoid","sue-tig-run","sue-tig-dance"],
                                help="Sequences to evaluate on.")
    parser_4dhumanoutfit.add_argument("--save_results", action="store_true",
                        help="Save evaluation results.")
    

    
    args = parser.parse_args()



    # process opt
    opt = load_config(os.path.join(args.results_path,"config.yaml"))
    opt["subaparser_name"] = args.subparser_name
    opt["results_path"] = args.results_path
    opt["results_unit"] = "mm"
    opt["show_stats"] = getattr(args, "show_stats", None)
    opt["save_results"] = getattr(args, "save_results", None)
    opt["visualize_results"] = getattr(args, "visualize_results", None)

    opt["dataset"] = {
        "dataset_name": getattr(args, "dataset_name", None),
        "load_only_standing_pose": getattr(args, "only_standing_pose", None),
        "load_only_sitting_pose": getattr(args, "only_sitting_pose", None),
        "subject_subset_path": getattr(args, "subject_subset_path", None),

        "dataset_path": getattr(args, "dataset_path", None),
        "what_to_return": ["landmarks","measurements", # for all datasets
                           "lm_displacement","landmarks_noisy", # for noisy caesar
                           "gender", "genders", "name", "names",
                           "fit_shape"
                           ],
        "use_landmarks": getattr(args, "use_landmarks", None),
        "pelvis_normalization": getattr(args, "pelvis_normalization", None),
        "load_gt": getattr(args, "load_gt", None),
        "measurement_type": getattr(args, "measurement_type", None),

        # DYNA
        "subject_id": getattr(args, "subject_id", None),
        "subject_action": getattr(args, "subject_action", None),

        # 4DHumanOutfit
        "transferred_landmarks_name": getattr(args, "transferred_landmarks_name", None),
        "parameters_path": getattr(args, "parameters_path", None),
        "sequence_list": getattr(args, "sequence_list", None),
        "body_models_path": opt["paths"]["body_models_path"],
    }

    tester = Tester(opt)
    tester.build_network()
    tester.build_dataset_test()
    tester.evaluate()
    tester.visualize_stats()


