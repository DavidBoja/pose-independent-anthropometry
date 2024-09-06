import torch
import torch.optim as optim
import json
import os
from termcolor import colored
import time

import models
import meter
import visualization
from utils import (weights_init, create_results_directory, 
                   SMPL_INDEX_LANDMARKS, SMPL_INDEX_LANDAMRKS_REVISED,
                   StepLR, ConstantLR,
                   LM2Features)
import dataset
from tqdm import tqdm

class AbstractTrainer(object):
    def __init__(self, opt):
        super(AbstractTrainer, self).__init__()

        self.save_path_root = opt["paths"]["save_path_root"]

    def start_visdom(self):
        print("setting up visdom")
        if self.display:
            self.visualizer = visualization.Visualizer(self.port,self.env)

    def create_results_dir(self):
        """
        Get paths to save and reload networks
        :return:
        """

        self.save_path = create_results_directory(self.save_path_root,
                                                  self.continue_experiment)
        self.opt["paths"]["save_path"] = self.save_path
        self.dataset_config.update({"save_path": self.save_path})

        self.logname = os.path.join(self.save_path, "log.txt")
        self.checkpointname = os.path.join(self.save_path, 'optimizer_last.pth')

        config_name = os.path.join(self.save_path,"config.yaml")
        with open(config_name, 'w') as file:
            _ = json.dump(self.opt, file, default=lambda o: str(o))

    def init_meters(self):
        self.log = meter.Logs()

        if self.continue_experiment:
            self.log.continue_experiment(self.logname)

        self.lr_tracker = []

    def save_network(self):
        if self.save_model:
            print("saving net...")
            torch.save(self.network.state_dict(), f"{self.save_path}/network.pth")
            torch.save(self.optimizer.state_dict(), f"{self.save_path}/optimizer_last.pth")

            if self.log.curves["loss_train_total"][-1] < self.best_train_loss:
                self.best_train_loss = self.log.curves["loss_train_total"][-1]
                torch.save(self.network.state_dict(), f"{self.save_path}/network_best_train.pth")
                torch.save(self.optimizer.state_dict(), f"{self.save_path}/optimizer_best_train.pth")

            if self.log.curves["loss_val_total"][-1] < self.best_val_loss:
                self.best_val_loss = self.log.curves["loss_val_total"][-1]
                torch.save(self.network.state_dict(), f"{self.save_path}/network_best_val.pth")
                torch.save(self.optimizer.state_dict(), f"{self.save_path}/optimizer_best_val.pth")

    def learning_rate_scheduler(self):
        updated = self.lrate_scheduler.update(self.epoch)
        if updated:
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lrate_scheduler.lr)
        self.lr_tracker.append(self.lrate_scheduler.lr)

    def set_epoch(self):
        self.epoch = 0
        if self.continue_experiment:
            random_loss_name = self.log.curves_names[0]
            number_of_losses = len(self.log.curves[random_loss_name])
            self.epoch = number_of_losses

    def build_optimizer(self):
        """
        Create optimizer
        """
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.init_lr)

        if self.continue_experiment:
            self.optimizer.load_state_dict(torch.load(f'{self.checkpointname}'))

    def print_iteration_stats(self, loss, iteration):
        """
        print stats at each iteration
        """
        current_time = time.time()
        ellpased_time = current_time - self.start_train_time
        print(
            f"\r["
            + colored(f"{self.epoch}", "cyan")
            + f": "
            + colored(f"{iteration}", "red")
            + "/"
            + colored(f"{self.len_dataset}", "red")
            + "] train loss:  "
            + colored(f"{loss.item()} ", "yellow")
            + colored(f"Ellapsed Time: {ellpased_time/60/60}h ", "cyan"),
            end="",
        )

    def dump_stats(self):
        """
        Save stats at each epoch
        """

        log_table = {
            "epoch": self.epoch + 1, #FIXME: change to self.epoch
            "lr": self.lr_tracker,
            "losses": self.log.curves
        }

        with open(self.logname, "w") as f:
            f.write(json.dumps(log_table))

    def increment_epoch(self):
        self.epoch = self.epoch + 1

class TrainLm2MeasPosedReal(AbstractTrainer):
    def __init__(self, opt):
        super().__init__(opt)
        self.start_time = time.time()
        self.opt = opt
        self.process_opt(opt)
        self.start_visdom()
        self.create_results_dir()
        self.init_meters()
        self.set_epoch()


    def process_opt(self, opt):

        # general params
        continue_exp = self.opt["general"]["continue_experiment"]
        self.continue_experiment = continue_exp if not isinstance(continue_exp,type(None)) else None
        

        # learning params
        self.init_lr =  self.opt["learning"]["init_lr"]
        scheduler_name = self.opt["learning"]["lrate_update_func"]
        if isinstance(self.opt["learning_rate_schedulers"][scheduler_name],type(None)):
            self.opt["learning_rate_schedulers"][scheduler_name] = {}
        self.opt["learning_rate_schedulers"][scheduler_name]["init_lr"] = self.init_lr
        self.lrate_scheduler = eval(scheduler_name)(**self.opt["learning_rate_schedulers"][scheduler_name])
        self.workers = self.opt["learning"]["n_workers"]
        self.batch_size = self.opt["learning"]["batch_size"]
        self.model_name = self.opt["learning"]["model_name"]
        self.model_configs = self.opt["model_configs"][self.model_name]
        self.save_model = self.opt["learning"]["save_model"]
        self.weight_init_option = self.opt["learning"]["weight_init_option"]
        self.weight_init_params = self.opt["weight_init_options"][self.weight_init_option] if not isinstance(self.weight_init_option,type(None)) else None
        self.what_to_return = self.opt["learning"]["what_to_return"]
        if "gender" in self.what_to_return: self.what_to_return[self.what_to_return.index("gender")] = "gender_encoded"

        # assertions
        msg = "Output dim of model must match the number of measurements"
        assert self.model_configs["output_dim"] == len(self.opt["learning"]["measurements"]), msg

        # visualization params
        self.display = self.opt["visualization"]["display"]
        self.port = self.opt["visualization"]["port"]
        self.env = self.opt["visualization"]["env"]

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

            self.model_configs["encoder_input_dim"] = lm2feats_dim # model input is 


        # adjust model input dimension
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
        

        # dataset config
        self.dataset_name = self.opt["learning"]["dataset_name"]
        self.dataset_config = self.opt["dataset_configs"][self.dataset_name]
        self.dataset_config.update(self.opt["paths"])
        self.dataset_config["use_measurements"] = self.opt["learning"]["measurements"]
        self.dataset_config["use_landmarks"] = self.opt["learning"]["landmarks"]
        self.dataset_config["landmark_normalization"] = self.opt["learning"]["landmark_normalization"]
        self.dataset_config["what_to_return"] = self.opt["learning"]["what_to_return"]
        # self.dataset_config["use_transferred_lm_path"] = self.opt["learning"]["use_transferred_lm_path"]


        self.best_train_loss = 1000000
        self.best_val_loss = 1000000
             
    def build_network(self):

        try:
            network = getattr(models, self.model_name, None)(**self.model_configs)
        except Exception as e:
            print(e)
            print(f"Network {self.model_name} not found or config not defined properly.")

        if self.continue_experiment:
            try:
                network_path = os.path.join(self.save_path, "network.pth")
                network.load_state_dict(torch.load(network_path))
                print(" Previous network weights loaded! From ", network_path)
            except:
                print("Failed to reload ", network_path)
        else:
            network = weights_init(network, self.weight_init_option, self.weight_init_params)
        network.cuda()

        self.network = network

    def build_dataset_train(self):

        self.dataset_config_train = self.dataset_config.copy()
        self.dataset_config_train.update(self.dataset_config_train["train"])
        self.dataset_config_train.pop("train",None)
        self.dataset_config_train.pop("val",None)
        self.dataset_config_train.pop("test",None)

        self.dataset_train = getattr(dataset, self.dataset_name, None)(**self.dataset_config_train)
        
        self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train, 
                                                            batch_size=self.batch_size,
                                                            shuffle=True, 
                                                            num_workers=int(self.workers),
                                                            drop_last=False)
        self.len_dataset_train = len(self.dataloader_train)       

    def build_dataset_val(self):
        """
        Create validation dataset
        """
        self.dataset_config_val = self.dataset_config.copy()
        self.dataset_config_val.update(self.dataset_config_val["val"])
        self.dataset_config_val.pop("train",None)
        self.dataset_config_val.pop("val",None)
        self.dataset_config_val.pop("test",None)

        self.dataset_val = getattr(dataset, self.dataset_name, None)(**self.dataset_config_val)
                
        self.dataloader_val = torch.utils.data.DataLoader(self.dataset_val, 
                                                          batch_size=self.batch_size,
                                                          shuffle=False, 
                                                          num_workers=int(self.workers),
                                                          drop_last=False)
        self.len_dataset_val = len(self.dataloader_val)

    def build_losses(self):
        self.loss_func = torch.nn.MSELoss(reduction="mean")

    def train_epoch(self):

        self.log.reset()
        self.network.train()
        self.learning_rate_scheduler()
        start = time.time()
        self.len_dataset = self.len_dataset_train

        iterator = tqdm(self.dataloader_train)
        for batch in iterator:
            landmarks = batch['landmarks'] # (B, n_landm, 3)
            measurements_gt = batch['measurements'] # (B, n_measurements)
            current_batch_size = landmarks.shape[0]

            if self.transform_landmarks:
                landmark_features = [lm2feat(landmarks) for lm2feat in self.lm2feats] # each tensor (B, kfeatures)
                if len(landmark_features[0].shape) > 2:
                    batch['landmarks'] = torch.cat(landmark_features,dim=2).float() # (B, N_lm, sum of K_i)
                else:
                    batch['landmarks'] = torch.cat(landmark_features,dim=1).float() # (B, sum of K_i)

            inputs = tuple(batch[name].view(batch[name].shape[0],-1) 
                            # batch[name].view(self.batch_size,-1) 
                           for name in self.what_to_return 
                           if name not in self.not_input_data
                        #    if (name != "measurements")
                           )
            inputs = torch.cat(inputs,1)

            inputs, measurements_gt = inputs.float().cuda(), measurements_gt.cuda()

            self.optimizer.zero_grad()
            pred_measurements = self.network(inputs)
            with torch.no_grad():
                diff = torch.abs(pred_measurements-measurements_gt)
                for m_ind, m_name in enumerate(self.dataset_train.use_measurements):
                    self.log.update(f"loss_train_{m_name.replace(' ','_')}", 
                                    torch.sum(diff[:,m_ind]), 
                                    n=current_batch_size) # MAE FOR EACH MEASUREMENT

            loss = self.loss_func(pred_measurements, measurements_gt)
            loss.backward()
            self.log.update("loss_train_total", loss)
            self.optimizer.step()
            iterator.set_description(f"Loss {loss.item():.4f}")

        print("Ellapsed time : ", time.time() - start)

    def val_epoch(self):
        self.network.eval()
        self.len_dataset = self.len_dataset_val
        start = time.time()

        iterator = tqdm(self.dataloader_val)
        for batch in iterator:

            landmarks = batch['landmarks'] # (B, n_landm, 3)
            measurements_gt = batch['measurements'] # (B, n_measurements)
            current_batch_size = landmarks.shape[0]

            if self.transform_landmarks:
                landmark_features = [lm2feat(landmarks) for lm2feat in self.lm2feats] # each tensor (B, kfeatures)
                if len(landmark_features[0].shape) > 2:
                    batch['landmarks'] = torch.cat(landmark_features,dim=2).float() # (B, N_lm, sum of K_i)
                else:
                    batch['landmarks'] = torch.cat(landmark_features,dim=1).float() # (B, sum of K_i)

            inputs = tuple(batch[name].view(batch[name].shape[0],-1) 
                            #batch[name].view(self.batch_size,-1) 
                           for name in self.what_to_return 
                           if name not in self.not_input_data
                        #    if (name != "measurements")
                           )
            inputs = torch.cat(inputs,1)

            inputs, measurements_gt = inputs.float().cuda(), measurements_gt.cuda()

            pred_measurements = self.network(inputs)
            with torch.no_grad():
                diff = torch.abs(pred_measurements-measurements_gt)
                for m_ind, m_name in enumerate(self.dataset_val.use_measurements):
                    self.log.update(f"loss_val_{m_name.replace(' ','_')}", 
                                    torch.sum(diff[:,m_ind]), 
                                    n=current_batch_size) # MAE FOR EACH MEASUREMENT
            loss = self.loss_func(pred_measurements, measurements_gt)
            self.log.update("loss_val_total", loss)

            iterator.set_description(f"Loss {loss.item():.4f}")

        print("Ellapsed time : ", time.time() - start)

        self.log.end_epoch()
        if self.display:
            self.log.update_curves(self.visualizer.vis)

