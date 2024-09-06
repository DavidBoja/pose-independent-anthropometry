
# Pose-independent 3D anthropometry

This Github presents the code for the following paper: ["Pose-independent 3D Anthropometry from Sparse Data"](https://inria.hal.science/hal-04683475/file/eccv_2024_pose_independent_anthropometry_camera_ready.pdf) presented at ECCV 2024 workshop ["T-CAP 2024 Towards a Complete Analysis of People"](https://sites.google.com/view/t-cap-2024/home).

<p align="center">
  <img src="https://github.com/DavidBoja/pose-independent-anthropometry/blob/master/assets/eccv_teaser.png" width="950">
</p>

<b> TL;DR :</b> Estimate 11 body measurements from 70 body landmarks of a posed subject. 

<br>
<br>




## 🔨 Getting started

You can use 🐳 docker to facilitate running the code. After cloning the repo, run in terminal:

```bash
cd docker
sh build.sh
sh docker_run.sh CODE_PATH DATA_PATH
```

by adjusting the `CODE_PATH` to the cloned `pose-independent-anthropometry` directory and `DATA_PATH` to the data directory you want to access in the docker container. This creates a `pose-independent-anthropometry-container` which you can attach to by running:

```bash
docker exec -it pose-independent-anthropometry-container /bin/bash
```

🚧 If you do not want to use docker, you can install the `docker/requirements.txt` into  your own environment. 🚧

<br>

Download:
- the SMPL body model from [here](https://github.com/vchoutas/smplx#downloading-the-model) and put the `SMPL_{SEX}.pkl` (MALE, FEMALE and NEUTRAL) models into the `data/body_models/smpl` folder
- the `smpl_train_poses.npz` and `smpl_val_poses.npz` from [here](https://drive.google.com/drive/folders/1lvxwKcqi4HaxTLQlEicPhN5Q3L-aWjYN) and put them in the folder `data/poses`
- the `gmm_08.pkl` file from [here](https://smplify.is.tue.mpg.de/) and put it in the `data/prior` folder.

Finally, initialize the smpl-anthropometry submodule by running:
```bash
git submodule update --init --recursive
```

<br>
<br>






## 💻 Datasets

The datasets used in the paper are either based on [CAESAR](https://www.sae.org/standardsdev/tsb/cooperative/caesar.htm), [DYNA](http://dyna.is.tue.mpg.de/) or [4DHumanOutfit](https://kinovis.inria.fr/4dhumanoutfit/):

- The CAESAR dataset is available commercially [here](https://bodysizeshape.com/page-1855750)
- The DYNA dataset is available freely [here](http://dyna.is.tue.mpg.de/)
- The 4DHumanOutfit dataset is available freely upon request [here](https://kinovis.inria.fr/4dhumanoutfit/) or by contacting the authors [David Bojanić](https://www.fer.unizg.hr/david.bojanic?) or [Stefanie Wuhrer](https://swuhrer.gitlabpages.inria.fr/website/)

Once you obtain all of the datasets, we provide scripts to create all of the dataset versions used in the paper.

### CAESAR dataset preprocessing

The dataset structure assumed is the following: 
```
{path/to/CAESAR}/Data AE2000/{country}/PLY and LND {country}/
``` 
which contains scans in `.ply.gz` format and landmarks in `.lnd` format, and `country` can be any of the following: Italy, North America or The Netherlands.

<br>
You also need the SMPL fittings to the scans (both the parameter fittings and the vertex fittings) in format 

```
{path/to/fitting}/{subject_name}.npz
```

To create the fittings in this format you use the [SMPL-Fitting](https://github.com/DavidBoja/SMPL-Fitting) repository and run:

```bash
python fit_body_model.py onto_dataset --dataset_name CAESAR

python fit_vertices.py onto_dataset --dataset_name CAESAR --start_from_previous_results <path-to-previously-fitted-bm-results>
```

where `<path-to-previously-fitted-bm-results>` is the path to the fitted SMPL body model from the first line of code.


Finally, adjust the following paths in `configs/config_real.yaml`:

- `caesar_dir`: path to the CAESAR dataset
- `fitted_bm_dir`: path to the fitted parameters to the CAESAR dataset obtained from the SMPL-Fitting repository
- `fitted_nrd_dir`: path to the fitted vertices to the CAESAR dataset obtained from the SMPL-Fitting repository

<br>

### Training data
<!-- /data/wear3d_preprocessed/data_train_posed_normalized_tsoli_without_bad -->

To create the training dataset, first complete the steps from [CAESAR preprocessing](#caesar-dataset-preprocessing). Then, you need to create the poses used for training by running:
```bash
python dataset.py cluster_dataset 
cd scripts
python create_training_poses.py --fitted_bm_dir <path/to/fitted/SMPL/to/CAESAR> 
```
where `fitted_bm_dir` is the path to the fitted SMPL  body model to the CAESAR scans (see [CAESAR preprocessing](#caesar-dataset-preprocessing))

Finally, you can create the training data using:

```bash
cd scripts

python create_CAESAR_POSED_train_dataset.py --save_to <path/to/save/the/dataset/to>
```

where `save_to` is the path where you want to save the created dataset to.

<br>

### Validation data
<!-- /pose-independent-anthropometry/data/train_simple_models/data_val_posed_tsoli_without_bad.npz -->


To create the validation dataset, first complete the steps from [CAESAR preprocessing](#caesar-dataset-preprocessing). Then, you can run:

```bash
cd scripts

python create_CAESAR_POSED_val_dataset.py --save_to <path/to/save/the/dataset/to>
```

where `save_to` is the path where you want to save the created dataset to.

<br>


### Testing data


#### 🧍🏽 CAESAR A-pose (Tables 1 & 2 left part)
<!-- /pose-independent-anthropometry/data/train_simple_models/data_test_unposed_tsoli_without_bad.npz -->

To create the dataset, first complete the steps from [CAESAR preprocessing](#caesar-dataset-preprocessing). Then, you can run:

```bash
cd scripts

python create_CAESAR_APOSE_test_dataset.py --save_to <path/to/save/the/dataset/to>
```

where `save_to` is the path where you want to save the created dataset to.

<br>

#### 🧍🏽 CAESAR A-pose with noisy landmarks (Tables 1 & 2 right part)
<!-- /data/wear3d_preprocessed/NoisyCaesarAPOSE_5mm -->
To create the dataset, first complete the steps from [CAESAR preprocessing](#caesar-dataset-preprocessing).
Since the noise is added randomly, we provide the vector displacements from the original landmarks in `data/processed_datasets/dataset_test_unposed_noisy_displacements`. To obtain the noisy landmarks you need to add the displacements to the original landmarks provided in the CAESAR dataset as:
 ```python
 ceasar_noisy_landmarks = caesar_landmarks + displacement_vector
 ```


If you, however, want to create your own noisy dataset, you can run:
```bash
cd scripts

python create_CAESAR_NOISY_test_dataset.py --save_to <path/to/save/the/dataset/to>
```

where `save_to` is the path where you want to save the created dataset to.

<br>

#### 🪑 CAESAR sitting B-pose (Table 3)
<!-- /pose-independent-anthropometry/data/train_simple_models/data_test_sit_transf_lm_bm_tsoli_without_bad.npz -->
Because the sitting B-pose in CAESAR does not have all of the landmarks necessary to run our method, we transfer the missing landmarks using the fitted SMPL body model.
To transfer the landmarks, run:
```bash
cd annotate

python annotate_CAESAR_landmarks.py --caesar_path <path/to/CAESAR> --fitting_path <path/to/fitted/SMPL/to/scans> --save_to <path/to/save/the/landmarks/to> 

cd ..
```

Then, you can create the dataset with:
```bash
cd scripts

python create_CAESAR_SITTING_test_dataset.py --save_to <path/to/save/the/dataset/to> --transferred_landmark_path <path/to/transferred/landmarks>
```

where `save_to` is the path where you want to save the created dataset to and `transferred_landmark_path` is the path where you saved the transferred landmarks from the code above.

<br>

#### 💃 CAESAR arbitrary pose (Table 4)
<!-- /pose-independent-anthropometry/data/train_simple_models/data_test_posed_tsoli_without_bad.npz -->
To create the dataset, first complete the steps from [CAESAR preprocessing](#caesar-dataset-preprocessing). Then, you can run:

```bash
cd scripts

python create_CAESAR_POSED_test_dataset.py --save_to <path/to/save/the/dataset/to>
```

where `save_to` is the path where you want to save the created dataset to.

<br>


#### 👯‍♀️ DYNA dynamic sequence (Table 5)
Download the dataset from [here](http://dyna.is.tue.mpg.de/) (you will need to sign up). 
You only need the `dyna_male.h5` and `dyna_female.h5` files.

<br>

#### 🕺 4DHumanOutfit clothed sequences (Table 6)
<!-- /data/FourDHumanOutfit/SCANS and /FourDHumanOutfit-FITS -->

The dataset structure assumed is the following: 

```
{path/to/4DHumanOutfit}/{subject_name}/{subject_name}-{clothing_type}-{action}/*/model-*.obj
```

After you get the dataset, you can use:
```bash
cd scripts
bash unzip_4DHumanOutfit_scans.sh <path/to/4DHumanOutfit> <unzip/destination/path>
```

to unzip the dataset, where `<unzip/destination/path>` is the folder where you want to unzip it.

The following subjects are used:
```
ben
leo
mat
kim
mia
sue
```

in tight clothing, performing the following actions:
```
dance
run
avoid
```

The scans provided by 4DHumanOutfit are the ones with resolution `OBJ_4DCVT_15k`.

The dataset also comes with the fitted SMPL parameters (upon request) using the approach from [3], in the same format as the provided scans: 
```
{path/to/fittings}/{subject_name}/{subject_name}-{clothing_type}-{action}/{parameter}.pt
```

where `{parameter}` is any of the follwing: `betas.pt`,`poses.pt` and `trans.pt`.


Finally, you can obtain the landmarks by running:
```bash
cd annotate

python annotate_4DHumanOutfit_landmarks.py --scan_paths <path/to/4DHumanOutfit> --fit_paths <path/to/fittings> --transfer_method simple
```

where

- `scan_paths` is the path to the downloaded and unzipped dataset
- `fit_paths` is the path to the SMPL parameters
- `transfer_method` is the way of obtaning the landmarks and can be one of the following:

  - `simple` where the transferred landmarks correspond to the actual SMPL landmarks
  - `nn` where the transferred landmarks correspond to the nearest neighbor of each fitted SMPL landmark and the scan
  - `nn_threshold` where the transferred landmark is the nearest neighbor of the fitted SMPL landmark to the scan if the distance is below the `nn_threshold` (defined below), else it is the actual SMPL landmark

- `nn_threshold` is the nearest neighbor threshold in meters for the nn_threshold transfer method

The landmarks will be saved in the same folder as the parameters `fit_paths` and the `transfer_method` used in the paper is the `simple` one.

<br>






## 🏋️ Training

To train our model you can run:
```bash
python train.py
```

To visualize the loss curves during training run in a seperate terminal:
```bash
visdom -p <port>
```
and navigate in your browser to `http://localhost:<port>/`. If you do not see any curves, make sure you choose the `lm2meas` environment in the dropdown menu and make sure the port you choose is the same one as in `configs/config_real.yaml` under `visualization/port`.

<br>

The training parameters are set in `configs/config_real.yaml`. To train the same model as in our paper, you can leave all the parameters as they are except potentially fixing the paths defined throught the configuration file. We briefly explain all the parameters for easier reference:

**general parameters:**

- `continue_experiment`: (str) name of the experiment in format "%Y_%m_%d_%H_%M_%S" from the results folder you wish to continue running
- `experiment_name`: (str) name your experiment for easier reference

**visualization parameters:**

- `display`: (bool) visualize loss curves with visdom or not
- `port`: (int) the loss curves will be visualized on `http://localhost:<port>/`
- `env`: (str) visdom environment of the experiment

**learning parameters:**

- `model_name`: (str) name of the model to train, defined in `models.py`
- `dataset_name`: (str) name of the dataset to use defined in `dataset.py` - actual data used is defined in `dataset_configs` below
- `save_model`: (bool) save trained model or not
- `what_to_return`: (list) list of variables the dataset will return to the training script (depends on the dataset used)
- `transform_landmarks`: (list) of feature transformers defined in `feature_transformers` below - they transform the input data for the model
- `landmark_normalization`: (str) of how to normalize the landmarks (only `pelvis` is acceptable)
- `batch_size`: (int) model batch size
- `n_workers`: (int) number of pytorch workers to load the batches
- `nepoch`: (int) number of training epochs to run
- `init_lr`: (float) initial value of the learning rate
- `lrate_update_func`: (str) learning rate scheduler defined in `learning_rate_schedulers` below
- `measurements`: (list) of measurements to train the model on
- `landmarks`: (dict) of landmarks defined on the SMPL body model (dict keys are used)
- `seed`: (int) training random seed
- `weight_init_option`: (str) initialization of the model weights defined in `weight_init_options` below

**paths parameters:**

- `save_path_root`: (str) where to save the training results
- `caesar_dir`: (str) path to the CAESAR dataset
- `fitted_bm_dir`: (str) path to the SMPL fitted body models (see [CAESAR preprocessing](#caesar-dataset-preprocessing))
- `fitted_nrd_dir`: (str) path to the SMPL fitted vertices (see [CAESAR preprocessing](#caesar-dataset-preprocessing))
- `body_models_path`: (str) path to the SMPL body models (see [Getting started](#🔨getting-started))
- `pose_prior_path`: (str) path to the pose prior from [2]
- `preprocessed_path`: (str) path to the preprocessed files used to unpose and repose the CAESAR scans in the OnTheFlyCAESAR dataset
- `moyo_poses_path`: (str) path to the MOYO dataset
- `caesar_gender_mapper`: (str) path to the CAESAR gender mapper file which maps a subject name to their provided sex

**model_configs parameters:**

- `SimpleMLP`: setup the layer dimensions for the SimpleMLP model

**dataset_configs parameters:**

- `NPZDataset`: setup the train and validation paths for the datasets you created in [💻 Datasets](#💻-datasets)

**feature_transformers parameters:**

- `coords`: returns the input landmarks and ravels into a single dimension vector if required
- `distances_all`: returns the distances between all the input landmarks
- `distances_grouped`: returns the distances between the desired landmarks defined in the path `grouping_inds_path`

**learning_rate_schedulers parameters:**

- `ConstantLR`: keeps a constant learning rate defined in `init_lr`

**weight_init_options parameters:**

- `output_layer_bias_to_mean_measurement`: initialize the output layer of the SimpleMLP to the mean training set measurements

<br>

Note that many of the parameters are not necessary to successfully train the model.

<br>
<br>

## 💯 Evaluation

You can use `evaluate.py` to reproduce the results from the paper. We provide our trained model in `results/2024_07_11_09_42_48`.

The `dataset_path` input variable to the `evaluate.py` script should correspond to the paths of the datasets you created in [💻 Datasets](#💻-datasets). If you used our default paths, then you can omit it in the following calls.

#### 🧍🏽 CAESAR A-pose (Tables 1 & 2 left part)
<!-- /pose-independent-anthropometry/data/train_simple_models/data_test_unposed_tsoli_without_bad.npz -->

```bash
python evaluate.py CAESAR_STAND -R results/2024_07_11_09_42_48 --dataset_path <path/to/dataset>
```

<br>

#### 🧍🏽 CAESAR A-pose with noisy landmarks (Tables 1 & 2 right part)
<!-- /data/wear3d_preprocessed/NoisyCaesarAPOSE_5mm -->
```bash
python evaluate.py CAESAR_NOISY -R results/2024_07_11_09_42_48 --pelvis_normalization --dataset_path <path/to/dataset>
```

<br>

#### 🪑 CAESAR sitting B-pose (Table 3)
<!-- /pose-independent-anthropometry/data/train_simple_models/data_test_sit_transf_lm_bm_tsoli_without_bad.npz -->
```bash
python evaluate.py CAESAR_SIT_TRANS_BM -R results/2024_07_11_09_42_48 --dataset_path <path/to/dataset>
```

<br>

#### 💃 CAESAR arbitrary pose (Table 4)
<!-- /pose-independent-anthropometry/data/train_simple_models/data_test_posed_tsoli_without_bad.npz -->
```bash
python evaluate.py CAESAR_POSED -R results/2024_07_11_09_42_48 --dataset_path <path/to/dataset>
```

<br>

#### 👯‍♀️ DYNA dynamic sequence (Table 5)
```bash
python evaluate.py DYNA_POSED -R results/2024_07_11_09_42_48 --dataset_path <path/to/dataset> --pelvis_normalization
```

<br>

#### 🕺 4DHumanOutfit clothed sequences (Table 6)
<!-- /data/FourDHumanOutfit/SCANS and /FourDHumanOutfit-FITS -->
```bash
python evaluate.py 4DHumanOutfit -R results/2024_07_11_09_42_48 --pelvis_normalization --parameters_path <path/to/params>
```
where `parameters_path` is the path to the SMPL parameters fitted to the scans along with the obtained landmarks from [💻 Datasets](#💻-datasets).


<br>
<br>



## 0️⃣ Baseline models

In order to evaluate the baseline models described in the paper on a given dataset, first you need to fit the SMPL body model onto the provided landmarks by running:
```bash
cd scripts
python add_shape_to_dataset.py --dataset_path <path/to/dataset>
```

after which you can evaluate it with:
```bash
python evaluate_baseline.py --dataset_path <path/to/evaluation/dataset>
```

<br>

## 1️⃣ Running [4]
To run the method from [4], you can clone their repository [Landmarks2Anthropometry](https://github.com/DavidBoja/Landmarks2Anthropometry) and switch to the eccv24 branch where the authors provide the scripts to run their method on the datasets from the paper. The datasets are set with the variable `dataset_path` and correspond to the ones created in [💻 Datasets](#💻-datasets).

To evaluate on the CAESAR A-pose, run:
```bash
python eccv_stand.py --dataset_path <path/to/CAESAR/Apose/dataset>
```
To evaluate on the CAESAR A-pose with noisy landmarks, run:
```bash
python eccv_noisy.py --dataset_path <path/to/CAESAR/Apose/noisy/dataset>
```
To evaluate on the CAESAR sitting B-pose, run:
```bash
python eccv_sit.py --dataset_path <path/to/CAESAR/Bpose/dataset>
```
To evaluate on the CAESAR arbitrary pose, run:
```bash
python eccv_posed.py --dataset_path <path/to/CAESAR/posed/dataset>
```

<br>
<br>

## 📝 Notes

### Subjects
The subjects we use for training, validation and testing are the same ones as used in [1], excluding the ones that have missing landmarks or measurements. See paper for more details.


### Latex tables

We provide the latex tables from the paper so you can easily compare with our model:

- Table 1 is provided in `latex_tables/caesar-standing-male-results.tex`
- Table 2 is provided in `latex_tables/caesar-standing-female-results.tex`
- Table 3 is provided in `latex_tables/caesar-sit-results.tex`
- Table 4 is provided in `latex_tables/caesar-posed-results.tex`
- Table 5 is provided in `latex_tables/dyna-results.tex`
- Table 6 is provided in `latex_tables/four-d-human-outfit-results.tex`

### Find pose invariant features
We already provide the pose-invariant features in `data/landmarks2features/lm2features_distances_grouped_from_SMPL_INDEX_LANDAMRKS_REVISED_inds_removed_inds_with_median_dist_bigger_than_one.npy`.

To recreate these features or create others, you can run:
```bash
cd scripts
python find_pose_invariant_landmark_features.py --caesar_dir <path/to/caesar/dataset>
```


### Landmark-measurement ambiguity

To create Figure 4 from the paper you can run:
```bash
cd scripts
python find_landmark_measurements_ambiguity.py
```

which will create a figure named `ambiguity_max_landmarks_wrt_measurements.pdf`.



## 📊 Dataset statistics

To find out the average displacement of each landmark in the Noisy CAESAR dataset run:

```bash
python compute_stats.py NoisyCaesar
```

To find out how many and which landmarks are missing in the original CAESAR sitting dataset, run:

```bash
python compute_stats.py CAESAR_SITTING
```



## References

Parts of the code are inspired from [smplify-x](https://github.com/vchoutas/smplify-x) and [3D-CODED](https://github.com/ThibaultGROUEIX/3D-CODED/tree/master).
We thank the authors for providing the code.

<br>

[1] Tsoli et al.: "Model-based Anthropometry: Predicting Measurements from 3D Human Scans in Multiple Poses" <br>
[2] Pavlakos et al.: "Expressive Body Capture: 3D Hands, Face, and Body from a Single Image" <br>
[3] Marsot et al.: "Representing motion as a sequence of latent primitives, a flexible approach for human motion modelling" <br>
[4] Bojanić et al.: "Direct 3D Body Measurement Estimation from Sparse Landmarks"
