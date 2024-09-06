

from utils import load_config, set_seed
import time
from termcolor import colored

from trainers import TrainLm2MeasPosedReal as Trainer




if __name__ == "__main__":

    opt = load_config("configs/config_real.yaml")

    continuing_experiment = opt["general"]["continue_experiment"]
    if not isinstance(continuing_experiment,type(None)):
        opt = load_config(f"results/{continuing_experiment}/config.yaml")
        opt["general"]["continue_experiment"] = continuing_experiment
    
    set_seed(opt["learning"]["seed"])
    
    trainer = Trainer(opt)
    trainer.build_dataset_train()
    trainer.build_dataset_val()
    trainer.build_network()
    trainer.build_optimizer()
    trainer.build_losses()
    trainer.start_train_time = time.time()

for epoch in range(opt["learning"]["nepoch"]):
    print(colored(f"Epoch {epoch}", "red"))
    trainer.train_epoch()
    trainer.val_epoch()
    trainer.dump_stats()
    trainer.save_network()
    trainer.increment_epoch()