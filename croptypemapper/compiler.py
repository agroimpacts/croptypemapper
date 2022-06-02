import boto3
import urllib.parse as urlparse
import os
from pathlib import Path
from datetime import datetime, timedelta
import torch
from torch import optim
from tensorboardX import SummaryWriter
from utils import get_optimizer, PolynomialLR
from .train import train
from .validate import validate
from .accuracy import accuracy_evaluation
from .predict import inference

class ModelCompiler:
    """
    Compiler of specified model
    Args:
        model (''nn.Module'') -- pytorch model for segmentation.
        working_dir (sys.path or str) -- path to the working directory.
        out_dir (sys.path or str) -- Path to the directory to store output prediction and associated files.
        gpuDevices (tuple) -- indices of gpu devices to use.
        br_weights (tuple) -- weights to decide the influence of each triple branches in the LSTM model (e.g. s1, s2, fused).
        params_init (sys.path or str) -- Path to the saved model parameters to load.
        freeze_params (list of int) -- list of indices of the trainable layers in the network to freeze the gradients.
                                       Useful in fine-tunning the model.
    """

    def __init__(self, model, working_dir, out_dir, gpuDevices=(0),
                 br_weights=(0.3, 0.3, 0.4), params_init=None, freeze_params=None):

        self.s3_client = boto3.client("s3")
        self.working_dir = working_dir
        self.out_dir = out_dir
        self.gpuDevices = gpuDevices
        self.br_weights = br_weights
        self.model = model

        self.model_name = self.model.__class__.__name__

        if params_init:
            self.load_params(params_init, freeze_params)

        # gpu
        self.gpu = torch.cuda.is_available()
        if self.gpu:
            print("----------GPU available----------")
            # GPU setting
            if gpuDevices:
                torch.cuda.set_device(gpuDevices[0])
                self.model = torch.nn.DataParallel(self.model, device_ids=gpuDevices)
            self.model = self.model.cuda()

        num_params = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print("total number of trainable parameters: {:2.1f}M".format(num_params / 1000000))

        if params_init:
            print("---------- Pre-trained model compiled successfully ----------")
        else:
            print("---------- Vanilla Model compiled successfully ----------")

    def load_params(self, dir_params, freeze_params):

        params_init = urlparse.urlparse(dir_params)
        # load from s3
        if params_init.scheme == "s3":

            bucket = params_init.netloc
            params_key = params_init.path
            params_key = params_key[1:] if params_key.startswith('/') else params_key
            _, fn_params = os.path.split(params_key)

            self.s3_client.download_file(Bucket=bucket,
                                         Key=params_key,
                                         Filename=fn_params)
            inparams = torch.load(fn_params, map_location="cuda:{}".format(self.gpuDevices[0]))

            os.remove(fn_params)  # remove after loaded

        ## or load from local
        else:
            inparams = torch.load(dir_params)

        ## overwrite model entries with new parameters
        model_dict = self.model.state_dict()

        if "module" in list(inparams.keys())[0]:
            inparams_filter = {k[7:]: v.cpu() for k, v in inparams.items() if k[7:] in model_dict}

        else:
            inparams_filter = {k: v.cpu() for k, v in inparams.items() if k in model_dict}

        model_dict.update(inparams_filter)
        self.model.load_state_dict(model_dict)

        if freeze_params != None:
            for i, p in enumerate(self.model.parameters()):
                if i in freeze_params:
                    p.requires_grad = False

    def fit(self, trainDataset, valDataset, epochs, optimizer_name, lr_init,
            LR_policy, criterion, momentum=None):


        # Set the folder to save results.
        working_dir = self.working_dir
        out_dir = self.out_dir
        model_name = self.model_name
        self.model_dir = "{}/{}/{}_ep{}".format(working_dir, self.out_dir, model_name, epochs)

        if not os.path.exists(Path(working_dir) / out_dir / self.model_dir):
            os.makedirs(Path(working_dir) / out_dir / self.model_dir)

        os.chdir(Path(working_dir) / out_dir / self.model_dir)

        print("--------------- Start training ---------------")
        start = datetime.now()

        # Tensorboard writer setting
        writer = SummaryWriter('./')

        train_loss = []
        val_loss = []
        lr = lr_init

        optimizer = get_optimizer(optimizer_name, self.model.parameters(), lr, momentum)

        # Initialize the learning rate scheduler
        if LR_policy == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=10,
                                                  gamma=0.85, )
        elif LR_policy == "Exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                         gamma=0.85, )

        elif LR_policy == "PolynomialLR":
            scheduler = PolynomialLR(optimizer,
                                     max_decay_steps=100,
                                     min_learning_rate=1e-5,
                                     power=0.9)
        else:
            scheduler = None

        if isinstance(criterion, tuple) or isinstance(criterion, list):
            train_criterion = criterion[0]
            val_criterion = criterion[1]
        else:
            train_criterion = criterion
            val_criterion = criterion

        for t in range(epochs):

            print("[{}/{}]".format(t + 1, epochs))
            # start fitting
            start_epoch = datetime.now()
            train(trainDataset, self.model, train_criterion, optimizer, self.br_weights, gpu=self.gpu,
                  train_loss=train_loss)
            validate(valDataset, self.model, val_criterion, self.br_weights, gpu=self.gpu, val_loss=val_loss)

            # Update the scheduler
            if LR_policy in ["StepLR", "Exponential"]:
                scheduler.step()
                print("LR: {}".format(scheduler.get_last_lr()))

            if LR_policy == "PolynomialLR":
                scheduler.step(t)
                print("LR: {}".format(optimizer.param_groups[0]['lr']))

            # time spent on single iteration
            print("time:", (datetime.now() - start_epoch).seconds)

            # if t > 1 and t % lr_decay[1] == 0:
            # lr *= lr_decay[0]

            writer.add_scalars("Loss", {"train_loss": train_loss[t], "validation_loss": val_loss[t]}, t + 1)

            writer.close()

        print("--------------- Training finished in {}s ---------------".format((datetime.now() - start).seconds))

    def accuracy_evaluation(self, evalDataset, outPrefix, bucket=None):

        if not os.path.exists(Path(self.working_dir) / self.out_dir):
            os.makedirs(Path(self.working_dir) / self.out_dir)

        os.chdir(Path(self.working_dir) / self.out_dir)

        print("--------------- Start evaluation ---------------")
        start = datetime.now()

        accuracy_evaluation(evalDataset, self.model, self.gpu, outPrefix, self.br_weights, bucket)

        print("--------------- Evaluation finished in {}s ---------------".format((datetime.now() - start).seconds))

    def inference(self, predDataset, out_prefix=None):

        print("-------------------------- Start Inference(Test) --------------------------")

        start = datetime.now()
        if out_prefix is None:
            out_prefix = Path(self.working_dir) / self.out_dir / "Inference_output"

        prefix_hard = Path(out_prefix) / "HardScore"
        prefix_soft = Path(out_prefix) / "SoftProb"

        if not os.path.exists(prefix_hard):
            os.makedirs(prefix_hard)
        if not os.path.exists(prefix_soft):
            os.makedirs(prefix_soft)

        os.chdir(Path(out_prefix))

        inference(predDataset, self.model, prefix_soft, prefix_hard, gpu=self.gpu, weights=self.br_weights)

        duration_in_sec = (datetime.now() - start).seconds
        duration_format = str(timedelta(seconds=duration_in_sec))
        print("-------------------------- Inference finished in {}s --------------------------".format(duration_format))

    def save(self, save_fldr, bucket=None, save_object="params"):

        outPrefix = Path(self.working_dir) / self.out_dir / save_fldr

        if save_object == "params":

            fn_params = "{}_params.pth".format(self.model_name)

            if bucket:
                torch.save(self.model.state_dict(), fn_params)

                self.s3_client.upload_file(Filename=fn_params,
                                           Bucket=bucket,
                                           Key=os.path.join(outPrefix, fn_params))
                print("model parameters uploaded to s3!, at ", outPrefix)

                os.remove(Path(outPrefix) / fn_params)

            else:

                if not os.path.exists(Path(outPrefix)):
                    os.makedirs(Path(outPrefix))

                torch.save(self.model.state_dict(), Path(outPrefix) / fn_params)
                print("model parameters is saved locally, at ", outPrefix)

        elif save_object == "model":

            fn_model = "{}.pth".format(self.model_name)

            if bucket:
                torch.save(self.model, fn_model)

                self.s3_client.upload_file(Filename=fn_model,
                                           Bucket=bucket,
                                           Key=os.path.join(outPrefix, fn_model))
                print("model uploaded to s3!, at ", outPrefix)

                os.remove(Path(outPrefix) / fn_model)

            else:

                if not os.path.exists(Path(outPrefix)):
                    os.makedirs(Path(outPrefix))

                torch.save(self.model, Path(outPrefix) / fn_model)
                print("model saved locally, at ", outPrefix)

        else:
            raise ValueError("Object type is not acceptable.")

