import argparse
from datetime import datetime
from functools import partial
from pathlib import Path
from pprint import pprint
from random import randint
from collections.abc import Mapping

import torch
import wandb
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
import os
import numpy as np
from network import R3D_model_online
from network.S3D_online import ModifiedS3D, replace_layers_arch_v1, replace_layers_arch_v2
from dataloaders.dataset import VideoDataset
from losses.fast_loss import FastCrossEntropyLoss
from torchvision.models.video import R3D_18_Weights, S3D_Weights, r3d_18, s3d
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\033[0;1;31m{device=}\033[0m")

dyn_depth_channel_stats = 16

def R3D_online_v1(num_classes, **kwargs,):
    model = R3D_model_online.ModifiedResNet(num_classes=num_classes)
    R3D_model_online.replace_layers_arch_v1(model, depth_channel_stats=16)
    model.layer4[0].conv1[1].depth_channel_stats = 2
    model.layer4[0].conv2[1].depth_channel_stats = 2
    model.layer4[0].downsample[1].depth_channel_stats = 2
    model.layer4[1].conv1[1].depth_channel_stats = 2
    model.layer4[1].conv2[1].depth_channel_stats = 2
    return model

def R3D_online_v2(num_classes, **kwargs,):
    model = R3D_model_online.ModifiedResNet(num_classes=num_classes)
    R3D_model_online.replace_layers_arch_v2(model, depth_channel_stats=16)
    model.layer4[0].conv1[1].depth_channel_stats = 2
    model.layer4[0].conv2[1].depth_channel_stats = 2
    model.layer4[0].downsample[1].depth_channel_stats = 2
    model.layer4[1].conv1[1].depth_channel_stats = 2
    model.layer4[1].conv2[1].depth_channel_stats = 2
    return model


def S3D_online_v1(num_classes, **kwargs,):
    model = ModifiedS3D(num_classes=num_classes)
    replace_layers_arch_v1(model, depth_channel_stats = dyn_depth_channel_stats)
    return model

def S3D_online_v2(num_classes, **kwargs,):
    model = ModifiedS3D(num_classes=num_classes)
    replace_layers_arch_v2(model, depth_channel_stats = dyn_depth_channel_stats)
    return model

def R3D_18(num_classes, **kwargs,):
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    return model

def S3D(num_classes, **kwargs,):
    weights = S3D_Weights.DEFAULT
    model = s3d(weights=weights)
    dropoout = model.classifier[0]
    num_ftrs = model.classifier[1].in_channels
    kernel_size = model.classifier[1].kernel_size
    stride = model.classifier[1].stride
    model.classifier = torch.nn.Identity()
    model.classifier = torch.nn.Sequential(dropoout,
                                    torch.nn.Conv3d(num_ftrs, num_classes, kernel_size=kernel_size, stride=stride))
    return model

def nested_dict(opts):
    res = {}
    for keys, val in opts.items():
        tmp = res
        levels = keys.split(".")
        for level in levels[:-1]:
            tmp = tmp.setdefault(level, {})
        tmp[levels[-1]] = val
    return res

def update(orig, new):
    for key, val in new.items():
        if isinstance(val, Mapping):
            orig[key] = update(orig.get(key, type(val)()), val)
        else:
            orig[key] = val
    return orig

def fit_model(config: dict, sweep: bool = False) -> None:
    dir_name = str(datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    if sweep:
        run = wandb.init(settings=wandb.Settings(start_method="fork"))
        dir_name = str(run.name)
        opts = wandb.config
        opts = nested_dict(opts)
        config = update(config, opts)
    
    config["results"]["datetime"] = dir_name
    pprint(config)

    if config["exp_params"]["seed"] is None:
        config["exp_params"]["seed"] = randint(10, 10000)
    torch.manual_seed(config["exp_params"]["seed"])
    np.random.seed(config["exp_params"]["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed(config["exp_params"]["seed"])

    train_dataloader = DataLoader(VideoDataset(dataset='ucf101', model = config["exp_params"]["model_name"], split='train',clip_len=16), batch_size=config["exp_params"]["train_bs"], shuffle=True, num_workers=4)
    val_dataloader   = DataLoader(VideoDataset(dataset='ucf101', model = config["exp_params"]["model_name"], split='val',  clip_len=16), batch_size=config["exp_params"]["train_bs"], num_workers=4)
    test_dataloader  = DataLoader(VideoDataset(dataset='ucf101', model = config["exp_params"]["model_name"],  split='test', clip_len=16), batch_size=config["exp_params"]["train_bs"], num_workers=4)


    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)


    
    if config["exp_params"]["model_name"] == "S3D_online_v1":
        model = S3D_online_v1(num_classes=101)
    elif config["exp_params"]["model_name"] == "S3D_online_v2":
        model = S3D_online_v2(num_classes=101)
    elif config["exp_params"]["model_name"] == "R3D_online_v1":
        model = R3D_online_v1(num_classes=101)
    elif config["exp_params"]["model_name"] == "R3D_online_v2":
        model = R3D_online_v2(num_classes=101)
    elif config["exp_params"]["model_name"] == "S3D":
        model = S3D(num_classes=101)
    elif config["exp_params"]["model_name"] == "R3D_18":
        model = R3D_18(num_classes=101)
    else:
        raise NotImplemented()
    
    ########## finetuning model ######
    
    # resume_path = "/resume_path/checkpoint/best_model.pth"

    # if torch.cuda.is_available():
    #     checkpoint = torch.load(resume_path)
    # else:
    #     checkpoint = torch.load(resume_path, map_location="cpu")

    # model.load_state_dict(checkpoint["model_state_dict"])
    # print(f"Successfully loaded model from {resume_path}")
    #####################################

    model = model.to(device)

    
    if config["exp_params"]["loss"] == "ce":
        criterion = torch.nn.CrossEntropyLoss()
        print("\033[0;36mUsing cross entropy loss!\033[0m")
    elif config["exp_params"]["loss"] == "custom_ce":
        criterion = CustomCrossEntropyLoss(exponent_value=10)
        print("\033[0;36mUsing custom cross entropy loss!\033[0m")
    elif config["exp_params"]["loss"] == "fast_ce":
        initial_lambda_value = config["exp_params"]["lambda_value"]
        criterion = FastCrossEntropyLoss(lambda_value=initial_lambda_value)
        print("\033[0;36mUsing fast cross entropy loss!\033[0m")
    else:
        raise NotImplemented()

    results_dirs = {
        "root": f"{config['results']['root']}/{config['results']['datetime']}"
    }
    for key, val in config["results"]["dirs"].items():
        results_dirs[key] = f"{results_dirs['root']}/{val}"
        Path(results_dirs[key]).mkdir(exist_ok=True, parents=True)

    if config["results"]["logger"] == "wandb":
        if not sweep:
            wandb.init(
                project=config["name"],
                entity="root",
                config=config,
                dir=results_dirs["log"],
                name=config["results"]["datetime"],
                settings=wandb.Settings(start_method="fork"),
            )

        for step_name in ["NewPart", "FineTune"]:
            wandb.define_metric(f"{step_name}_epoch")
            for phase in ["train", "test"]:
                wandb.define_metric(f"{step_name}_{phase}/epoch")
                wandb.define_metric(
                    f"{step_name}_{phase}/loss", step_metric=f"{step_name}_{phase}/epoch"
                )

            wandb.define_metric(
                f"{step_name}_train/lr", step_metric=f"{step_name}_epoch"
            )
            wandb.define_metric(
                f"{step_name}_test/ACC", step_metric=f"{step_name}_test_epoch"
            )

    if isinstance(config["exp_params"]["epochs"], str):
        config["exp_params"]["epochs"] = {
            key: val
            for key, val in zip(
                ["NewPart", "FineTune"],
                [int(x) for x in config["exp_params"]["epochs"].split("-")],
            )
        }
    if isinstance(config["exp_params"]["lr"], str):
        config["exp_params"]["lr"] = {
            key: float(val)
            for key, val in zip(
                ["NewPart", "FineTune"],
                [float(x) for x in config["exp_params"]["lr"].split("-")],
            )
        }

    if config["results"]["save_model"]:
        config["results"]["resume_path"] = f"{results_dirs['checkpoint']}/model.pth"
    with open(f"{results_dirs['root']}/config.yaml", "w") as yaml_file:
        yaml.safe_dump(
            config,
            yaml_file,
            default_style=None,
            default_flow_style=False,
            sort_keys=False,
        )
    
    for step_name in ["NewPart", "FineTune"]:
        if config["exp_params"]["epochs"][step_name] == 0:
            continue

        if step_name == "NewPart":
            for param in model.parameters():
                    param.requires_grad = False

            for param in model.fc.parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\033[0;1;31mThe number of parameters to train: {num_params:.0f}\033[0m")

        if config["exp_params"]["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config["exp_params"]["lr"][step_name],
                weight_decay=config["exp_params"]["weight_decay"],
            )
        elif config["exp_params"]["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config["exp_params"]["lr"][step_name],
                momentum=0.9,
            )
            

        scheduler = None
        if config["exp_params"]["scheduler"] is not None:
            gamma = (
                config["exp_params"]["min_lr_sheduler"] / config["exp_params"]["lr"][step_name]
            ) ** (1 / (0.9 * config["exp_params"]["epochs"][step_name]))
            if config["exp_params"]["scheduler"] == "CyclicLR":
                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=config["exp_params"]["min_lr_sheduler"],
                    max_lr=1.2 * config["exp_params"]["lr"][step_name],
                    step_size_up=10,
                    mode="exp_range",
                    scale_mode="cycle",
                    cycle_momentum=False,
                    gamma=gamma,
                )
            elif config["exp_params"]["scheduler"] == "LambdaLR":
                labda = (
                    lambda v: gamma**v
                    if v < 0.9 * config["exp_params"]["epochs"][step_name]
                    else config["exp_params"]["min_lr_sheduler"]
                    / config["exp_params"]["lr"][step_name]
                )
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=labda
                )

        maxACC = 0
        highest_probs_indx_list = []
        expected_values_indx_list = []
        all_first_frame_indices = []
        for epoch in range(config["exp_params"]["epochs"][step_name]):
            ####################################
            #             train                #
            ####################################
            for phase in ['train', 'val']:
                running_loss = 0.0
                running_corrects = 0.0
            
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                for inputs, labels in tqdm(trainval_loaders[phase]):
                    inputs = Variable(inputs, requires_grad=True).to(device)
                    labels = Variable(labels).to(device)

                    optimizer.zero_grad()

                    if phase == 'train':
                        outputs = model(inputs)

                    else:
                        with torch.no_grad():
                            outputs = model(inputs)
                    
                    proba = torch.softmax(outputs, dim=-1)
                    probs, _ = torch.max(proba, dim=1)
                    outputs = outputs.max(dim=1).values
                    loss, _ = criterion(outputs, labels)
                    assert torch.isfinite(loss).item(), f"Loss is not finite [{loss}]"

                    preds = torch.argmax(probs, dim = -1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / trainval_sizes[phase]
                epoch_acc = running_corrects.double() / (trainval_sizes[phase])

                # ===================logger========================
                if phase == 'train':
                    if config["results"]["logger"] == "wandb":
                        wandb.log(
                            {
                                f"{step_name}_train/loss": epoch_loss,
                                f"{step_name}_train/epoch": epoch,
                            }
                        )
                        wandb.log(
                            {
                                f"{step_name}_train/acc": epoch_acc,
                                f"{step_name}_train/epoch": epoch,
                            }
                        )

                else:
                    if config["results"]["logger"] == "wandb":
                        wandb.log(
                            {
                                f"{step_name}_val/loss": epoch_loss,
                                f"{step_name}_val/epoch": epoch,
                            }
                        )
                        wandb.log(
                            {
                                f"{step_name}_val/acc": epoch_acc,
                                f"{step_name}_val/epoch": epoch,
                            }
                        )

                
                print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, config["exp_params"]["epochs"][step_name], epoch_loss, epoch_acc))

            ####################################
            #          test step               #
            ####################################
            model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)

                loss, idx, probs, _ = criterion(outputs, labels)
                assert torch.isfinite(loss).item(), f"Loss is not finite [{loss}]"
        
                highest_probs_indx = idx.gather(1, torch.argmax(probs, dim=-1).unsqueeze(1)).squeeze()
                if highest_probs_indx.ndimension() == 0:
                    continue
                else:
                    highest_probs_indx_list.extend(highest_probs_indx.tolist())

                preds = torch.argmax(probs, dim = -1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / (test_size)
            print("[test]] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, config["exp_params"]["epochs"][step_name], epoch_loss, epoch_acc))
            # ===================logger========================
            if config["results"]["logger"] == "wandb":
                wandb.log(
                    {
                        f"{step_name}_test/loss": epoch_loss,
                        f"{step_name}_test/epoch": epoch,
                    }
                )
                wandb.log(
                    {
                        f"{step_name}_test/ACC": epoch_acc,
                        f"{step_name}_test/epoch": epoch,
                    }
                )

            ############################
            #       save model         #
            ############################
            if step_name == "FineTune":
                if config["results"]["save_model"]:
                    torch.save(
                        {
                            **{"model_state_dict": model.state_dict(), "epoch": epoch},
                        },
                        config["results"]["resume_path"],
                    )

                    if maxACC < epoch_acc.item():
                        maxACC = epoch_acc.item()
                        torch.save(
                            {
                                **{
                                    "model_state_dict": model.state_dict(),
                                    "epoch": epoch,
                                },
                            },
                            f"{results_dirs['checkpoint']}/best_model.pth",
                        )

            ####################################
            #           scheduler              #
            ####################################
            if scheduler is not None:
                if config["results"]["logger"] == "wandb":
                    wandb.log(
                        {
                            f"{step_name}_train/lr": scheduler.get_last_lr()[0],
                            f"{step_name}_epoch": epoch,
                        }
                    )
                scheduler.step()

        
        
        wandb.log({"Histogram of decisive frame number": wandb.Histogram(highest_probs_indx_list)})
        mean_value = sum(highest_probs_indx_list)/ len(highest_probs_indx_list)
        NET = mean_value / 16
        wandb.log({"NET": NET})

    if config["results"]["logger"] == "wandb":
        wandb.finish()

    print("\033[0;32mDONE\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner for VideoClassifier")
    parser.add_argument(
        "--config",
        help="path to the config file",
        default="configs/demo.yaml",
    )

    parser.add_argument("--run", choices=["train", "sweep"], default="train")
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--sweep_count", type=int, default=5)
    args = parser.parse_args()
    print(args)
    params = vars(args)

    if args.run == "train":
        with open(args.config, "r") as file:
            try:
                cfg = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        params = update(params, cfg)
        fit_model(params)
    else:
        with open(args.config, "r") as file:
            try:
                cfg = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        if args.sweep_id is None:
            args.sweep_id = wandb.sweep(
                sweep=cfg,
                entity="root",
                project=cfg["name"],
            )
            print(f"\033[0;1;31mSweep ID: {args.sweep_id}\033[0m")
            with open(
                f"{cfg['parameters']['results.root']['value']}/sweepID.txt", "w"
            ) as f:
                print(args.sweep_id, end="", file=f)

        wandb.agent(
            args.sweep_id,
            function=partial(fit_model, params, True),
            entity="root",
            project=cfg["name"],
            count=args.sweep_count,
        )
