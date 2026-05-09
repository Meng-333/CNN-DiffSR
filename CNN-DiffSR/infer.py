import argparse
import os

import numpy as np
import torch
from litsr.data import PairedImageDataset, SingleImageDataset, DownsampledDataset
from litsr.metrics import *
from litsr.utils import mkdirs, read_yaml
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from tqdm import tqdm
import random

from models import load_model

from skimage import io

seed_everything(123)


def make_dataloaders(datasets, type, scale, config):
    dataloaders = []
    for dataset_name in datasets:
        if type == "LRHR_paired":
            dataset = PairedImageDataset(
                lr_path="load/benchmark/{0}/LR_bicubic/X{1}".format(
                    dataset_name, scale
                ),
                hr_path="load/benchmark/{0}/HR".format(dataset_name),
                scale=scale,
                is_train=False,
                cache="bin",
                rgb_range=config.data_module.args.rgb_range,
                mean=config.data_module.args.get("mean"),
                std=config.data_module.args.get("std"),
                return_img_name=True,
            )
        elif type == "LR_only":
            dataset = SingleImageDataset(
                img_path="load/NWPU_infer".format(dataset_name),
                rgb_range=config.data_module.args.rgb_range,
                cache="bin",
                mean=config.data_module.args.get("mean"),
                std=config.data_module.args.get("std"),
                return_img_name=True,
            )
        elif type == "HR_only":
            dataset = SingleImageDataset(
                img_path="load/Test/AID_Test".format(dataset_name),
                #img_path="load/Test/UCM_Test".format(dataset_name),
                #img_path="load/Test/NWPU_21Class_HR".format(dataset_name),
                #img_path="load/Test/NWPU_Test/tennis_court".format(dataset_name),
                rgb_range=config.data_module.args.rgb_range,
                cache="bin",
                mean=config.data_module.args.get("mean"),
                std=config.data_module.args.get("std"),
                return_img_name=True,
            )
        elif type == "HR_downsampled":
            dataset = DownsampledDataset(
                datapath="load/benchmark/{0}/HR".format(dataset_name),
                scale=scale,
                is_train=False,
                rgb_range=config.data_module.args.rgb_range,
                cache="bin",
                mean=config.data_module.args.get("mean"),
                std=config.data_module.args.get("std"),
                return_img_name=True,
            )
        else:
            raise "Unknown dataset type"
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        dataloaders.append((dataset_name, loader))
    return dataloaders


def test(args):
    # setup device
    if args.random_seed:
        seed_everything(random.randint(0, 1000))
    device = (
        torch.device("cuda", index=int(args.gpu)) if args.gpu else torch.device("cpu")
    )

    # setup datasets
    test_datasets = [_ for _ in args.datasets.split(",")]

    exp_path = os.path.dirname(os.path.dirname(args.checkpoint))
    ckpt_path = args.checkpoint

    # read config
    config = read_yaml(os.path.join(exp_path, "hparams.yaml"))

    config.lit_model.args.valid.skip = args.skip
    config.lit_model.args.valid.eta = args.eta

    # create model
    model = load_model(config, ckpt_path, strict=False)
    model.to(device)
    model.eval()

    scale = config.data_module.args.scale

    dataloaders = make_dataloaders(
        datasets=test_datasets, type=args.datatype, scale=scale, config=config
    )

    f = open('./logs/cnn-diffsr_x4/test_log.txt', 'a')

    for dataset_name, loader in dataloaders:
        # config result path
        rslt_path = os.path.join(
            exp_path,
            "results",
            dataset_name,
            "x" + str(scale),
        )

        if args.eta != 1:
            rslt_path = rslt_path + "_eta_{0}".format(args.eta)
        if args.skip != 50:
            rslt_path = rslt_path + "_skip_{0}".format(args.skip)
        if args.approxdiff != "STEP":
            rslt_path = rslt_path.replace(
                "x4",
                "x4_approx_{0}_schedule_{1}".format(args.approxdiff, args.schedule),
            )
        lr_path = rslt_path.replace("results", "lr_sample")
        mkdirs([rslt_path, lr_path])

        psnrs, ssims, run_times, losses = [], [], [], []
        mses, ergass, lpipss = [], [], []
        bmses, bpsnrs, bssims, bergass, blpipss = [], [], [], [], []

        print("approxdiff: {0}, schedule: {1}".format(args.approxdiff, args.schedule))
        print3 = "approxdiff: {0}, schedule: {1}".format(args.approxdiff, args.schedule)
        f.write(print3 + os.linesep)

        for batch in tqdm(loader, total=len(loader.dataset)):
            if args.datatype in ("LRHR_paired", "HR_downsampled"):
                lr, hr, name = batch
                batch = (lr.to(device), hr.to(device), name)
            elif args.datatype == "LR_only":
                lr, name = batch
                batch = (lr.to(device), name)
            elif args.datatype == "HR_only":
                hr, name = batch
                h, w = hr.shape[2:]
                h_, w_ = h % scale, w % scale
                hr = hr[:, :, : (h - h_), : (w - w_)]
                batch = (hr.to(device), name)
            else:
                raise "Unknown datatype"

            # do test
            with torch.no_grad():
                if args.datatype == "LR_only":
                    rslt = model.test_step_lr_only(batch, 1)
                else:
                    rslt = model.test_step(
                        batch, 1, approxdiff=args.approxdiff, schedule=args.schedule
                    )

            file_path = os.path.join(rslt_path, rslt["name"])
            plot_path = os.path.join(rslt_path, rslt["name"].split('.')[0] + '_plot.png')
            lr_file_path = os.path.join(lr_path, rslt["name"])

            # print(rslt)
            # break
            if "log_img_sr" in rslt.keys():
                io.imsave(file_path, rslt["log_img_sr"])
            #if "log_img_lr" in rslt.keys():
                #io.imsave(lr_file_path, rslt["log_img_lr"])
            if "log_img_plot" in rslt.keys():
                plot_img(rslt["log_img_plot"][0], rslt["log_img_plot"][1], rslt["log_img_plot"][2],
                rslt["log_img_plot"][3], rslt["log_img_plot"][4], rslt["log_img_plot"][5], plot_path)
            if "ctx" in rslt.keys():
                io.imsave(file_path.replace(".png", "_ctx.png"), rslt["ctx"])
            if "val_loss" in rslt.keys():
                losses.append(rslt["val_loss"])

            if "val_bmse" in rslt.keys():
                bmses.append(rslt["val_bmse"])
            if "val_bpsnr" in rslt.keys():
                bpsnrs.append(rslt["val_bpsnr"])
            if "val_bssim" in rslt.keys():
                bssims.append(rslt["val_bssim"])
            if "val_bergas" in rslt.keys():
                bergass.append(rslt["val_bergas"])
            if "val_blpips" in rslt.keys():
                blpipss.append(rslt["val_blpips"])

            if "val_mse" in rslt.keys():
                mses.append(rslt["val_mse"])
            if "val_psnr" in rslt.keys():
                psnrs.append(rslt["val_psnr"])
            if "val_ssim" in rslt.keys():
                ssims.append(rslt["val_ssim"])
            if "val_ergas" in rslt.keys():
                ergass.append(rslt["val_ergas"])
            if "val_lpips" in rslt.keys():
                lpipss.append(rslt["val_lpips"])

            if "time" in rslt.keys():
                run_times.append(rslt["time"])

        if losses:
            mean_loss = torch.stack(losses).mean()
            print("- Loss: {:.4f}".format(mean_loss))
            print4 = "- Loss: {:.4f}".format(mean_loss)
            f.write(print4 + os.linesep)

        if bmses:
            mean_bmse = np.array(bmses).mean()
            print("- BIC_MSE: {:.5f}".format(mean_bmse))
            print5 = "- BIC_MSE: {:.5f}".format(mean_bmse)
            f.write(print5 + os.linesep)
        if bpsnrs:
            mean_bpsnr = np.array(bpsnrs).mean()
            print("- BIC_PSNR: {:.5f}".format(mean_bpsnr))
            print6 = "- BIC_PSNR: {:.5f}".format(mean_bpsnr)
            f.write(print6 + os.linesep)
        if bssims:
            mean_bssim = np.array(bssims).mean()
            print("- BIC_SSIM: {:.5f}".format(mean_bssim))
            print7 = "- BIC_SSIM: {:.5f}".format(mean_bssim)
            f.write(print7 + os.linesep)
        if bergass:
            mean_bergas = np.array(bergass).mean()
            print("- BIC_ERGAS: {:.5f}".format(mean_bergas))
            print8 = "- BIC_ERGAS: {:.5f}".format(mean_bergas)
            f.write(print8 + os.linesep)
        if blpipss:
            mean_blpips = np.array(blpipss).mean()
            print("- BIC_LPIPS: {:.5f}".format(mean_blpips))
            print9 = "- BIC_LPIPS: {:.5f}".format(mean_blpips)
            f.write(print9 + os.linesep)

        if mses:
            mean_mse = np.array(mses).mean()
            print("- SR_MSE: {:.5f}".format(mean_mse))
            print10 = "- SR_MSE: {:.5f}".format(mean_mse)
            f.write(print10 + os.linesep)
        if psnrs:
            mean_psnr = np.array(psnrs).mean()
            print("- SR_PSNR: {:.5f}".format(mean_psnr))
            print11 = "- SR_PSNR: {:.5f}".format(mean_psnr)
            f.write(print11 + os.linesep)
        if ssims:
            mean_ssim = np.array(ssims).mean()
            print("- SR_SSIM: {:.5f}".format(mean_ssim))
            print12 = "- SR_SSIM: {:.5f}".format(mean_ssim)
            f.write(print12 + os.linesep)
        if ergass:
            mean_ergas = np.array(ergass).mean()
            print("- SR_ERGAS: {:.5f}".format(mean_ergas))
            print13 = "- SR_ERGAS: {:.5f}".format(mean_ergas)
            f.write(print13 + os.linesep)
        if lpipss:
            mean_lpips = np.array(lpipss).mean()
            print("- SR_LPIPS: {:.5f}".format(mean_lpips))
            print14 = "- SR_LPIPS: {:.5f}".format(mean_lpips)
            f.write(print14 + os.linesep)


        if run_times:
            mean_runtime = np.array(run_times[1:]).mean()
            print("- Runtime : {:.5f}".format(mean_runtime))
            print15 = "- Runtime : {:.5f}".format(mean_runtime)
            f.write(print15 + os.linesep)

        print("=" * 42)
        print17 = "=" * 42
        f.write(print17 + os.linesep)
        f.close()


def getTestParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, help="checkpoint index")
    parser.add_argument(
        "-g", "--gpu", default="0", type=str, help="indices of GPUs to enable"
    )
    parser.add_argument(
        "--datasets",default="load/Test/NWPU_infer", type=str, help="dataset names"

    )
    parser.add_argument(
        "--datatype",
        default="LR_only",
        type=str,
        help="dataset type, options: (HR_only, LR_only, LRHR_paired)",
    )
    parser.add_argument("--skip", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0)
    parser.add_argument("--random_seed", action="store_true")
    parser.add_argument("--approxdiff", default="STEP")
    parser.add_argument("--schedule", default="linear")

    return parser

test_parser = getTestParser()

if __name__ == "__main__":
    args = test_parser.parse_args()
    test(args)
