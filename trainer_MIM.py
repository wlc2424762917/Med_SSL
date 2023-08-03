import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch


def train_epoch(model, loader, optimizer, scaler, epoch, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        img_data, mask_data = batch_data
        img = img_data["image"]
        data = img[:, :, :, :, 0]  # B, C, H, W
        data, mask_data = data.cuda(args.rank), mask_data.cuda(args.rank)

        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            loss = model(data, mask_data)
        if args.accu_grad == 1:
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if args.distributed:
                loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                run_loss.update(
                    np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
                )
            else:
                run_loss.update(loss.item(), n=args.batch_size)
        else:
            loss = loss / args.accu_grad
            if args.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (idx + 1) % args.accu_grad == 0:
                if args.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if args.distributed:
                    loss_list = distributed_all_gather(
                        [loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                    )
                    run_loss.update(
                        np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                        n=args.batch_size * args.world_size,
                    )
                else:
                    run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, args, model_inferer=None):
    model.eval()
    run_acc = 0
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            img_data, mask_data = batch_data
            img = img_data["image"]
            data = img[:, :, :, :, 0]  # B, C, H, W
            data, mask_data = data.cuda(args.rank), mask_data.cuda(args.rank)

            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data, mask_data)
                else:
                    logits = model(data, mask_data)
            if not logits.is_cuda:
                target = target.cpu()
            mean_logits = torch.mean(logits)
            acc = mean_logits.cuda(args.rank)

            if args.distributed:
                for al in acc:
                    acc = distributed_all_gather([al], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                    run_acc += np.mean(acc)
            else:
                run_acc += acc/len(loader)

            if args.rank == 0:
                avg_acc = np.mean(run_acc)
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",
                    acc/len(loader),
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
    return avg_acc


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_min = 100.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            print("Validating...")
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                model_inferer=model_inferer,
                args=args,
            )

            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc < val_acc_min:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_min, val_avg_acc))
                    val_acc_min = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_min, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_min, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_min)

    return val_acc_min
