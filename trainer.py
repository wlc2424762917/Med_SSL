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


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            if args.seg_sr:
                data, seg_target, sr_target = batch_data
            else:
                data, target = batch_data
        else:
            if args.seg:
                data, target = batch_data["image"], batch_data["label_seg"]
            elif args.sr:
                data, target = batch_data["image"], batch_data["label_sr"]
            elif args.seg_sr:
                data, seg_target, sr_target = batch_data["image"], batch_data["label_seg"], batch_data["label_sr"]
            else:
                raise NotImplementedError

        if args.seg_sr:
            data = data[:, :, :, :]
            seg_target = seg_target[:, :, :, :]
            sr_target = sr_target[:, :, :, :]
            data, seg_target, sr_target = data.cuda(args.rank), seg_target.cuda(args.rank), sr_target.cuda(args.rank)
        else:
            data = data[:, :, :, :]  # B, C, H, W
            target = target[:, :, :, :]
            data, target = data.cuda(args.rank), target.cuda(args.rank)

        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            if args.seg_sr:
                logits = model(data)
                seg_logits, sr_logits = logits
                seg_loss_func, sr_loss_func = loss_func
                seg_loss = seg_loss_func(seg_logits, seg_target)
                sr_loss = sr_loss_func(sr_logits, sr_target)
                loss = args.segloss2srloss * seg_loss + args.srloss2segloss * sr_loss
            else:
                logits = model(data)
                loss = loss_func(logits, target)
            #
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
        if args.rank == 0:
            if args.seg_sr:
                print(
                    "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "loss: {:.4f}".format(run_loss.avg),
                    "seg_loss: {:.4f}".format(seg_loss.item()),
                    "sr_loss: {:.4f}".format(sr_loss.item()),
                    "time {:.2f}s".format(time.time() - start_time),
                )
            else:
                print(
                    "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "loss: {:.4f}".format(run_loss.avg),
                    "time {:.2f}s".format(time.time() - start_time),
                )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                if args.seg:
                    data, target = batch_data["image"], batch_data["label_seg"]
                elif args.sr:
                    data, target = batch_data["image"], batch_data["label_sr"]
            data = data[:, :, :, :]  # B, C, H, W
            target = target[:, :, :, :]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            if post_label is not None:
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            else:
                val_labels_convert = val_labels_list
            val_outputs_list = decollate_batch(logits)
            if post_pred is not None:
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            else:
                val_output_convert = val_outputs_list

            acc_func(y_pred=val_output_convert, y=val_labels_convert)
        acc = acc_func.aggregate().item()
        acc_func.reset()

        if args.rank == 0:
            print(
                "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "acc",
                acc,
                "time {:.2f}s".format(time.time() - start_time),
            )
    return acc


def val_epoch_seg_sr(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, seg_target, sr_target = batch_data
            else:
                data, seg_target, sr_target = batch_data["image"], batch_data["label_seg"], batch_data["label_sr"]
            data = data[:, :, :, :]  # B, C, H, W
            seg_target = seg_target[:, :, :, :]
            sr_target = sr_target[:, :, :, :]
            data, seg_target, sr_target = data.cuda(args.rank), seg_target.cuda(args.rank), sr_target.cuda(args.rank)

            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
                logits_seg, logits_sr = logits

            if not logits_seg.is_cuda:
                seg_target = seg_target.cpu()
                sr_target = sr_target.cpu()
            val_labels_list_seg = decollate_batch(seg_target)
            val_labels_list_sr = decollate_batch(sr_target)
            val_labels_convert_seg = [post_label(val_label_tensor) for val_label_tensor in val_labels_list_seg]
            val_labels_convert_sr = val_labels_list_sr

            val_outputs_list_seg = decollate_batch(logits_seg)
            val_output_convert_seg = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list_seg]
            val_outputs_list_sr = decollate_batch(logits_sr)
            val_output_convert_sr = val_outputs_list_sr

            acc_func_seg, acc_func_sr = acc_func
            acc_func_seg(y_pred=val_output_convert_seg, y=val_labels_convert_seg)
            acc_func_sr(y_pred=val_output_convert_sr, y=val_labels_convert_sr)
            # acc_seg, not_nans = acc_func_seg.aggregate()
            # acc_sr, not_nans = acc_func_sr.aggregate()
        acc_seg = acc_func_seg.aggregate().item()
        acc_sr = acc_func_sr.aggregate().item()
        acc_func_seg.reset()
        acc_func_sr.reset()

        if args.rank == 0:
            avg_acc_seg = np.mean(acc_seg)
            avg_acc_sr = np.mean(acc_sr)
            print(
                "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "acc_seg",
                avg_acc_seg,
                "acc_sr",
                avg_acc_sr,
                "time {:.2f}s".format(time.time() - start_time),
            )
    return avg_acc_seg, avg_acc_sr


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
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
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
            if args.seg_sr:
                val_avg_acc_seg, val_avg_acc_sr = val_epoch_seg_sr(
                    model,
                    val_loader,
                    epoch=epoch,
                    acc_func=acc_func,
                    model_inferer=model_inferer,
                    args=args,
                    post_label=post_label,
                    post_pred=post_pred,)
                # 根据seg的acc来决定是否保存模型
                if args.val_metric == "dice":
                    val_avg_acc = np.mean(val_avg_acc_seg)
                else:
                    val_avg_acc = np.mean(val_avg_acc_sr)

            else:
                val_avg_acc = val_epoch(
                    model,
                    val_loader,
                    epoch=epoch,
                    acc_func=acc_func,
                    model_inferer=model_inferer,
                    args=args,
                    post_label=post_label,
                    post_pred=post_pred,
                )

                val_avg_acc = np.mean(val_avg_acc)

            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
