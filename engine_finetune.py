import os
import csv
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional
import util.misc as misc
from timm.data import Mixup
import util.lr_sched as lr_sched
import pandas as pd

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None
):
    """Train the model for one epoch for age prediction (regression)."""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq, accum_iter = 20, args.accum_iter
    optimizer.zero_grad()

    if log_writer:
        print(f'log_dir: {log_writer.log_dir}')

    for data_iter_step, (_, samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples, targets = samples.to(device, non_blocking=True), targets.to(device, non_blocking=True).float()

        with torch.cuda.amp.autocast():
            outputs = model(samples).squeeze(1)  # shape [B]
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        loss /= accum_iter

        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        min_lr, max_lr = float('inf'), 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/train', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, args, epoch, mode, num_class, log_writer):
    """Evaluate the model for age prediction (regression)."""
    criterion = nn.MSELoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)

    model.eval()
    preds, trues = [], []
    name_list = []

    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        name, images, targets = batch[0], batch[1].to(device, non_blocking=True), batch[-1].to(device, non_blocking=True).float()
        with torch.cuda.amp.autocast():
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, targets)

        metric_logger.update(loss=loss.item())
        preds.extend(outputs.detach().cpu().numpy())
        trues.extend(targets.detach().cpu().numpy())
        name_list.extend(name)

    preds = np.array(preds)
    trues = np.array(trues)

    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(np.mean((preds - trues)**2))
    me = np.mean(trues - preds)

    if log_writer:
        log_writer.add_scalar(f'perf/mae', mae, epoch)
        log_writer.add_scalar(f'perf/rmse', rmse, epoch)

    print(f'ME: {me:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}')

    metric_logger.synchronize_between_processes()

    results_path = os.path.join(args.output_dir, args.task, f'metrics_{mode}.csv')
    file_exists = os.path.isfile(results_path)
    with open(results_path, 'a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        if not file_exists:
            wf.writerow(['ME', 'MAE', 'RMSE'])
        wf.writerow([me, mae, rmse])

    # Create DataFrame
    df_output = pd.DataFrame({
        'Image_name': name_list,
        'True_age': trues,
        'Predicted_age': preds
    })
    
    # Save to CSV
    output_csv_path = os.path.join(args.output_dir, args.task, f'Predictions_{mode}.csv')
    df_output.to_csv(output_csv_path, index=False)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, mae
