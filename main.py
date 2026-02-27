import argparse
import datetime
import random
import numpy as np

import torch
import pickle
import time
import os
import pandas as pd

import settings
from CLSPRec import CLSPRec
from results.data_reader import print_output_to_file, calculate_average, clear_log_meta_model


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CLSPRec - POI Recommendation Model')
    
    # Checkpoint 相关参数
    parser.add_argument('--resume', type=str, default=None,
                        help='从指定的checkpoint路径恢复训练 (e.g., ./results/exp1/checkpoint_epoch_5.pt)')
    parser.add_argument('--save_checkpoint', action='store_true', default=True,
                        help='是否保存checkpoint (默认: True)')
    parser.add_argument('--no_save_checkpoint', action='store_false', dest='save_checkpoint',
                        help='禁用checkpoint保存')
    parser.add_argument('--checkpoint_freq', type=int, default=1,
                        help='每隔多少个epoch保存一次checkpoint (默认: 1)')
    parser.add_argument('--keep_last_k', type=int, default=3,
                        help='只保留最近k个checkpoint文件 (默认: 3, 设置为-1保留所有)')
    parser.add_argument('--save_best', action='store_true', default=True,
                        help='是否保存最佳模型checkpoint (默认: True)')
    
    # 覆盖 settings.py 中的参数
    parser.add_argument('--city', type=str, default=None,
                        help='覆盖settings中的city参数 (PHO, NYC, SIN)')
    parser.add_argument('--gpu', type=str, default=None,
                        help='覆盖settings中的GPU设置 (e.g., cuda:0)')
    parser.add_argument('--epoch', type=int, default=None,
                        help='覆盖settings中的epoch数量')
    parser.add_argument('--lr', type=float, default=None,
                        help='覆盖settings中的学习率')
    
    return parser.parse_args()

# Set random seed for reproducibility
def set_seed(seed=12345):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(12345)

device = settings.gpuId if torch.cuda.is_available() else 'cpu'
city = settings.city

if settings.enable_ssl and settings.enable_distance_sample:
    df_farthest_POIs = pd.read_csv(f"./raw_data/{city}_farthest_POIs.csv")


def generate_sample_to_device(sample):
    """
    Convert a sample to device tensors.
    Sample structure (when fused RoPE is enabled):
        seq[0:5]: POI, cat, user, hour, day
        seq[5]: date
        seq[6]: latitude
        seq[7]: longitude
        seq[8]: timestamp
    """
    sample_to_device = []
    use_fused_rope = getattr(settings, 'use_fused_rope_3d', False)
    
    
    if settings.enable_dynamic_day_length:
        last_day = sample[-1][5][0]
        for seq in sample:
            seq_day = seq[5][0]
            if last_day - seq_day < settings.sample_day_length:
                features = torch.tensor(seq[:5]).to(device)
                day_nums = torch.tensor(seq[5]).to(device)
                if use_fused_rope and len(seq) >= 9:
                    # Include spatiotemporal info: (features, day_nums, latitudes, longitudes, timestamps)
                    latitudes = torch.tensor(seq[6], dtype=torch.float32).to(device)
                    longitudes = torch.tensor(seq[7], dtype=torch.float32).to(device)
                    timestamps = torch.tensor(seq[8], dtype=torch.float32).to(device)
                    sample_to_device.append((features, day_nums, latitudes, longitudes, timestamps))
                else:
                    sample_to_device.append((features, day_nums))
    else:
        # #region agent log
        if not hasattr(generate_sample_to_device, '_sample_structure_debug'):
            import json, time
            seq_lengths = [len(seq) for seq in sample]
            log_data = {'location':'main.py:65','message':'sample结构检查','data':{'sample_len':len(sample),'seq_lengths':seq_lengths,'use_fused_rope':use_fused_rope},'timestamp':int(time.time()*1000),'hypothesisId':'F'}
            # with open('/data/xwx/code/CLSPRec/.cursor/debug.log','a') as f: f.write(json.dumps(log_data)+'\n')
            generate_sample_to_device._sample_structure_debug = True
        # #endregion
        
        for i, seq in enumerate(sample):
            features = torch.tensor(seq[:5]).to(device)
            day_nums = torch.tensor(seq[5]).to(device)
            # 检查数据中是否包含时空信息
            # 数据结构：seq = [[poi_seq], [cat_seq], [user_seq], [hour_seq], [day_seq], [date_seq], [lat_seq], [lon_seq], [ts_seq]]
            if use_fused_rope and len(seq) >= 9:
                # Include spatiotemporal info: (features, day_nums, latitudes, longitudes, timestamps)
                latitudes = torch.tensor(seq[6], dtype=torch.float32).to(device)
                longitudes = torch.tensor(seq[7], dtype=torch.float32).to(device)
                timestamps = torch.tensor(seq[8], dtype=torch.float32).to(device)
                sample_to_device.append((features, day_nums, latitudes, longitudes, timestamps))
            else:
                # #region agent log
                if not hasattr(generate_sample_to_device, '_missing_spatiotemporal'):
                    import json, time
                    log_data = {'location':'main.py:85','message':'序列缺少时空信息','data':{'seq_index':i,'seq_len':len(seq),'use_fused_rope':use_fused_rope,'total_seqs':len(sample)},'timestamp':int(time.time()*1000),'hypothesisId':'F'}
                    # with open('/data/xwx/code/CLSPRec/.cursor/debug.log','a') as f: f.write(json.dumps(log_data)+'\n')
                    generate_sample_to_device._missing_spatiotemporal = True
                # #endregion
                sample_to_device.append((features, day_nums))

    return sample_to_device


def generate_day_sample_to_device(day_trajectory):
    """
    Convert a single day trajectory to device tensors.
    """
    use_fused_rope = getattr(settings, 'use_fused_rope_3d', False)
    
    features = torch.tensor(day_trajectory[:5]).to(device)
    day_nums = torch.tensor(day_trajectory[5]).to(device)
    
    if use_fused_rope and len(day_trajectory) >= 9:
        latitudes = torch.tensor(day_trajectory[6], dtype=torch.float32).to(device)
        longitudes = torch.tensor(day_trajectory[7], dtype=torch.float32).to(device)
        timestamps = torch.tensor(day_trajectory[8], dtype=torch.float32).to(device)
        day_to_device = (features, day_nums, latitudes, longitudes, timestamps)
    else:
        day_to_device = (features, day_nums)
    
    return day_to_device


def generate_negative_sample_list(dataset, user_id, current_POI):
    k = settings.neg_sample_count
    neg_day_sample_to_device_list = []
    if settings.enable_distance_sample:
        # Random sample k negative samples from other users' trajectories
        # and the negative samples contain the farthest POIs from current POI
        farthest_POIs = set(df_farthest_POIs.iloc[current_POI].values.tolist())
        eligible_sequences = []
        for seq in dataset:
            if seq[0][2][0] != user_id:
                for i in range(len(seq)):
                    if set(seq[i][0]).intersection(farthest_POIs):
                        eligible_sequences.append(seq[i])

        if len(eligible_sequences) == 0:
            print(f'Can not find eligible_sequences, current POI {current_POI}')
        elif len(eligible_sequences) < k:
            neg_day_samples = eligible_sequences
        else:
            neg_day_samples = random.sample(eligible_sequences, k)
    else:
        # Random sample k negative samples from other users' trajectories
        neg_day_samples = random.sample([seq[-1] for seq in dataset if seq[0][2][0] != user_id], k)

    for neg_day_sample in neg_day_samples:
        neg_day_sample_to_device_list.append(generate_day_sample_to_device(neg_day_sample))
    return neg_day_sample_to_device_list


def save_checkpoint(checkpoint_path, model, optimizer, epoch, h_params, loss_dict, 
                    recalls, ndcgs, maps, best_metric=None):
    """
    保存完整的checkpoint，包含所有训练状态
    
    Args:
        checkpoint_path: checkpoint保存路径
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        h_params: 超参数
        loss_dict: 损失历史
        recalls: recall指标历史
        ndcgs: ndcg指标历史
        maps: map指标历史
        best_metric: 最佳指标值（用于best checkpoint）
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'h_params': h_params,
        'loss_dict': loss_dict,
        'recalls': recalls,
        'ndcgs': ndcgs,
        'maps': maps,
        'best_metric': best_metric,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'city': settings.city,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"[Checkpoint] 已保存 checkpoint 到: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """
    从checkpoint恢复训练状态
    
    Args:
        checkpoint_path: checkpoint文件路径
        model: 模型（已初始化）
        optimizer: 优化器（已初始化）
        device: 设备
    
    Returns:
        start_epoch: 恢复后的起始epoch
        loss_dict: 损失历史
        recalls: recall指标历史
        ndcgs: ndcg指标历史
        maps: map指标历史
        best_metric: 最佳指标值
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
    
    print(f"[Checkpoint] 正在从 {checkpoint_path} 加载checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    loss_dict = checkpoint.get('loss_dict', {})
    recalls = checkpoint.get('recalls', {})
    ndcgs = checkpoint.get('ndcgs', {})
    maps = checkpoint.get('maps', {})
    best_metric = checkpoint.get('best_metric', 0.0)
    
    print(f"[Checkpoint] 成功恢复训练状态:")
    print(f"  - 保存时间: {checkpoint.get('timestamp', 'N/A')}")
    print(f"  - 保存时的city: {checkpoint.get('city', 'N/A')}")
    print(f"  - 已完成epoch: {checkpoint['epoch']}")
    print(f"  - 将从epoch {start_epoch} 继续训练")
    if best_metric:
        print(f"  - 当前最佳指标: {best_metric:.6f}")
    
    return start_epoch, loss_dict, recalls, ndcgs, maps, best_metric


def cleanup_old_checkpoints(exp_dir, run_name, keep_last_k):
    """
    清理旧的checkpoint文件，只保留最近的k个
    
    Args:
        exp_dir: 实验目录
        run_name: 运行名称
        keep_last_k: 保留最近k个checkpoint（-1表示保留所有）
    """
    if keep_last_k < 0:
        return
    
    import glob
    pattern = os.path.join(exp_dir, f"{run_name}_checkpoint_epoch_*.pt")
    checkpoint_files = glob.glob(pattern)
    
    # 按epoch排序
    def get_epoch(filepath):
        basename = os.path.basename(filepath)
        try:
            epoch_str = basename.split('_epoch_')[1].replace('.pt', '')
            return int(epoch_str)
        except:
            return -1
    
    checkpoint_files.sort(key=get_epoch)
    
    # 删除旧的checkpoint
    if len(checkpoint_files) > keep_last_k:
        files_to_remove = checkpoint_files[:-keep_last_k]
        for f in files_to_remove:
            try:
                os.remove(f)
                print(f"[Checkpoint] 已删除旧checkpoint: {os.path.basename(f)}")
            except OSError:
                pass


def train_model(train_set, test_set, h_params, vocab_size, device, run_name, exp_dir, args):
    torch.cuda.empty_cache()
    model_path = f"{exp_dir}/{run_name}_model"
    log_path = f"{exp_dir}/{run_name}_log"
    meta_path = f"{exp_dir}/{run_name}_meta"

    print("parameters:", h_params)

    if os.path.isfile(model_path) and not args.resume:
        try:
            os.remove(meta_path)
            os.remove(model_path)
            os.remove(log_path)
        except OSError:
            pass
    
    if not args.resume:
        file = open(log_path, 'wb')
        pickle.dump(h_params, file)
        file.close()

    # construct model
    use_fused_rope = getattr(settings, 'use_fused_rope_3d', False)
    rec_model = CLSPRec(
        vocab_size=vocab_size,
        f_embed_size=h_params['embed_size'],
        num_encoder_layers=h_params['tfp_layer_num'],
        num_lstm_layers=h_params['lstm_layer_num'],
        num_heads=h_params['head_num'],
        forward_expansion=h_params['expansion'],
        dropout_p=h_params['dropout'],
        use_hstu=settings.use_hstu,
        use_fused_rope_3d=use_fused_rope,
        max_seq_len=h_params['max_seq_len'],
        enable_cross_day_attention=settings.enable_cross_day_attention,
        enable_long_short_cross_attention=settings.enable_long_short_cross_attention
    )

    rec_model = rec_model.to(device)

    params = list(rec_model.parameters())
    optimizer = torch.optim.Adam(params, lr=h_params['lr'])

    # 初始化训练状态
    start_epoch = 0
    loss_dict, recalls, ndcgs, maps = {}, {}, {}, {}
    best_metric = 0.0
    
    # 从checkpoint恢复训练
    if args.resume:
        start_epoch, loss_dict, recalls, ndcgs, maps, best_metric = load_checkpoint(
            args.resume, rec_model, optimizer, device
        )
        rec_model.train()
    elif os.path.isfile(model_path):
        # 兼容旧的恢复逻辑
        rec_model.load_state_dict(torch.load(model_path))
        rec_model.train()
        if os.path.isfile(meta_path):
            meta_file = open(meta_path, "rb")
            start_epoch = pickle.load(meta_file) + 1
            meta_file.close()

    for epoch in range(start_epoch, h_params['epoch']):
        begin_time = time.time()
        total_loss = 0.
        rec_model.train()
        
        for sample in train_set:
            sample_to_device = generate_sample_to_device(sample)

            neg_sample_to_device_list = []
            if settings.enable_ssl:
                user_id = sample[0][2][0]
                current_POI = sample[-1][0][-2]
                neg_sample_to_device_list = generate_negative_sample_list(train_set, user_id, current_POI)

            loss, _ = rec_model(sample_to_device, neg_sample_to_device_list)
            total_loss += loss.detach().cpu()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test
        recall, ndcg, map = test_model(test_set, rec_model)
        recalls[epoch] = recall
        ndcgs[epoch] = ndcg
        maps[epoch] = map

        # Record avg loss
        avg_loss = total_loss / len(train_set)
        loss_dict[epoch] = avg_loss
        time_taken = int(time.time() - begin_time)
        print(f"epoch: {epoch}; average loss: {avg_loss}, time taken: {time_taken}s")
        
        # 使用 NDCG@10 作为评价标准，只保存最佳模型
        current_metric = ndcg[10].item()
        if args.save_checkpoint and current_metric > best_metric:
            best_metric = current_metric
            best_checkpoint_path = f"{exp_dir}/{run_name}_best.pt"
            save_checkpoint(
                best_checkpoint_path, rec_model, optimizer, epoch, h_params,
                loss_dict, recalls, ndcgs, maps, best_metric
            )
            print(f"[Checkpoint] 新的最佳模型! NDCG@10: {best_metric:.6f}")

        
        # 兼容旧的保存逻辑
        torch.save(rec_model.state_dict(), model_path)
        meta_file = open(meta_path, 'wb')
        pickle.dump(epoch, meta_file)
        meta_file.close()

        # Early stop
        past_10_loss = list(loss_dict.values())[-11:-1]
        if len(past_10_loss) > 10 and abs(total_loss - np.mean(past_10_loss)) < h_params['loss_delta']:
            print(f"***Early stop at epoch {epoch}***")
            break

        file = open(log_path, 'wb')
        pickle.dump(loss_dict, file)
        pickle.dump(recalls, file)
        pickle.dump(ndcgs, file)
        pickle.dump(maps, file)
        file.close()


    print("============================")


def test_model(test_set, rec_model, ks=[1, 5, 10]):
    def calc_recall(labels, preds, k):
        return torch.sum(torch.sum(labels == preds[:, :k], dim=1)) / labels.shape[0]

    def calc_ndcg(labels, preds, k):
        exist_pos = (preds[:, :k] == labels).nonzero()[:, 1] + 1
        ndcg = 1 / torch.log2(exist_pos + 1)
        return torch.sum(ndcg) / labels.shape[0]

    def calc_map(labels, preds, k):
        exist_pos = (preds[:, :k] == labels).nonzero()[:, 1] + 1
        map = 1 / exist_pos
        return torch.sum(map) / labels.shape[0]

    preds, labels = [], []
    for sample in test_set:
        sample_to_device = generate_sample_to_device(sample)

        neg_sample_to_device_list = []
        if settings.enable_ssl:
            user_id = sample[0][2][0]
            current_POI = sample[-1][0][-2]
            neg_sample_to_device_list = generate_negative_sample_list(test_set, user_id, current_POI)

        pred, label = rec_model.predict(sample_to_device, neg_sample_to_device_list)
        preds.append(pred.detach())
        labels.append(label.detach())
    preds = torch.stack(preds, dim=0)
    labels = torch.unsqueeze(torch.stack(labels, dim=0), 1)

    recalls, NDCGs, MAPs = {}, {}, {}
    for k in ks:
        recalls[k] = calc_recall(labels, preds, k)
        NDCGs[k] = calc_ndcg(labels, preds, k)
        MAPs[k] = calc_map(labels, preds, k)
        print(f"Recall @{k} : {recalls[k]},\tNDCG@{k} : {NDCGs[k]},\tMAP@{k} : {MAPs[k]}")

    return recalls, NDCGs, MAPs


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    
    # 覆盖settings中的参数（如果命令行指定了）
    if args.city:
        settings.city = args.city
    if args.gpu:
        settings.gpuId = args.gpu
    if args.epoch:
        settings.epoch = args.epoch
    if args.lr:
        settings.lr = args.lr
    
    # 重新设置device和city（因为可能被命令行参数覆盖）
    device = settings.gpuId if torch.cuda.is_available() else 'cpu'
    city = settings.city
    
    # 如果启用了距离采样，重新加载数据
    if settings.enable_ssl and settings.enable_distance_sample:
        df_farthest_POIs = pd.read_csv(f"./raw_data/{city}_farthest_POIs.csv")
    
    # Get current time
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 60)
    print(f"CLSPRec 训练开始")
    print(f"时间: {now_str}")
    print("=" * 60)
    
    # 打印checkpoint相关配置
    print("\n[Checkpoint 配置]")
    print(f"  - 保存checkpoint: {args.save_checkpoint}")
    print(f"  - 保存频率: 每 {args.checkpoint_freq} 个epoch")
    print(f"  - 保留最近: {args.keep_last_k} 个checkpoint (-1表示全部保留)")
    print(f"  - 保存最佳模型: {args.save_best}")
    if args.resume:
        print(f"  - 从checkpoint恢复: {args.resume}")
    print()

    # Get parameters
    h_params = {
        'expansion': 4,
        'random_mask': settings.enable_random_mask,
        'mask_prop': 0.1,
        'lr': settings.lr,
        'epoch': settings.epoch,
        'loss_delta': 1e-3}

    # 使用根目录的数据（包含时空信息：lat, lon, timestamp）
    processed_data_directory = './processed_data'

    # Read training data
    file = open(f"{processed_data_directory}/{city}_train", 'rb')
    train_set = pickle.load(file)
    file = open(f"{processed_data_directory}/{city}_valid", 'rb')
    valid_set = pickle.load(file)

    # Read meta data
    file = open(f"{processed_data_directory}/{city}_meta", 'rb')
    meta = pickle.load(file)
    file.close()

    vocab_size = {"POI": torch.tensor(len(meta["POI"])).to(device),
                  "cat": torch.tensor(len(meta["cat"])).to(device),
                  "user": torch.tensor(len(meta["user"])).to(device),
                  "hour": torch.tensor(len(meta["hour"])).to(device),
                  "day": torch.tensor(len(meta["day"])).to(device)}

    # Adjust specific parameters for each city
    if city == 'SIN':
        h_params['embed_size'] = settings.embed_size
        h_params['tfp_layer_num'] = 1
        h_params['lstm_layer_num'] = 3
        h_params['dropout'] = 0.2
        h_params['head_num'] = 1
        h_params['max_seq_len'] = 200  # SIN数据集需要更大的序列长度
    elif city == 'NYC':
        h_params['embed_size'] = settings.embed_size
        h_params['tfp_layer_num'] = 1
        h_params['lstm_layer_num'] = 2
        h_params['dropout'] = 0.1
        h_params['head_num'] = 1
        h_params['max_seq_len'] = 200  # NYC数据集需要更大的序列长度
    elif city == 'PHO':
        h_params['embed_size'] = settings.embed_size
        h_params['tfp_layer_num'] = 4
        h_params['lstm_layer_num'] = 2
        h_params['dropout'] = 0.2
        h_params['head_num'] = 1
        h_params['max_seq_len'] = 100  # PHO数据集序列长度较短

    # Create output folder
    if not os.path.isdir('./results'):
        os.mkdir("./results")
    
    # 如果从checkpoint恢复，尝试使用原来的实验目录
    exp_dir = None
    if args.resume:
        # 从checkpoint路径推断实验目录
        resume_dir = os.path.dirname(args.resume)
        if os.path.isdir(resume_dir):
            exp_dir = resume_dir
            print(f"[Checkpoint] 使用原实验目录: {exp_dir}")
    
    if exp_dir is None:
        # Create experiment directory with auto-increment if exists
        base_exp_dir = f"./results/{settings.output_file_name}"
        exp_dir = base_exp_dir
        counter = 1
        
        # If directory exists, add increment number (-1, -2, -3, ...)
        while os.path.isdir(exp_dir):
            exp_dir = f"{base_exp_dir}-{counter}"
            counter += 1
        
        os.mkdir(exp_dir)
        print(f"Created experiment directory: {exp_dir}")
    
    # 保存实验配置到日志文件
    config_log_path = f"{exp_dir}/experiment_config.txt"
    with open(config_log_path, 'w') as f:
        f.write(f"实验配置日志\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"开始时间: {now_str}\n")
        f.write(f"城市: {city}\n")
        f.write(f"GPU: {settings.gpuId}\n")
        f.write(f"\n[Checkpoint 配置]\n")
        f.write(f"  保存checkpoint: {args.save_checkpoint}\n")
        f.write(f"  保存频率: 每 {args.checkpoint_freq} 个epoch\n")
        f.write(f"  保留最近: {args.keep_last_k} 个checkpoint\n")
        f.write(f"  保存最佳模型: {args.save_best}\n")
        if args.resume:
            f.write(f"  从checkpoint恢复: {args.resume}\n")
        f.write(f"\n[超参数]\n")
        for k, v in h_params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\n[Settings配置]\n")
        for attr in dir(settings):
            if not attr.startswith('_'):
                f.write(f"  {attr}: {getattr(settings, attr)}\n")
    print(f"实验配置已保存到: {config_log_path}")

    print(f'\nCurrent GPU {settings.gpuId}')
    for run_num in range(1, 1 + settings.run_times):
        run_name = f'{settings.output_file_name} {run_num}'
        print(f"\n{'=' * 60}")
        print(f"开始训练: {run_name}")
        print(f"{'=' * 60}")

        train_model(train_set, valid_set, h_params, vocab_size, device, 
                   run_name=run_name, exp_dir=exp_dir, args=args)
        print_output_to_file(settings.output_file_name, run_num, exp_dir)

        t = random.randint(1, 9)
        print(f"sleep {t} seconds")
        time.sleep(t)

        clear_log_meta_model(settings.output_file_name, run_num, exp_dir)
    
    calculate_average(settings.output_file_name, settings.run_times, exp_dir)
    
    # 训练完成后更新配置日志
    end_time = datetime.datetime.now()
    with open(config_log_path, 'a') as f:
        f.write(f"\n{'=' * 50}\n")
        f.write(f"训练完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {end_time - now}\n")
    
    print(f"\n{'=' * 60}")
    print(f"所有训练完成!")
    print(f"实验目录: {exp_dir}")
    print(f"总耗时: {end_time - now}")
    print(f"{'=' * 60}")
