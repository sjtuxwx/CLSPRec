#!/bin/bash

# =============================================================================
# 参数敏感性分析脚本
# 分析参数: memory_size, aux_weight, lr
# 三个数据集并行运行，各数据集内部串行
# =============================================================================

# 获取可用GPU数量
get_gpu_count() {
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l
}

GPU_COUNT=$(get_gpu_count)
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "警告: 无法检测GPU数量，默认使用3张GPU"
    GPU_COUNT=3
fi
echo "检测到 $GPU_COUNT 张GPU"

# 数据集配置
DATASETS=("PHO" "NYC" "SIN")
# GPU分配: PHO->0, NYC->1, SIN->2，如果GPU不够则取模
declare -A DATASET_GPU
for i in "${!DATASETS[@]}"; do
    DATASET_GPU[${DATASETS[$i]}]=$((i % GPU_COUNT))
done

echo "GPU分配: PHO->cuda:${DATASET_GPU[PHO]}, NYC->cuda:${DATASET_GPU[NYC]}, SIN->cuda:${DATASET_GPU[SIN]}"

# 学习率配置
declare -A DATASET_LR
DATASET_LR[PHO]="1e-4"
DATASET_LR[NYC]="2e-4"
DATASET_LR[SIN]="2e-4"

# Embedding size配置
declare -A DATASET_EMBED
DATASET_EMBED[PHO]=60
DATASET_EMBED[NYC]=40
DATASET_EMBED[SIN]=60

# Epoch配置
declare -A DATASET_EPOCH
DATASET_EPOCH[PHO]=25
DATASET_EPOCH[NYC]=25
DATASET_EPOCH[SIN]=25

# =============================================================================
# 敏感性分析参数配置
# =============================================================================

# memory_size: 从大到小
MEMORY_SIZES=(200 100 50 30 10)

# aux_weight: 辅助损失权重
AUX_WEIGHTS=(0.0 0.1 0.3 0.5 0.7 1.0)

# learning rate: 学习率
LR_VALUES=("5e-5" "1e-4" "2e-4" "5e-4" "1e-3")

# =============================================================================
# 函数定义
# =============================================================================

# 创建工作区并生成settings.py
create_workspace_and_settings() {
    local city=$1
    local gpu_id=$2
    local exp_name=$3
    local memory_size=$4
    local aux_weight=$5
    local lr=$6
    local embed_size=$7
    local epoch=$8
    
    local workspace="workspace_sensitivity_${city}"
    
    # 创建工作区目录
    mkdir -p "$workspace"
    
    # 复制必要文件
    cp main.py "$workspace/"
    cp CLSPRec.py "$workspace/"
    cp data_preprocessor.py "$workspace/"
    cp -r utils "$workspace/" 2>/dev/null || true
    cp -r results "$workspace/" 2>/dev/null || true
    cp -r processed_data "$workspace/" 2>/dev/null || true
    
    # 创建数据和结果的软链接
    ln -sf "$(pwd)/data" "$workspace/data"
    mkdir -p "./sensitive/${city}"
    ln -sf "$(pwd)/sensitive" "$workspace/sensitive"
    
    # 生成settings.py
    cat > "$workspace/settings.py" << SETTINGS_EOF
task_name = 'sensitivity_${exp_name}'
city = '${city}'
gpuId = "cuda:${gpu_id}"

enable_random_mask = True
mask_prop = 0.1
enable_enhance_user = True
enable_ssl = True
enable_distance_sample = False
neg_sample_count = 5
neg_weight = 1

# Spatiotemporal-aware contrastive learning
enable_spatiotemporal_ssl = False
ssl_time_scale = 604800
ssl_spatial_scale = 50
ssl_temperature = 0.07
aux_weight = ${aux_weight}
user_enhance_mode = 'lhuc'
memory_size = ${memory_size}
use_rope = True
use_fused_rope_3d = True
fused_rope_max_time_diff = 1440
fused_rope_max_distance = 50
use_hstu = True
enable_cross_day_attention = False
enable_long_short_cross_attention = False

enable_dynamic_day_length = False
sample_day_length = 14

lr = ${lr}
epoch = ${epoch}
embed_size = ${embed_size}
run_times = 1

output_file_name = f'{task_name} {city}' + "_epoch" + str(epoch)

if enable_dynamic_day_length:
    output_file_name = output_file_name + f"_DynamicDay{sample_day_length}"
else:
    output_file_name = output_file_name + "_StaticDay7"

if enable_random_mask:
    output_file_name = output_file_name + "_" + "Mask"
else:
    output_file_name = output_file_name + "_" + "NoMask"

if enable_enhance_user:
    output_file_name = output_file_name + "_" + "Enhance"
else:
    output_file_name = output_file_name + "_" + "NoEnhance"

if enable_ssl:
    if enable_distance_sample:
        output_file_name = output_file_name + "_" + "SSL" + "_" + "DistanceNegCount" + str(neg_sample_count)
    else:
        output_file_name = output_file_name + "_" + "SSL" + "_" + "NegCount" + str(neg_sample_count)
else:
    output_file_name = output_file_name + "_" + "NoSSL"

if use_hstu:
    output_file_name = output_file_name + "_" + "HSTU"

if enable_spatiotemporal_ssl:
    output_file_name = output_file_name + "_" + "STSSL"

if use_rope:
    if use_fused_rope_3d:
        output_file_name = output_file_name + "_" + "FusedRoPE3D"
    else:
        output_file_name = output_file_name + "_" + "RoPE"

if user_enhance_mode == 'lhuc':
    output_file_name = output_file_name + "_" + "LHUC"
elif user_enhance_mode == 'memory':
    output_file_name = output_file_name + "_" + "Memory"

if enable_cross_day_attention:
    output_file_name = output_file_name + "_" + "CrossDay"

if enable_long_short_cross_attention:
    output_file_name = output_file_name + "_" + "LSCrossAttn"

output_file_name = output_file_name + '_embeddingSize' + str(embed_size)
SETTINGS_EOF

    echo "$workspace"
}

# 运行单个数据集的所有敏感性实验（串行）
run_dataset_sensitivity() {
    local city=$1
    local gpu_id=${DATASET_GPU[$city]}
    local base_lr=${DATASET_LR[$city]}
    local embed_size=${DATASET_EMBED[$city]}
    local epoch=${DATASET_EPOCH[$city]}
    
    echo "=========================================="
    echo "开始 ${city} 数据集敏感性分析 (GPU: cuda:${gpu_id})"
    echo "=========================================="
    
    local summary_file="./sensitive/${city}/sensitivity_summary.csv"
    mkdir -p "./sensitive/${city}"
    
    # 创建汇总CSV头
    echo "experiment_type,parameter_name,parameter_value,HR5_Max,HR10_Max,NDCG5_Max,NDCG10_Max,sum" > "$summary_file"
    
    # -----------------------------------------------------------------
    # 1. memory_size 敏感性分析 (从大到小)
    # -----------------------------------------------------------------
    echo ""
    echo ">>> [${city}] memory_size 敏感性分析"
    for mem_size in "${MEMORY_SIZES[@]}"; do
        local exp_name="memory_size_${mem_size}"
        local log_dir="./sensitive/${city}/${exp_name}"
        mkdir -p "$log_dir"
        
        echo "  运行: memory_size=${mem_size}"
        
        local workspace=$(create_workspace_and_settings "$city" "$gpu_id" "$exp_name" "$mem_size" "0.5" "$base_lr" "$embed_size" "$epoch")
        
        cd "$workspace"
        python -u main.py > "../sensitive/${city}/${exp_name}/train.log" 2>&1
        
        # 提取结果并追加到汇总
        if [ -f "../sensitive/${city}/${exp_name}/train.log" ]; then
            # 尝试从日志中提取最终结果
            local result_line=$(grep -E "^[0-9.]+," "../sensitive/${city}/${exp_name}/train.log" | tail -1)
            if [ -n "$result_line" ]; then
                echo "memory_size,memory_size,${mem_size},${result_line}" >> "../$summary_file"
            fi
        fi
        
        cd ..
        echo "  完成: memory_size=${mem_size}"
    done
    
    # -----------------------------------------------------------------
    # 2. aux_weight 敏感性分析
    # -----------------------------------------------------------------
    echo ""
    echo ">>> [${city}] aux_weight 敏感性分析"
    for aux_w in "${AUX_WEIGHTS[@]}"; do
        local exp_name="aux_weight_${aux_w}"
        local log_dir="./sensitive/${city}/${exp_name}"
        mkdir -p "$log_dir"
        
        echo "  运行: aux_weight=${aux_w}"
        
        local workspace=$(create_workspace_and_settings "$city" "$gpu_id" "$exp_name" "50" "$aux_w" "$base_lr" "$embed_size" "$epoch")
        
        cd "$workspace"
        python -u main.py > "../sensitive/${city}/${exp_name}/train.log" 2>&1
        
        # 提取结果并追加到汇总
        if [ -f "../sensitive/${city}/${exp_name}/train.log" ]; then
            local result_line=$(grep -E "^[0-9.]+," "../sensitive/${city}/${exp_name}/train.log" | tail -1)
            if [ -n "$result_line" ]; then
                echo "aux_weight,aux_weight,${aux_w},${result_line}" >> "../$summary_file"
            fi
        fi
        
        cd ..
        echo "  完成: aux_weight=${aux_w}"
    done
    
    # -----------------------------------------------------------------
    # 3. learning_rate 敏感性分析
    # -----------------------------------------------------------------
    echo ""
    echo ">>> [${city}] learning_rate 敏感性分析"
    for lr_val in "${LR_VALUES[@]}"; do
        local exp_name="lr_${lr_val}"
        local log_dir="./sensitive/${city}/${exp_name}"
        mkdir -p "$log_dir"
        
        echo "  运行: lr=${lr_val}"
        
        local workspace=$(create_workspace_and_settings "$city" "$gpu_id" "$exp_name" "50" "0.5" "$lr_val" "$embed_size" "$epoch")
        
        cd "$workspace"
        python -u main.py > "../sensitive/${city}/${exp_name}/train.log" 2>&1
        
        # 提取结果并追加到汇总
        if [ -f "../sensitive/${city}/${exp_name}/train.log" ]; then
            local result_line=$(grep -E "^[0-9.]+," "../sensitive/${city}/${exp_name}/train.log" | tail -1)
            if [ -n "$result_line" ]; then
                echo "learning_rate,lr,${lr_val},${result_line}" >> "../$summary_file"
            fi
        fi
        
        cd ..
        echo "  完成: lr=${lr_val}"
    done
    
    echo ""
    echo "=========================================="
    echo "${city} 敏感性分析完成!"
    echo "汇总结果: $summary_file"
    echo "=========================================="
}

# 清理工作区
clean_workspaces() {
    echo "清理敏感性分析工作区..."
    rm -rf workspace_sensitivity_*
    echo "清理完成"
}

# 停止所有实验
stop_experiments() {
    echo "正在查找并终止所有 python main.py 进程..."
    PIDS=$(ps aux | grep "python.*main.py" | grep -v grep | awk '{print $2}')
    if [ -z "$PIDS" ]; then
        echo "未找到 python main.py 进程。"
    else
        echo "找到以下进程: $PIDS"
        kill $PIDS 2>/dev/null
        sleep 2
        PIDS_AFTER=$(ps aux | grep "python.*main.py" | grep -v grep | awk '{print $2}')
        if [ -n "$PIDS_AFTER" ]; then
            echo "部分进程未能正常终止，强制终止: $PIDS_AFTER"
            kill -9 $PIDS_AFTER 2>/dev/null
        fi
        echo "所有相关进程已终止。"
    fi
    nvidia-smi
}

# 显示帮助
show_help() {
    echo "用法: ./run_sensitivity.sh [命令]"
    echo ""
    echo "命令:"
    echo "  parallel    - 三个数据集并行运行敏感性分析 (推荐)"
    echo "  PHO         - 只运行 PHO 数据集"
    echo "  NYC         - 只运行 NYC 数据集"
    echo "  SIN         - 只运行 SIN 数据集"
    echo "  clean       - 清理工作区"
    echo "  stop        - 停止所有实验进程"
    echo "  help        - 显示帮助"
    echo ""
    echo "敏感性分析参数:"
    echo "  memory_size: ${MEMORY_SIZES[*]} (从大到小)"
    echo "  aux_weight:  ${AUX_WEIGHTS[*]}"
    echo "  lr:          ${LR_VALUES[*]}"
    echo ""
    echo "结果输出目录: ./sensitive/<dataset>/"
    echo "汇总CSV文件:  ./sensitive/<dataset>/sensitivity_summary.csv"
}

# 生成最终汇总报告
generate_final_report() {
    echo ""
    echo "=========================================="
    echo "生成最终汇总报告"
    echo "=========================================="
    
    local final_report="./sensitive/final_sensitivity_report.csv"
    echo "dataset,experiment_type,parameter_name,parameter_value,HR5_Max,HR10_Max,NDCG5_Max,NDCG10_Max,sum" > "$final_report"
    
    for city in "${DATASETS[@]}"; do
        local summary="./sensitive/${city}/sensitivity_summary.csv"
        if [ -f "$summary" ]; then
            tail -n +2 "$summary" | while read line; do
                echo "${city},${line}" >> "$final_report"
            done
        fi
    done
    
    echo "最终报告已生成: $final_report"
}

# =============================================================================
# 主程序
# =============================================================================

case "$1" in
    "parallel")
        echo "=========================================="
        echo "开始三数据集并行敏感性分析"
        echo "=========================================="
        
        mkdir -p ./sensitive
        
        # 并行启动三个数据集
        run_dataset_sensitivity "PHO" &
        PID_PHO=$!
        
        run_dataset_sensitivity "NYC" &
        PID_NYC=$!
        
        run_dataset_sensitivity "SIN" &
        PID_SIN=$!
        
        echo "已启动后台任务:"
        echo "  PHO: PID=$PID_PHO (cuda:${DATASET_GPU[PHO]})"
        echo "  NYC: PID=$PID_NYC (cuda:${DATASET_GPU[NYC]})"
        echo "  SIN: PID=$PID_SIN (cuda:${DATASET_GPU[SIN]})"
        echo ""
        echo "等待所有实验完成..."
        
        # 等待所有后台任务完成
        wait $PID_PHO
        echo "PHO 完成"
        wait $PID_NYC
        echo "NYC 完成"
        wait $PID_SIN
        echo "SIN 完成"
        
        # 生成最终汇总报告
        generate_final_report
        
        echo ""
        echo "=========================================="
        echo "所有敏感性分析实验完成!"
        echo "=========================================="
        ;;
    
    "PHO"|"NYC"|"SIN")
        mkdir -p ./sensitive
        run_dataset_sensitivity "$1"
        ;;
    
    "clean")
        clean_workspaces
        ;;
    
    "stop")
        stop_experiments
        ;;
    
    "help"|"--help"|"-h"|"")
        show_help
        ;;
    
    *)
        echo "未知命令: $1"
        show_help
        exit 1
        ;;
esac
