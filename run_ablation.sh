#!/bin/bash

# 消融实验脚本
# 三个数据集分别在三张卡上并行运行
# PHO -> cuda:0
# NYC -> cuda:1
# SIN -> cuda:2

# 每个数据集使用独立的工作目录，避免settings.py冲突

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 固定epoch
EPOCH=25

# 创建独立工作目录并运行实验
setup_and_run_dataset() {
    local city=$1
    local gpu=$2
    local embed_size=$3
    local lr=$4
    
    # 创建独立工作目录
    local work_dir="${SCRIPT_DIR}/workspace_${city}"
    mkdir -p "${work_dir}"
    
    # 复制必要文件到工作目录
    cp -f "${SCRIPT_DIR}/main.py" "${work_dir}/"
    cp -f "${SCRIPT_DIR}/CLSPRec.py" "${work_dir}/"
    cp -f "${SCRIPT_DIR}/data_preprocessor.py" "${work_dir}/"
    cp -f "${SCRIPT_DIR}/offline_training.py" "${work_dir}/" 2>/dev/null || true
    
    # 创建软链接到数据目录
    ln -sfn "${SCRIPT_DIR}/processed_data" "${work_dir}/processed_data"
    ln -sfn "${SCRIPT_DIR}/raw_data" "${work_dir}/raw_data"
    
    # 创建结果目录
    mkdir -p "${SCRIPT_DIR}/results/${city}"
    ln -sfn "${SCRIPT_DIR}/results" "${work_dir}/results"
    
    # 切换到工作目录
    cd "${work_dir}"
    
    echo "[${city}] 工作目录: ${work_dir}"
    echo "[${city}] ############################################"
    echo "[${city}] 开始消融实验 (GPU: ${gpu})"
    echo "[${city}] ############################################"
    
    # 实验配置数组: ablation_name user_enhance_mode use_rope use_fused_rope_3d use_hstu
    local experiments=(
        "full_model lhuc True True True"
        "ablation_user_enhance_memory memory True True True"
        "ablation_user_enhance_original original True True True"
        "ablation_no_rope lhuc False False True"
        "ablation_no_fused_rope_3d lhuc True False True"
        "ablation_no_rope_no_fused lhuc False False True"
        "ablation_no_hstu lhuc True True False"
    )
    
    for exp in "${experiments[@]}"; do
        read -r ablation_name user_enhance_mode use_rope use_fused_rope_3d use_hstu <<< "$exp"
        
        echo "[${city}] =========================================="
        echo "[${city}] Running: ${ablation_name}"
        echo "[${city}]   user_enhance_mode=${user_enhance_mode}"
        echo "[${city}]   use_rope=${use_rope}"
        echo "[${city}]   use_fused_rope_3d=${use_fused_rope_3d}"
        echo "[${city}]   use_hstu=${use_hstu}"
        echo "[${city}] =========================================="
        
        # 创建结果目录
        local result_dir="${SCRIPT_DIR}/results/${city}/${ablation_name}"
        mkdir -p "${result_dir}"
        
        # 在独立工作目录生成settings.py（不会冲突）
        cat > "${work_dir}/settings.py" << EOF
task_name = 'ablation'
city = '${city}'
gpuId = "${gpu}"

enable_random_mask = True
mask_prop = 0.1
enable_enhance_user = True
enable_ssl = True
enable_distance_sample = False
neg_sample_count = 5
neg_weight = 1

enable_spatiotemporal_ssl = False
ssl_time_scale = 604800
ssl_spatial_scale = 50
ssl_temperature = 0.07
aux_weight = 0.5
user_enhance_mode = '${user_enhance_mode}'
memory_size = 50
use_rope = ${use_rope}
use_fused_rope_3d = ${use_fused_rope_3d}
fused_rope_max_time_diff = 1440
fused_rope_max_distance = 50
use_hstu = ${use_hstu}
enable_cross_day_attention = False
enable_long_short_cross_attention = False

enable_dynamic_day_length = False
sample_day_length = 14

lr = ${lr}
epoch = ${EPOCH}
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
EOF

        # 在工作目录运行实验（使用 -u 禁用Python输出缓冲）
        cd "${work_dir}"
        python -u main.py 2>&1 | tee "${result_dir}/train.log"
        
        echo "[${city}] Finished: ${ablation_name}"
        echo ""
    done
    
    echo "[${city}] ############################################"
    echo "[${city}] 所有消融实验完成！"
    echo "[${city}] ############################################"
    
    # 清理工作目录（可选，保留用于调试）
    # rm -rf "${work_dir}"
}

# 显示使用方法
show_usage() {
    echo "使用方法:"
    echo "  $0 parallel - 三个数据集并行运行（推荐）"
    echo "  $0 PHO      - 只运行PHO数据集 (cuda:0)"
    echo "  $0 NYC      - 只运行NYC数据集 (cuda:1)"
    echo "  $0 SIN      - 只运行SIN数据集 (cuda:2)"
    echo "  $0 all      - 串行运行所有数据集"
    echo "  $0 clean    - 清理工作目录"
    echo "  $0 stop     - 停止所有正在运行的实验"
    echo ""
    echo "推荐使用方式（三卡并行）:"
    echo "  $0 parallel"
}

# 清理工作目录
clean_workspaces() {
    echo "清理工作目录..."
    rm -rf "${SCRIPT_DIR}/workspace_PHO"
    rm -rf "${SCRIPT_DIR}/workspace_NYC"
    rm -rf "${SCRIPT_DIR}/workspace_SIN"
    echo "清理完成"
}

# 主程序
case "${1}" in
    "PHO")
        setup_and_run_dataset "PHO" "cuda:0" 60 "1e-4"
        ;;
    "NYC")
        setup_and_run_dataset "NYC" "cuda:1" 40 "2e-4"
        ;;
    "SIN")
        setup_and_run_dataset "SIN" "cuda:2" 60 "2e-4"
        ;;
    "all")
        echo "============================================"
        echo "串行运行所有数据集消融实验"
        echo "============================================"
        setup_and_run_dataset "PHO" "cuda:0" 60 "1e-4"
        setup_and_run_dataset "NYC" "cuda:1" 40 "2e-4"
        setup_and_run_dataset "SIN" "cuda:2" 60 "2e-4"
        echo "所有实验完成！"
        ;;
    "parallel")
        echo "============================================"
        echo "并行运行三个数据集消融实验"
        echo "PHO -> cuda:0 (workspace_PHO/)"
        echo "NYC -> cuda:1 (workspace_NYC/)"
        echo "SIN -> cuda:2 (workspace_SIN/)"
        echo "Epoch: ${EPOCH}"
        echo "============================================"
        echo ""
        echo "每个数据集使用独立工作目录，避免settings.py冲突"
        echo ""
        
        # 创建结果目录
        mkdir -p "${SCRIPT_DIR}/results/PHO" "${SCRIPT_DIR}/results/NYC" "${SCRIPT_DIR}/results/SIN"
        
        # 并行启动三个数据集（每个在独立子shell中运行）
        (setup_and_run_dataset "PHO" "cuda:0" 60 "1e-4") > "${SCRIPT_DIR}/results/PHO/ablation_all.log" 2>&1 &
        PID_PHO=$!
        
        (setup_and_run_dataset "NYC" "cuda:1" 40 "2e-4") > "${SCRIPT_DIR}/results/NYC/ablation_all.log" 2>&1 &
        PID_NYC=$!
        
        (setup_and_run_dataset "SIN" "cuda:2" 60 "2e-4") > "${SCRIPT_DIR}/results/SIN/ablation_all.log" 2>&1 &
        PID_SIN=$!
        
        echo "三个数据集已启动并行运行:"
        echo "  PHO PID: ${PID_PHO} (日志: results/PHO/ablation_all.log)"
        echo "  NYC PID: ${PID_NYC} (日志: results/NYC/ablation_all.log)"
        echo "  SIN PID: ${PID_SIN} (日志: results/SIN/ablation_all.log)"
        echo ""
        echo "查看进度:"
        echo "  tail -f results/PHO/ablation_all.log"
        echo "  tail -f results/NYC/ablation_all.log"
        echo "  tail -f results/SIN/ablation_all.log"
        echo ""
        echo "等待所有实验完成..."
        
        wait ${PID_PHO}
        echo "✓ PHO 实验完成"
        
        wait ${PID_NYC}
        echo "✓ NYC 实验完成"
        
        wait ${PID_SIN}
        echo "✓ SIN 实验完成"
        
        echo ""
        echo "============================================"
        echo "所有消融实验完成！"
        echo "结果保存在 results/<数据集>/<消融名称>/ 目录下"
        echo ""
        echo "清理工作目录请运行: $0 clean"
        echo "============================================"
        ;;
    "clean")
        clean_workspaces
        ;;
    "stop")
        echo "正在停止所有消融实验进程..."
        
        # 杀掉所有 main.py 相关进程
        PIDS=$(pgrep -f "python.*main.py" 2>/dev/null)
        if [ -n "$PIDS" ]; then
            echo "找到以下进程:"
            ps -p $PIDS -o pid,start,cmd 2>/dev/null
            echo ""
            echo "正在终止..."
            pkill -f "python.*main.py"
            sleep 2
            
            # 检查是否还有残留
            REMAINING=$(pgrep -f "python.*main.py" 2>/dev/null)
            if [ -n "$REMAINING" ]; then
                echo "强制终止残留进程..."
                pkill -9 -f "python.*main.py"
            fi
            echo "✓ 所有实验进程已停止"
        else
            echo "没有找到正在运行的实验进程"
        fi
        
        echo ""
        echo "GPU状态:"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || echo "(nvidia-smi 不可用)"
        ;;
    *)
        show_usage
        ;;
esac
