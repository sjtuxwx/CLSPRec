#!/bin/bash

# ============================================================
# 粗粒度消融实验脚本
# ============================================================
# 数据集分配:
#   PHO -> cuda:0  (epoch=25, embed_size=60)
#   NYC -> cuda:1  (epoch=10, embed_size=40)
#   SIN -> cuda:2  (epoch=10, embed_size=60)
# 学习率统一: 1e-4
# 每个消融实验跑 3 次，取平均
# 结果输出到 ablation/ 目录
# ============================================================
#
# 消融方案（基准 = full_model: HSTU + FusedRoPE3D + LHUC）:
#   1. full_model        — 完整模型
#   2. wo_FusedRoPE3D    — 去掉 Fused 3D RoPE（use_rope=False, use_fused_rope_3d=False）
#   3. wo_HSTU           — 去掉 HSTU 注意力机制
#   4. enhance_memory    — 用户增强替换为 Memory Network
#   5. enhance_original  — 用户增强替换为 Original（线性插值）
#   6. enhance_none      — 去掉用户增强
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ABLATION_DIR="${SCRIPT_DIR}/ablation"
mkdir -p "${ABLATION_DIR}"

RUN_TIMES=3
LR="1e-4"

# ============================================================
# 消融实验列表
# 格式: "实验名 use_hstu use_rope use_fused_rope_3d user_enhance_mode"
# ============================================================
EXPERIMENTS=(
    "full_model          True  True  True  lhuc"
    "wo_FusedRoPE3D      True  False False lhuc"
    "wo_HSTU             False True  True  lhuc"
    "enhance_memory      True  True  True  memory"
    "enhance_original    True  True  True  original"
    "enhance_none        True  True  True  None"
)

# ============================================================
# 核心函数：为指定数据集执行所有消融实验
# ============================================================
setup_and_run_dataset() {
    local city=$1
    local gpu=$2
    local embed_size=$3
    local epoch=$4

    local work_dir="${SCRIPT_DIR}/workspace_ablation_${city}"
    mkdir -p "${work_dir}"

    # 复制必要源文件到独立工作目录（避免 settings.py 冲突）
    cp -f "${SCRIPT_DIR}/main.py"              "${work_dir}/"
    cp -f "${SCRIPT_DIR}/CLSPRec.py"           "${work_dir}/"
    cp -f "${SCRIPT_DIR}/data_preprocessor.py" "${work_dir}/" 2>/dev/null || true

    # 数据目录软链接
    ln -sfn "${SCRIPT_DIR}/processed_data" "${work_dir}/processed_data"
    ln -sfn "${SCRIPT_DIR}/raw_data"       "${work_dir}/raw_data"

    # results/ 子目录（main.py 需要 template.csv 和 data_reader.py）
    mkdir -p "${work_dir}/results"
    cp -f "${SCRIPT_DIR}/results/template.csv"   "${work_dir}/results/"
    cp -f "${SCRIPT_DIR}/results/data_reader.py" "${work_dir}/results/"
    # 如果有 __init__.py 也一并复制
    cp -f "${SCRIPT_DIR}/results/__init__.py"    "${work_dir}/results/" 2>/dev/null || true

    # 城市专属消融结果目录
    local city_dir="${ABLATION_DIR}/${city}"
    mkdir -p "${city_dir}"

    echo ""
    echo "[${city}] ########################################################"
    echo "[${city}] 开始粗粒度消融实验  GPU=${gpu}  epoch=${epoch}  lr=${LR}"
    echo "[${city}] ########################################################"

    for exp in "${EXPERIMENTS[@]}"; do
        # 解析参数
        read -r exp_name use_hstu use_rope use_fused_rope_3d user_enhance_mode <<< "$exp"

        echo ""
        echo "[${city}] ======================================================"
        echo "[${city}] 实验: ${exp_name}"
        echo "[${city}]   use_hstu          = ${use_hstu}"
        echo "[${city}]   use_rope          = ${use_rope}"
        echo "[${city}]   use_fused_rope_3d = ${use_fused_rope_3d}"
        echo "[${city}]   user_enhance_mode = ${user_enhance_mode}"
        echo "[${city}] ======================================================"

        # 清理上一轮在 workspace 内生成的结果子目录
        find "${work_dir}/results" -maxdepth 1 -mindepth 1 -type d -exec rm -rf {} + 2>/dev/null

        # ----------------------------------------------------------
        # 动态生成 settings.py
        # ----------------------------------------------------------
        cat > "${work_dir}/settings.py" << SETTINGSEOF
task_name = '${exp_name}'
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

lr = ${LR}
epoch = ${epoch}
embed_size = ${embed_size}
run_times = ${RUN_TIMES}

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
SETTINGSEOF

        # ----------------------------------------------------------
        # 运行实验
        # ----------------------------------------------------------
        local exp_out_dir="${city_dir}/${exp_name}"
        mkdir -p "${exp_out_dir}"

        cd "${work_dir}"
        python -u main.py 2>&1 | tee "${exp_out_dir}/train.log"

        # 将 workspace 内生成的结果复制到 ablation/ 目录
        local ws_result_dirs=$(find "${work_dir}/results" -maxdepth 1 -mindepth 1 -type d 2>/dev/null)
        for rd in ${ws_result_dirs}; do
            cp -r "${rd}"/* "${exp_out_dir}/" 2>/dev/null
        done

        echo "[${city}] ✓ 完成: ${exp_name}"
    done

    echo ""
    echo "[${city}] ########################################################"
    echo "[${city}] 所有消融实验完成！结果在 ablation/${city}/"
    echo "[${city}] ########################################################"
}

# ============================================================
# 汇总函数：将所有结果收集到一个 CSV
# ============================================================
generate_summary() {
    echo ""
    echo "============================================================"
    echo "汇总所有消融实验结果到 ablation/ablation_summary.csv ..."
    echo "============================================================"

    python3 - "${ABLATION_DIR}" << 'PYEOF'
import os, sys, csv, glob

ablation_dir = sys.argv[1]
summary_path = os.path.join(ablation_dir, 'ablation_summary.csv')

header = ['experiment', 'city', 'run', 'HR5_Max', 'HR10_Max', 'NDCG5_Max', 'NDCG10_Max', 'sum']
rows = []

cities = ['PHO', 'NYC', 'SIN']
exp_names = [
    'full_model', 'wo_FusedRoPE3D', 'wo_HSTU',
    'enhance_memory', 'enhance_original', 'enhance_none'
]

for city in cities:
    for exp_name in exp_names:
        exp_dir = os.path.join(ablation_dir, city, exp_name)
        if not os.path.isdir(exp_dir):
            continue

        csv_files = glob.glob(os.path.join(exp_dir, '*.csv'))
        for csv_file in csv_files:
            try:
                with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    csv_header = next(reader)          # 跳过表头
                    data_rows = []
                    for row in reader:
                        row = [c.strip() for c in row]
                        if len(row) < 5 or not row[0]:
                            continue
                        try:
                            vals = [float(row[i]) for i in range(5)]
                            name = row[-1] if len(row) > 5 else ''
                            data_rows.append((*vals, name))
                        except (ValueError, IndexError):
                            continue

                    if not data_rows:
                        continue

                    # 前 N-1 行为单次运行，最后一行为平均
                    for i, (hr5, hr10, ndcg5, ndcg10, s, name) in enumerate(data_rows[:-1]):
                        rows.append([
                            exp_name, city, f'run_{i+1}',
                            f'{hr5:.6f}', f'{hr10:.6f}',
                            f'{ndcg5:.6f}', f'{ndcg10:.6f}', f'{s:.6f}'
                        ])
                    # 平均行
                    hr5, hr10, ndcg5, ndcg10, s, name = data_rows[-1]
                    rows.append([
                        exp_name, city, 'average',
                        f'{hr5:.6f}', f'{hr10:.6f}',
                        f'{ndcg5:.6f}', f'{ndcg10:.6f}', f'{s:.6f}'
                    ])
            except Exception as e:
                print(f"[WARN] 读取 {csv_file} 出错: {e}")

with open(summary_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)

print(f"汇总完成: {summary_path}  ({len(rows)} 条记录)")
PYEOF
}

# ============================================================
# 清理工作目录
# ============================================================
clean_workspaces() {
    echo "清理消融实验工作目录..."
    rm -rf "${SCRIPT_DIR}/workspace_ablation_PHO"
    rm -rf "${SCRIPT_DIR}/workspace_ablation_NYC"
    rm -rf "${SCRIPT_DIR}/workspace_ablation_SIN"
    echo "✓ 清理完成"
}

# ============================================================
# 使用说明
# ============================================================
show_usage() {
    echo "粗粒度消融实验脚本"
    echo ""
    echo "用法:"
    echo "  $0 parallel   — 三个数据集并行运行（推荐，三张卡同时跑）"
    echo "  $0 PHO        — 只运行 PHO (cuda:0)"
    echo "  $0 NYC        — 只运行 NYC (cuda:1)"
    echo "  $0 SIN        — 只运行 SIN (cuda:2)"
    echo "  $0 all        — 串行运行所有数据集"
    echo "  $0 summary    — 仅汇总已有结果到 CSV"
    echo "  $0 clean      — 清理工作目录"
    echo "  $0 stop       — 停止所有正在运行的实验"
    echo ""
    echo "推荐: $0 parallel"
}

# ============================================================
# 主入口
# ============================================================
case "${1}" in
    "PHO")
        setup_and_run_dataset "PHO" "cuda:0" 60 25
        generate_summary
        ;;
    "NYC")
        setup_and_run_dataset "NYC" "cuda:1" 40 10
        generate_summary
        ;;
    "SIN")
        setup_and_run_dataset "SIN" "cuda:2" 60 10
        generate_summary
        ;;
    "all")
        echo "============================================"
        echo "串行运行所有数据集消融实验"
        echo "============================================"
        setup_and_run_dataset "PHO" "cuda:0" 60 25
        setup_and_run_dataset "NYC" "cuda:1" 40 10
        setup_and_run_dataset "SIN" "cuda:2" 60 10
        generate_summary
        echo "所有实验完成！"
        ;;
    "parallel")
        echo "============================================================"
        echo "并行运行三个数据集的粗粒度消融实验"
        echo "  PHO -> cuda:0  (epoch=25, embed=60)"
        echo "  NYC -> cuda:1  (epoch=10, embed=40)"
        echo "  SIN -> cuda:2  (epoch=10, embed=60)"
        echo "  lr = ${LR}, run_times = ${RUN_TIMES}"
        echo "============================================================"
        echo ""

        mkdir -p "${ABLATION_DIR}/PHO" "${ABLATION_DIR}/NYC" "${ABLATION_DIR}/SIN"

        (setup_and_run_dataset "PHO" "cuda:0" 60 25) > "${ABLATION_DIR}/PHO/all.log" 2>&1 &
        PID_PHO=$!

        (setup_and_run_dataset "NYC" "cuda:1" 40 10) > "${ABLATION_DIR}/NYC/all.log" 2>&1 &
        PID_NYC=$!

        (setup_and_run_dataset "SIN" "cuda:2" 60 10) > "${ABLATION_DIR}/SIN/all.log" 2>&1 &
        PID_SIN=$!

        echo "三个数据集已并行启动:"
        echo "  PHO PID: ${PID_PHO}  日志: ablation/PHO/all.log"
        echo "  NYC PID: ${PID_NYC}  日志: ablation/NYC/all.log"
        echo "  SIN PID: ${PID_SIN}  日志: ablation/SIN/all.log"
        echo ""
        echo "查看进度:"
        echo "  tail -f ablation/PHO/all.log"
        echo "  tail -f ablation/NYC/all.log"
        echo "  tail -f ablation/SIN/all.log"
        echo ""
        echo "等待所有实验完成..."

        wait ${PID_PHO}
        echo "✓ PHO 完成"

        wait ${PID_NYC}
        echo "✓ NYC 完成"

        wait ${PID_SIN}
        echo "✓ SIN 完成"

        # 汇总
        generate_summary

        echo ""
        echo "============================================================"
        echo "所有消融实验完成！"
        echo "结果目录: ablation/"
        echo "汇总CSV:  ablation/ablation_summary.csv"
        echo ""
        echo "清理工作目录: $0 clean"
        echo "============================================================"
        ;;
    "summary")
        generate_summary
        ;;
    "clean")
        clean_workspaces
        ;;
    "stop")
        echo "正在停止所有消融实验进程..."
        PIDS=$(pgrep -f "python.*main.py" 2>/dev/null)
        if [ -n "$PIDS" ]; then
            echo "找到以下进程:"
            ps -p $PIDS -o pid,start,cmd 2>/dev/null
            echo ""
            pkill -f "python.*main.py"
            sleep 2
            REMAINING=$(pgrep -f "python.*main.py" 2>/dev/null)
            if [ -n "$REMAINING" ]; then
                pkill -9 -f "python.*main.py"
            fi
            echo "✓ 所有实验进程已停止"
        else
            echo "没有找到正在运行的实验进程"
        fi
        ;;
    *)
        show_usage
        ;;
esac
