"""
测试修复后的Fused 3D RoPE实现
对比相邻差值 vs 累积差值
"""
import torch
import math

# 模拟原始的相邻差值实现
def compute_time_diffs_old(timestamps):
    """原始实现：相邻POI的时间差"""
    seq_len = timestamps.shape[0]
    time_diffs = torch.zeros(seq_len, device=timestamps.device, dtype=timestamps.dtype)
    if seq_len > 1:
        diffs = (timestamps[1:] - timestamps[:-1]) / 60.0
        time_diffs[1:] = diffs
    return time_diffs

# 新的累积差值实现
def compute_time_diffs_new(timestamps):
    """新实现：相对于起点的累积时间差"""
    seq_len = timestamps.shape[0]
    if seq_len == 0:
        return torch.zeros(0, device=timestamps.device, dtype=timestamps.dtype)
    time_diffs = (timestamps - timestamps[0]) / 60.0
    return time_diffs

# 测试场景
print("=" * 60)
print("测试场景：用户的一天行为")
print("=" * 60)

# 用户行为序列
POIs = ["家", "咖啡店", "地铁站", "办公楼", "餐厅", "办公楼", "健身房", "家"]
times = ["8:00", "8:30", "8:50", "9:00", "12:00", "13:00", "18:00", "19:00"]

# 时间戳（秒）
timestamps = torch.tensor([
    8*3600,      # 8:00
    8.5*3600,    # 8:30
    8.833*3600,  # 8:50
    9*3600,      # 9:00
    12*3600,     # 12:00
    13*3600,     # 13:00
    18*3600,     # 18:00
    19*3600,     # 19:00
], dtype=torch.float32)

# 计算时间差
time_diffs_old = compute_time_diffs_old(timestamps)
time_diffs_new = compute_time_diffs_new(timestamps)

print("\n用户行为序列:")
print("-" * 60)
for i, (poi, time) in enumerate(zip(POIs, times)):
    print(f"POI {i}: {poi:10s} @ {time}")

print("\n时间差对比:")
print("-" * 60)
print(f"{'POI':<5} {'地点':<10} {'时间':<8} {'相邻差值(旧)':<15} {'累积差值(新)':<15}")
print("-" * 60)
for i, (poi, time, old, new) in enumerate(zip(POIs, times, time_diffs_old, time_diffs_new)):
    print(f"{i:<5} {poi:<10} {time:<8} {old:>10.1f}分钟    {new:>10.1f}分钟")

print("\n" + "=" * 60)
print("关键观察：")
print("=" * 60)

print("\n1. 相对位置的语义:")
print("-" * 60)
print("问题：餐厅(POI 4) 相对于 家(POI 0) 的真实时间差是多少？")
print(f"   - 真实差值: {(timestamps[4] - timestamps[0]) / 60:.1f} 分钟 (4小时)")
print(f"   - 旧编码差值: {time_diffs_old[4] - time_diffs_old[0]:.1f} 分钟 ❌ (错误！)")
print(f"   - 新编码差值: {time_diffs_new[4] - time_diffs_new[0]:.1f} 分钟 ✓ (正确！)")

print("\n2. 任意两个POI的相对关系:")
print("-" * 60)
print("问题：健身房(POI 6) 相对于 咖啡店(POI 1) 的时间差？")
print(f"   - 真实差值: {(timestamps[6] - timestamps[1]) / 60:.1f} 分钟 (9.5小时)")
print(f"   - 旧编码差值: {time_diffs_old[6] - time_diffs_old[1]:.1f} 分钟 ❌ (错误！)")
print(f"   - 新编码差值: {time_diffs_new[6] - time_diffs_new[1]:.1f} 分钟 ✓ (正确！)")

print("\n3. 注意力机制中的影响:")
print("-" * 60)
print("在Self-Attention中，RoPE通过旋转角度差来编码相对位置：")
print("   score(i, j) ∝ cos(θ_i - θ_j)")
print("")
print("旧实现问题：")
print("   - θ_i 是相对于前一个POI的差值")
print("   - θ_i - θ_j 无法正确反映 POI i 和 POI j 的真实时空距离")
print("")
print("新实现优势：")
print("   - θ_i 是相对于起点的累积差值")
print("   - θ_i - θ_j 正确反映了 POI i 和 POI j 的真实时空距离")

print("\n" + "=" * 60)
print("预期改进：")
print("=" * 60)
print("✓ 长序列场景性能提升（累积误差减少）")
print("✓ 远距离POI之间的注意力权重更准确")
print("✓ 模型更容易学习时空模式")
print("✓ 相对位置编码语义更清晰")

print("\n" + "=" * 60)
