task_name = 'PHO'
city = 'PHO'  # PHO, NYC, SIN
gpuId = "cuda:3"

enable_random_mask = True
mask_prop = 0.1
enable_enhance_user = True
enable_ssl = True  # whether enable contrastive learning
enable_distance_sample = False  # whether sample negative samples by distance
neg_sample_count = 5
neg_weight = 1

# Spatiotemporal-aware contrastive learning
enable_spatiotemporal_ssl = False  # whether enable spatiotemporal-aware contrastive learning
ssl_time_scale = 604800  # time scale in seconds (7 days = 604800s, 更大的尺度让权重衰减更慢)
ssl_spatial_scale = 50  # spatial scale in kilometers (更大的尺度适应城市规模)
ssl_temperature = 0.07  # temperature for contrastive learning
aux_weight = 0.5  # weight for auxiliary losses (time and category prediction)
user_enhance_mode = 'None'  # user enhancement mode: 'lhuc', 'memory', 'original', "None"
memory_size = 50  # number of memory slots (used in 'lhuc' and 'memory' modes)
use_rope = True  # whether use RoPE (Rotary Position Embedding) instead of learnable position bias
use_fused_rope_3d = True  # whether use fused 3D RoPE (position + time diff + distance diff)
fused_rope_max_time_diff = 1440  # maximum time difference in minutes (default: 24 hours)
fused_rope_max_distance = 50  # maximum distance in kilometers
use_hstu = True  # whether use HSTU attention mechanism
enable_cross_day_attention = False  # whether enable cross-day attention for long-term sequences
enable_long_short_cross_attention = False  # whether enable cross-attention between long-term and short-term

enable_dynamic_day_length = False
sample_day_length = 14  # range [3,14]

lr = 1e-4
epoch =25
if city == 'SIN':
    embed_size = 60
    run_times = 10
    epoch = 11
    gpuId = "cuda:0"
elif city == 'NYC':
    embed_size = 40
    run_times = 10
    epoch = 11
    gpuId = "cuda:1"
elif city == 'PHO':
    embed_size = 60
    run_times = 10
    epoch = 25
    gpuId = "cuda:0"

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
# 'original' mode does not add suffix

if enable_cross_day_attention:
    output_file_name = output_file_name + "_" + "CrossDay"

if enable_long_short_cross_attention:
    output_file_name = output_file_name + "_" + "LSCrossAttn"

output_file_name = output_file_name + '_embeddingSize' + str(embed_size)
