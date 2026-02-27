task_name = 'full_model'
city = 'NYC'
gpuId = "cuda:1"

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
user_enhance_mode = 'lhuc'
memory_size = 50
use_rope = True
use_fused_rope_3d = True
fused_rope_max_time_diff = 1440
fused_rope_max_distance = 50
use_hstu = True
enable_cross_day_attention = False
enable_long_short_cross_attention = False

enable_dynamic_day_length = False
sample_day_length = 14

lr = 1e-4
epoch = 10
embed_size = 40
run_times = 3

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
