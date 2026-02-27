import torch
from torch import nn
import torch.nn.functional as F

import settings

device = settings.gpuId if torch.cuda.is_available() else 'cpu'


# RoPE (Rotary Position Embedding) utility functions
def apply_rotary_pos_emb(x, cos, sin):
    """
    Apply rotary position embedding to input tensor.
    Args:
        x: tensor with shape matching cos/sin for broadcasting
        cos, sin: precomputed cos/sin values
    Returns:
        Rotated tensor with same shape as x
    """
    # Split x into even and odd dimensions
    x1 = x[..., 0::2]  # [..., head_dim//2]
    x2 = x[..., 1::2]  # [..., head_dim//2]
    
    # cos and sin should be [..., head_dim], extract even indices for half dimension
    cos_half = cos[..., 0::2]  # [..., head_dim//2]
    sin_half = sin[..., 0::2]  # [..., head_dim//2]
    
    # Apply rotation
    rotated_x1 = x1 * cos_half - x2 * sin_half
    rotated_x2 = x1 * sin_half + x2 * cos_half
    
    # Interleave back: stack and flatten
    rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
    return rotated.flatten(-2)


def precompute_rope_params(head_dim, max_seq_len, base=10000):
    """
    Precompute cos and sin for RoPE.
    Args:
        head_dim: dimension of each attention head
        max_seq_len: maximum sequence length
        base: base for frequency computation
    Returns:
        cos, sin: [max_seq_len, head_dim] tensors
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    position = torch.arange(max_seq_len).float()
    freqs = torch.outer(position, inv_freq)  # [max_seq_len, head_dim//2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, head_dim]
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin


# ============== Fused 3D RoPE Functions ==============
def compute_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute Haversine distance between two points on Earth.
    Args:
        lat1, lon1: latitude and longitude of point 1 (in degrees, tensor)
        lat2, lon2: latitude and longitude of point 2 (in degrees, tensor)
    Returns:
        distance in kilometers (tensor)
    """
    # Convert to radians
    lat1_rad = lat1 * (3.141592653589793 / 180.0)
    lon1_rad = lon1 * (3.141592653589793 / 180.0)
    lat2_rad = lat2 * (3.141592653589793 / 180.0)
    lon2_rad = lon2 * (3.141592653589793 / 180.0)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))  # clamp to avoid numerical issues
    
    r = 6371.0  # Earth's radius in kilometers
    return c * r


def compute_time_diffs(timestamps):
    """
    Compute time differences between consecutive POI visits.
    Args:
        timestamps: tensor of shape [seq_len] containing Unix timestamps (seconds)
    Returns:
        time_diffs: tensor of shape [seq_len] containing time differences in minutes
                    First element is 0 (no previous visit)
    """
    seq_len = timestamps.shape[0]
    time_diffs = torch.zeros(seq_len, device=timestamps.device, dtype=timestamps.dtype)
    if seq_len > 1:
        # Compute differences: timestamps[1:] - timestamps[:-1]
        diffs = (timestamps[1:] - timestamps[:-1]) / 60.0  # Convert seconds to minutes
        time_diffs[1:] = diffs
    return time_diffs


def compute_distance_diffs(latitudes, longitudes):
    """
    Compute distance differences between consecutive POI visits.
    Args:
        latitudes: tensor of shape [seq_len] containing latitudes
        longitudes: tensor of shape [seq_len] containing longitudes
    Returns:
        distance_diffs: tensor of shape [seq_len] containing distances in kilometers
                        First element is 0 (no previous visit)
    """
    seq_len = latitudes.shape[0]
    distance_diffs = torch.zeros(seq_len, device=latitudes.device, dtype=latitudes.dtype)
    if seq_len > 1:
        # Compute distances between consecutive points
        dists = compute_haversine_distance(
            latitudes[:-1], longitudes[:-1],
            latitudes[1:], longitudes[1:]
        )
        distance_diffs[1:] = dists
    return distance_diffs


def compute_fused_rope_3d(seq_indices, time_diffs, distances, head_dim, 
                          max_time_diff=1440, max_distance=50, max_seq_len=100, base=10000):
    """
    Compute fused RoPE cos/sin values combining position, time difference, and distance.
    The head_dim is divided into 3 parts, each encoding one type of information.
    
    Args:
        seq_indices: tensor of shape [seq_len] containing position indices (0, 1, 2, ...)
        time_diffs: tensor of shape [seq_len] containing time differences in minutes
        distances: tensor of shape [seq_len] containing distances in kilometers
        head_dim: dimension of each attention head (must be divisible by 6 for proper splitting)
        max_time_diff: maximum time difference for normalization (default 1440 minutes = 24 hours)
        max_distance: maximum distance for normalization (default 50 km)
        max_seq_len: maximum sequence length for position normalization
        base: base for frequency computation
    Returns:
        cos, sin: tensors of shape [seq_len, head_dim]
    """
    seq_len = seq_indices.shape[0]
    device = seq_indices.device
    
    # Ensure head_dim is divisible by 6 (each of 3 parts needs to be divisible by 2 for RoPE)
    # If not perfectly divisible, we'll handle the remainder
    dim_per_type = head_dim // 3
    remainder = head_dim % 3
    
    # Dimensions for each type
    pos_dim = dim_per_type + (1 if remainder > 0 else 0)
    time_dim = dim_per_type + (1 if remainder > 1 else 0)
    dist_dim = head_dim - pos_dim - time_dim
    
    # Normalize inputs to [0, max_seq_len] range for consistent frequency scaling
    # Position: already in [0, seq_len-1], scale to [0, max_seq_len]
    pos_normalized = seq_indices.float()
    
    # Time: clamp and normalize to [0, max_seq_len]
    time_normalized = torch.clamp(time_diffs, 0, max_time_diff) / max_time_diff * max_seq_len
    
    # Distance: clamp and normalize to [0, max_seq_len]
    dist_normalized = torch.clamp(distances, 0, max_distance) / max_distance * max_seq_len
    
    # Compute inverse frequencies for each dimension type
    # Position encoding
    pos_inv_freq = 1.0 / (base ** (torch.arange(0, pos_dim, 2, device=device).float() / pos_dim))
    pos_freqs = torch.outer(pos_normalized, pos_inv_freq)  # [seq_len, pos_dim//2]
    pos_emb = torch.cat([pos_freqs, pos_freqs], dim=-1)[:, :pos_dim]  # [seq_len, pos_dim]
    
    # Time encoding
    time_inv_freq = 1.0 / (base ** (torch.arange(0, time_dim, 2, device=device).float() / time_dim))
    time_freqs = torch.outer(time_normalized, time_inv_freq)  # [seq_len, time_dim//2]
    time_emb = torch.cat([time_freqs, time_freqs], dim=-1)[:, :time_dim]  # [seq_len, time_dim]
    
    # Distance encoding
    dist_inv_freq = 1.0 / (base ** (torch.arange(0, dist_dim, 2, device=device).float() / dist_dim))
    dist_freqs = torch.outer(dist_normalized, dist_inv_freq)  # [seq_len, dist_dim//2]
    dist_emb = torch.cat([dist_freqs, dist_freqs], dim=-1)[:, :dist_dim]  # [seq_len, dist_dim]
    
    # Concatenate all embeddings
    fused_emb = torch.cat([pos_emb, time_emb, dist_emb], dim=-1)  # [seq_len, head_dim]
    
    cos = fused_emb.cos()
    sin = fused_emb.sin()
    
    return cos, sin


class CheckInEmbedding(nn.Module):
    def __init__(self, f_embed_size, vocab_size):
        super().__init__()
        self.embed_size = f_embed_size
        poi_num = vocab_size["POI"]
        cat_num = vocab_size["cat"]
        user_num = vocab_size["user"]
        hour_num = vocab_size["hour"]
        day_num = vocab_size["day"]

        self.poi_embed = nn.Embedding(poi_num + 1, self.embed_size, padding_idx=poi_num)
        self.cat_embed = nn.Embedding(cat_num + 1, self.embed_size, padding_idx=cat_num)
        self.user_embed = nn.Embedding(user_num + 1, self.embed_size, padding_idx=user_num)
        self.hour_embed = nn.Embedding(hour_num + 1, self.embed_size, padding_idx=hour_num)
        self.day_embed = nn.Embedding(day_num + 1, self.embed_size, padding_idx=day_num)

    def forward(self, x):
        poi_emb = self.poi_embed(x[0])
        cat_emb = self.cat_embed(x[1])
        user_emb = self.user_embed(x[2])
        hour_emb = self.hour_embed(x[3])
        day_emb = self.day_embed(x[4])

        return torch.cat((poi_emb, cat_emb, user_emb, hour_emb, day_emb), 1)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, use_rope=False, use_fused_rope_3d=False, max_seq_len=100):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads
        self.use_rope = use_rope
        self.use_fused_rope_3d = use_fused_rope_3d
        self.max_seq_len = max_seq_len

        assert (
                self.head_dim * self.heads == self.embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size)
        
        # RoPE: 预计算cos/sin，注册为buffer（不可训练）
        # Only precompute if using standard RoPE (not fused 3D RoPE)
        if use_rope and not use_fused_rope_3d:
            cos, sin = precompute_rope_params(self.head_dim, max_seq_len)
            self.register_buffer('rope_cos', cos)
            self.register_buffer('rope_sin', sin)

    def forward(self, values, keys, query, time_diffs=None, distances=None):
        """
        Args:
            values, keys, query: input tensors
            time_diffs: tensor of shape [seq_len] containing time differences in minutes (for fused RoPE)
            distances: tensor of shape [seq_len] containing distances in km (for fused RoPE)
        """
        value_len, key_len, query_len = values.shape[0], keys.shape[0], query.shape[0]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.reshape(value_len, self.heads, self.head_dim)
        keys = keys.reshape(key_len, self.heads, self.head_dim)
        queries = queries.reshape(query_len, self.heads, self.head_dim)
        
        # Apply RoPE if enabled
        if self.use_rope:
            seq_len = query_len
            if self.use_fused_rope_3d:
                # Use fused 3D RoPE with position, time, and distance
                seq_indices = torch.arange(seq_len, device=query.device)
                # If time_diffs or distances not provided, use zeros (position-only encoding)
                if time_diffs is None:
                    time_diffs = torch.zeros(seq_len, device=query.device)
                if distances is None:
                    distances = torch.zeros(seq_len, device=query.device)
                rope_cos, rope_sin = compute_fused_rope_3d(
                    seq_indices, time_diffs, distances, self.head_dim,
                    max_time_diff=settings.fused_rope_max_time_diff,
                    max_distance=settings.fused_rope_max_distance,
                    max_seq_len=self.max_seq_len
                )
                rope_cos = rope_cos.unsqueeze(1)  # [seq_len, 1, head_dim]
                rope_sin = rope_sin.unsqueeze(1)  # [seq_len, 1, head_dim]
            else:
                # Use standard position-only RoPE
                rope_cos = self.rope_cos[:seq_len, :].unsqueeze(1)  # [seq_len, 1, head_dim]
                rope_sin = self.rope_sin[:seq_len, :].unsqueeze(1)  # [seq_len, 1, head_dim]
            queries = apply_rotary_pos_emb(queries, rope_cos, rope_sin)
            keys = apply_rotary_pos_emb(keys, rope_cos, rope_sin)

        energy = torch.einsum("qhd,khd->hqk", [queries, keys])

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)

        out = torch.einsum("hql,lhd->qhd", [attention, values]).reshape(
            query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        return out


class HSTUAttention(nn.Module):
    """
    HSTU Attention mechanism based on the formula:
    U(X), V(X), Q(X), K(X) = Split(φ1(f1(X)))
    A(X)V(X) = φ2(Q(X)K(X)^T + rab^{p,t})V(X)
    Y(X) = f2(Norm(A(X)V(X)) ⊙ U(X))
    """
    def __init__(self, embed_size, heads, use_rope=False, use_fused_rope_3d=False, max_seq_len=100):
        super(HSTUAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.use_fused_rope_3d = use_fused_rope_3d

        assert (
                self.head_dim * self.heads == self.embed_size
        ), "Embedding size needs to be divisible by heads"

        # φ1: Linear projection + SiLU activation
        self.phi1_linear = nn.Linear(self.embed_size, 4 * self.embed_size, bias=False)
        self.phi1_activation = nn.SiLU()  # SiLU/Swish activation
        
        # Position encoding: either learnable bias or RoPE
        if not use_rope:
            # Relative position bias rab^{p,t}: 包含位置和时间信息
            # [heads, max_seq_len, max_seq_len] - 每个head学习不同的位置-时间偏置模式
            self.relative_position_bias = nn.Parameter(
                torch.zeros(self.heads, max_seq_len, max_seq_len)
            )
        elif not use_fused_rope_3d:
            # Standard RoPE: 预计算cos/sin，注册为buffer（不可训练）
            cos, sin = precompute_rope_params(self.head_dim, max_seq_len)
            self.register_buffer('rope_cos', cos)
            self.register_buffer('rope_sin', sin)
        # If use_fused_rope_3d, cos/sin will be computed dynamically in forward
        
        # φ2: SiLU activation for attention scores
        self.phi2_activation = nn.SiLU()
        
        # f2: final projection
        self.f2 = nn.Linear(self.embed_size, self.embed_size)
        
        # Layer norm
        self.norm = nn.LayerNorm(self.embed_size)

    def forward(self, values, keys, query, time_diffs=None, distances=None):
        """
        For encoder self-attention: values = keys = query
        Args:
            values, keys, query: input tensors
            time_diffs: tensor of shape [seq_len] containing time differences in minutes (for fused RoPE)
            distances: tensor of shape [seq_len] containing distances in km (for fused RoPE)
        """
        seq_len = query.shape[0]
        
        # Apply φ1 (Linear + SiLU) and split into U, V, Q, K
        uvqk = self.phi1_linear(query)  # [seq_len, 4 * embed_size]
        uvqk = self.phi1_activation(uvqk)  # Apply SiLU activation
        u, v, q, k = torch.chunk(uvqk, 4, dim=-1)  # each: [seq_len, embed_size]
        
        # Reshape for multi-head attention: [seq_len, heads, head_dim]
        u = u.reshape(seq_len, self.heads, self.head_dim)
        v = v.reshape(seq_len, self.heads, self.head_dim)
        q = q.reshape(seq_len, self.heads, self.head_dim)
        k = k.reshape(seq_len, self.heads, self.head_dim)
        
        # Transpose for matrix multiplication: [heads, seq_len, head_dim]
        q = q.permute(1, 0, 2)  # [heads, seq_len, head_dim]
        k = k.permute(1, 0, 2)  # [heads, seq_len, head_dim]
        v = v.permute(1, 0, 2)  # [heads, seq_len, head_dim]
        
        # Apply RoPE if enabled
        if self.use_rope:
            # Apply RoPE to q and k
            # q, k are [heads, seq_len, head_dim]
            if self.use_fused_rope_3d:
                # Use fused 3D RoPE with position, time, and distance
                seq_indices = torch.arange(seq_len, device=query.device)
                # If time_diffs or distances not provided, use zeros (position-only encoding)
                if time_diffs is None:
                    time_diffs = torch.zeros(seq_len, device=query.device)
                if distances is None:
                    distances = torch.zeros(seq_len, device=query.device)
                rope_cos, rope_sin = compute_fused_rope_3d(
                    seq_indices, time_diffs, distances, self.head_dim,
                    max_time_diff=settings.fused_rope_max_time_diff,
                    max_distance=settings.fused_rope_max_distance,
                    max_seq_len=self.max_seq_len
                )
                rope_cos = rope_cos.unsqueeze(0)  # [1, seq_len, head_dim] - broadcasts over heads
                rope_sin = rope_sin.unsqueeze(0)  # [1, seq_len, head_dim]
            else:
                # Use standard position-only RoPE
                rope_cos = self.rope_cos[:seq_len, :].unsqueeze(0)  # [1, seq_len, head_dim] - broadcasts over heads
                rope_sin = self.rope_sin[:seq_len, :].unsqueeze(0)  # [1, seq_len, head_dim]
            q = apply_rotary_pos_emb(q, rope_cos, rope_sin)
            k = apply_rotary_pos_emb(k, rope_cos, rope_sin)
        
        # Compute Q(X)K(X)^T: [heads, seq_len, head_dim] @ [heads, head_dim, seq_len] -> [heads, seq_len, seq_len]
        energy = torch.bmm(q, k.transpose(1, 2))  # [heads, seq_len, seq_len]
        
        # Scale by sqrt(head_dim) for numerical stability
        energy = energy / (self.head_dim ** 0.5)
        
        # Add relative position-time bias rab^{p,t} (only if not using RoPE)
        if not self.use_rope:
            # Extract the corresponding bias for current sequence length
            rel_bias = self.relative_position_bias[:, :seq_len, :seq_len]  # [heads, seq_len, seq_len]
            energy = energy + rel_bias
        # If using RoPE, position information is already in Q and K
        
        # Apply φ2 (SiLU) to get attention weights: A(X) = φ2(Q(X)K(X)^T/√d + rab^{p,t})
        attention = self.phi2_activation(energy)  # [heads, seq_len, seq_len]
        
        # A(X)V(X): [heads, seq_len, seq_len] @ [heads, seq_len, head_dim] -> [heads, seq_len, head_dim]
        av = torch.bmm(attention, v)  # [heads, seq_len, head_dim]
        
        # Transpose back: [seq_len, heads, head_dim]
        av = av.permute(1, 0, 2)
        
        # Reshape to [seq_len, embed_size]
        av = av.reshape(seq_len, self.embed_size)
        
        # Normalize
        av_norm = self.norm(av)
        
        # Element-wise multiplication with U (门控机制)
        u_flat = u.reshape(seq_len, self.embed_size)
        out = av_norm * u_flat  # ⊙ operation
        
        # Apply f2
        out = self.f2(out)
        
        return out


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, use_hstu=False, 
                 use_fused_rope_3d=False, max_seq_len=100):
        super(EncoderBlock, self).__init__()
        self.embed_size = embed_size
        self.use_hstu = use_hstu
        self.use_fused_rope_3d = use_fused_rope_3d
        
        # Choose attention mechanism
        if use_hstu:
            self.attention = HSTUAttention(
                self.embed_size, heads, 
                use_rope=settings.use_rope, 
                use_fused_rope_3d=use_fused_rope_3d,
                max_seq_len=max_seq_len
            )
        else:
            self.attention = SelfAttention(
                self.embed_size, heads, 
                use_rope=settings.use_rope, 
                use_fused_rope_3d=use_fused_rope_3d,
                max_seq_len=max_seq_len
            )
        
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, forward_expansion * self.embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * self.embed_size, self.embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, time_diffs=None, distances=None):
        """
        Args:
            value, key, query: input tensors
            time_diffs: tensor of shape [seq_len] containing time differences (for fused RoPE)
            distances: tensor of shape [seq_len] containing distances (for fused RoPE)
        """
        attention = self.attention(value, key, query, time_diffs=time_diffs, distances=distances)  # [len * embed_size]

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        if self.use_hstu:
            forward = x
        else:
            forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            embedding_layer,
            embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout,
            use_hstu=False,
            use_fused_rope_3d=False,
            max_seq_len=100,
    ):
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = embedding_layer
        self.add_module('embedding', self.embedding_layer)
        self.use_hstu = use_hstu
        self.use_fused_rope_3d = use_fused_rope_3d

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    use_hstu=use_hstu,
                    use_fused_rope_3d=use_fused_rope_3d,
                    max_seq_len=max_seq_len,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, feature_seq, latitudes=None, longitudes=None, timestamps=None):
        """
        Args:
            feature_seq: input feature tensor [5, seq_len]
            latitudes: tensor of shape [seq_len] containing latitudes (for fused RoPE)
            longitudes: tensor of shape [seq_len] containing longitudes (for fused RoPE)
            timestamps: tensor of shape [seq_len] containing timestamps in seconds (for fused RoPE)
        """
        embedding = self.embedding_layer(feature_seq)  # [len, embedding]
        out = self.dropout(embedding)

        # Compute time differences and distances if fused RoPE is enabled
        time_diffs = None
        distances = None
        if self.use_fused_rope_3d and latitudes is not None and longitudes is not None and timestamps is not None:
            time_diffs = compute_time_diffs(timestamps)
            distances = compute_distance_diffs(latitudes, longitudes)

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case
        for layer in self.layers:
            out = layer(out, out, out, time_diffs=time_diffs, distances=distances)

        return out


# Attention for query and key with different dimension
class Attention(nn.Module):
    def __init__(
            self,
            qdim,
            kdim,
    ):
        super().__init__()

        # Resize q's dimension to k
        self.expansion = nn.Linear(qdim, kdim)

    def forward(self, query, key, value):
        q = self.expansion(query)  # [embed_size]
        temp = torch.inner(q, key)
        weight = torch.softmax(temp, dim=0)  # [len, 1]
        weight = torch.unsqueeze(weight, 1)
        temp2 = torch.mul(value, weight)
        out = torch.sum(temp2, 0)  # sum([len, embed_size] * [len, 1])  -> [embed_size]

        return out


class CLSPRec(nn.Module):
    def __init__(
            self,
            vocab_size,
            f_embed_size=60,
            num_encoder_layers=1,
            num_lstm_layers=1,
            num_heads=1,
            forward_expansion=2,
            dropout_p=0.5,
            use_hstu=False,
            use_fused_rope_3d=False,
            max_seq_len=100,
            enable_cross_day_attention=False,
            enable_long_short_cross_attention=False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.total_embed_size = f_embed_size * 5
        self.enable_long_short_cross_attention = enable_long_short_cross_attention
        self.use_fused_rope_3d = use_fused_rope_3d

        # Layers
        self.embedding = CheckInEmbedding(
            f_embed_size,
            vocab_size
        )
        self.encoder = TransformerEncoder(
            self.embedding,
            self.total_embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout_p,
            use_hstu=use_hstu,
            use_fused_rope_3d=use_fused_rope_3d,
            max_seq_len=max_seq_len,
        )
        
        # Cross-Attention: 短期序列(Query) 关注 长期序列(Key, Value)
        if enable_long_short_cross_attention:
            if use_hstu:
                self.long_short_cross_attention = HSTUAttention(
                    self.total_embed_size, 
                    num_heads, 
                    max_seq_len=max_seq_len
                )
            else:
                self.long_short_cross_attention = SelfAttention(
                    self.total_embed_size, 
                    num_heads
                )
            self.cross_attn_norm = nn.LayerNorm(self.total_embed_size)
            self.cross_attn_ffn = nn.Sequential(
                nn.Linear(self.total_embed_size, forward_expansion * self.total_embed_size),
                nn.ReLU(),
                nn.Linear(forward_expansion * self.total_embed_size, self.total_embed_size),
            )
            self.cross_attn_norm2 = nn.LayerNorm(self.total_embed_size)
            self.cross_attn_dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(
            input_size=self.total_embed_size,
            hidden_size=self.total_embed_size,
            num_layers=num_lstm_layers,
            dropout=0
        )
        self.final_attention = Attention(
            qdim=f_embed_size,
            kdim=self.total_embed_size
        )
        self.out_linear = nn.Sequential(nn.Linear(self.total_embed_size, self.total_embed_size * forward_expansion),
                                        nn.LeakyReLU(),
                                        nn.Dropout(dropout_p),
                                        nn.Linear(self.total_embed_size * forward_expansion, vocab_size["POI"]))
        
        # Auxiliary prediction heads
        self.out_linear_cat = nn.Sequential(nn.Linear(self.total_embed_size, self.total_embed_size * forward_expansion),
                                            nn.LeakyReLU(),
                                            nn.Dropout(dropout_p),
                                            nn.Linear(self.total_embed_size * forward_expansion, vocab_size["cat"]))
        self.out_linear_hour = nn.Sequential(nn.Linear(self.total_embed_size, self.total_embed_size * forward_expansion),
                                             nn.LeakyReLU(),
                                             nn.Dropout(dropout_p),
                                             nn.Linear(self.total_embed_size * forward_expansion, vocab_size["hour"]))

        self.loss_func = nn.CrossEntropyLoss()

        self.tryone_line2 = nn.Linear(self.total_embed_size, f_embed_size)
        self.enhance_val = nn.Parameter(torch.tensor(0.5))  # Used in 'original' mode
        self.enable_cross_day_attention = enable_cross_day_attention

        # User enhancement components (mode-dependent)
        if settings.user_enhance_mode in ['memory', 'lhuc']:
            # Memory bank: K and V (shared by 'memory' and 'lhuc' modes)
            self.memory_keys = nn.Parameter(torch.randn(settings.memory_size, f_embed_size) * 0.1)
            self.memory_values = nn.Parameter(torch.randn(settings.memory_size, f_embed_size) * 0.1)
            
            # Query network: f_Q (shared by 'memory' and 'lhuc' modes)
            self.query_net = nn.Sequential(
                nn.Linear(f_embed_size, f_embed_size),
                nn.ReLU(),
                nn.Linear(f_embed_size, f_embed_size)
            )
        
        if settings.user_enhance_mode == 'memory':
            # Gated fusion for Memory Network mode
            self.gate_static = nn.Linear(f_embed_size * 2, 1)
            self.gate_memory = nn.Linear(f_embed_size * 2, 1)
        
        if settings.user_enhance_mode == 'lhuc':
            # LHUC scaling weight
            self.lhuc_weight = nn.Linear(f_embed_size, f_embed_size)

    def feature_mask(self, sequences, mask_prop):
        masked_sequences = []
        for seq in sequences:  # each long term sequences
            feature_seq, day_nums = seq[0], seq[1]
            seq_len = len(feature_seq[0])
            mask_count = torch.ceil(mask_prop * torch.tensor(seq_len)).int()
            masked_index = torch.randperm(seq_len - 1) + torch.tensor(1)
            masked_index = masked_index[:mask_count]  # randomly generate mask index

            feature_seq[0, masked_index] = self.vocab_size["POI"]  # mask POI
            feature_seq[1, masked_index] = self.vocab_size["cat"]  # mask cat
            feature_seq[3, masked_index] = self.vocab_size["hour"]  # mask hour
            feature_seq[4, masked_index] = self.vocab_size["day"]  # mask day

            # 保留完整的序列信息（包括时空信息）
            if len(seq) >= 5:
                # seq包含: (features, day_nums, latitudes, longitudes, timestamps)
                masked_sequences.append(seq)
            else:
                # 只有基础特征
                masked_sequences.append((feature_seq, day_nums))
        return masked_sequences

    def compute_time_weight(self, timestamp1, timestamp2, tau):
        """
        计算时间权重（指数衰减）
        Args:
            timestamp1, timestamp2: Unix时间戳（秒）
            tau: 时间尺度参数
        Returns:
            weight: [0, 1]之间的权重值
        """
        time_diff = torch.abs(timestamp1 - timestamp2)
        weight = torch.exp(-time_diff / tau)
        return weight

    def compute_spatial_weight(self, lat1, lon1, lat2, lon2, tau):
        """
        计算空间权重（指数衰减）
        Args:
            lat1, lon1: 地点1的经纬度
            lat2, lon2: 地点2的经纬度
            tau: 空间尺度参数（公里）
        Returns:
            weight: [0, 1]之间的权重值
        """
        distance = compute_haversine_distance(lat1, lon1, lat2, lon2)
        weight = torch.exp(-distance / tau)
        return weight

    def ssl(self, embedding_1, embedding_2, neg_embedding, 
            time1=None, time2=None, neg_times=None,
            loc1=None, loc2=None, neg_locs=None):
        """
        对比学习损失函数（支持时空感知）
        Args:
            embedding_1, embedding_2: 正样本对的表示
            neg_embedding: 负样本的表示
            time1, time2, neg_times: 时间戳（可选，用于时空感知）
            loc1, loc2, neg_locs: 位置信息 [lat, lon]（可选，用于时空感知）
        """
        def score(x1, x2):
            return torch.mean(torch.mul(x1, x2))
        
        # 判断是否启用时空感知
        use_spatiotemporal = (settings.enable_spatiotemporal_ssl and 
                              time1 is not None and loc1 is not None)
        
        if use_spatiotemporal:
            # 时空感知版本
            # 打印一次确认信息
            if not hasattr(self, '_spatiotemporal_confirmed'):
                print('✅ 时空感知对比学习已启用！')
                self._spatiotemporal_confirmed = True
            
            # 1. 计算正样本的语义相似度
            pos_semantic = score(embedding_1, embedding_2)
            
            # 2. 计算正样本的时间权重
            pos_time_weight = self.compute_time_weight(
                time1, time2, settings.ssl_time_scale
            )
            
            # 3. 计算正样本的空间权重
            pos_spatial_weight = self.compute_spatial_weight(
                loc1[0], loc1[1], loc2[0], loc2[1], settings.ssl_spatial_scale
            )
            
            # 4. 融合得到正样本分数
            pos = pos_semantic * pos_time_weight * pos_spatial_weight
            
            # 5. 计算负样本分数
            neg1_semantic = score(embedding_1, neg_embedding)
            neg2_semantic = score(embedding_2, neg_embedding)
            
            # 负样本的时空权重
            neg1_time_weight = self.compute_time_weight(
                time1, neg_times, settings.ssl_time_scale
            )
            neg2_time_weight = self.compute_time_weight(
                time2, neg_times, settings.ssl_time_scale
            )
            
            neg1_spatial_weight = self.compute_spatial_weight(
                loc1[0], loc1[1], neg_locs[0], neg_locs[1], settings.ssl_spatial_scale
            )
            neg2_spatial_weight = self.compute_spatial_weight(
                loc2[0], loc2[1], neg_locs[0], neg_locs[1], settings.ssl_spatial_scale
            )
            
            # 融合负样本分数
            neg1 = neg1_semantic * neg1_time_weight * neg1_spatial_weight
            neg2 = neg2_semantic * neg2_time_weight * neg2_spatial_weight
            neg = (neg1 + neg2) / 2
        else:
            # 原始版本（只用语义相似度）
            # 打印一次确认信息
            if not hasattr(self, '_original_ssl_confirmed'):
                print('ℹ️  使用原始对比学习（未启用时空感知）')
                if settings.enable_spatiotemporal_ssl:
                    print('   原因：时空信息缺失 (time1={}, loc1={})'.format(time1 is not None, loc1 is not None))
                self._original_ssl_confirmed = True
            
            pos = score(embedding_1, embedding_2)
            neg1 = score(embedding_1, neg_embedding)
            neg2 = score(embedding_2, neg_embedding)
            neg = (neg1 + neg2) / 2
        
        # InfoNCE损失
        one = torch.ones(1, device=embedding_1.device)
        con_loss = torch.sum(
            -torch.log(1e-8 + torch.sigmoid(pos)) - 
            torch.log(1e-8 + (one - torch.sigmoid(neg)))
        )
        return con_loss

    def forward(self, sample, neg_sample_list):
        # Process input sample
        long_term_sequences = sample[:-1]
        short_term_sequence = sample[-1]
        short_term_features = short_term_sequence[0][:, :- 1]
        target = short_term_sequence[0][0, -1]
        target_cat = short_term_sequence[0][1, -1]
        target_hour = short_term_sequence[0][3, -1]
        user_id = short_term_sequence[0][2, 0]
        
        # Extract spatiotemporal features for fused RoPE (if enabled)
        # short_term_sequence[0] shape: [5, seq_len] for basic features
        # short_term_sequence may have additional spatiotemporal info
        short_term_latitudes = None
        short_term_longitudes = None
        short_term_timestamps = None
        
        
        if self.use_fused_rope_3d and len(short_term_sequence) >= 5:
            # Spatiotemporal info: (features, day_nums, latitudes, longitudes, timestamps)
            short_term_latitudes = short_term_sequence[2][:- 1]  # exclude target
            short_term_longitudes = short_term_sequence[3][:- 1]
            short_term_timestamps = short_term_sequence[4][:- 1]

        # Random mask long-term sequences
        long_term_sequences = self.feature_mask(long_term_sequences, settings.mask_prop)

        # Long-term
        # 初始化时空信息变量（用于SSL）
        concat_latitudes = None
        concat_longitudes = None
        concat_timestamps = None
        
        if not self.enable_cross_day_attention:
            # 方式1: 每天独立编码（天内attention）
            long_term_out = []
            all_long_term_latitudes = []
            all_long_term_longitudes = []
            all_long_term_timestamps = []
            
            for seq in long_term_sequences:
                # Extract spatiotemporal info for this long-term sequence
                seq_latitudes = None
                seq_longitudes = None
                seq_timestamps = None
                if self.use_fused_rope_3d and len(seq) >= 5:
                    seq_latitudes = seq[2]
                    seq_longitudes = seq[3]
                    seq_timestamps = seq[4]
                    # 收集时空信息用于SSL
                    all_long_term_latitudes.append(seq[2])
                    all_long_term_longitudes.append(seq[3])
                    all_long_term_timestamps.append(seq[4])
                
                output = self.encoder(
                    feature_seq=seq[0],
                    latitudes=seq_latitudes,
                    longitudes=seq_longitudes,
                    timestamps=seq_timestamps
                )
                long_term_out.append(output)
            
            long_term_catted = torch.cat(long_term_out, dim=0)
            
            # 拼接所有长期序列的时空信息（用于SSL）
            if self.use_fused_rope_3d and len(all_long_term_latitudes) > 0:
                concat_latitudes = torch.cat(all_long_term_latitudes, dim=0)
                concat_longitudes = torch.cat(all_long_term_longitudes, dim=0)
                concat_timestamps = torch.cat(all_long_term_timestamps, dim=0)
        else:
            # 方式2: 跨天attention - 先拼接所有天的特征，然后一起编码
            # 收集所有长期序列的特征
            all_long_term_features = []
            all_long_term_latitudes = []
            all_long_term_longitudes = []
            all_long_term_timestamps = []
            for seq in long_term_sequences:
                # seq[0]: [5, seq_len] - 某一天的特征
                all_long_term_features.append(seq[0])
                if self.use_fused_rope_3d and len(seq) >= 5:
                    all_long_term_latitudes.append(seq[2])
                    all_long_term_longitudes.append(seq[3])
                    all_long_term_timestamps.append(seq[4])
            
            # 在时间步维度（dim=1）上拼接所有天的特征
            # 结果: [5, total_seq_len] 其中 total_seq_len = sum of all days' seq_len
            long_term_features_concat = torch.cat(all_long_term_features, dim=1)
            
            # Concatenate spatiotemporal info if using fused RoPE
            concat_latitudes = None
            concat_longitudes = None
            concat_timestamps = None
            if self.use_fused_rope_3d and len(all_long_term_latitudes) > 0:
                concat_latitudes = torch.cat(all_long_term_latitudes, dim=0)
                concat_longitudes = torch.cat(all_long_term_longitudes, dim=0)
                concat_timestamps = torch.cat(all_long_term_timestamps, dim=0)
            
            # 对拼接后的所有POI一起做attention（跨天交互）
            long_term_catted = self.encoder(
                feature_seq=long_term_features_concat,
                latitudes=concat_latitudes,
                longitudes=concat_longitudes,
                timestamps=concat_timestamps
            )
            

        # Short-term
        if not self.enable_long_short_cross_attention:
            # 原始方式: 短期序列独立编码（Self-Attention）
            short_term_state = self.encoder(
                feature_seq=short_term_features,
                latitudes=short_term_latitudes,
                longitudes=short_term_longitudes,
                timestamps=short_term_timestamps
            )
        else:
            # 新方式: 短期序列先自编码，然后通过Cross-Attention查询长期信息
            # Step 1: 短期序列自编码（Self-Attention）
            short_term_self_encoded = self.encoder(
                feature_seq=short_term_features,
                latitudes=short_term_latitudes,
                longitudes=short_term_longitudes,
                timestamps=short_term_timestamps
            )
            
            # Step 2: Cross-Attention
            # Query: 短期序列的表示
            # Key & Value: 长期序列的表示
            cross_attn_out = self.long_short_cross_attention(
                values=long_term_catted,      # 长期序列作为Value
                keys=long_term_catted,        # 长期序列作为Key
                query=short_term_self_encoded # 短期序列作为Query
            )
            
            # Step 3: 残差连接 + LayerNorm
            short_term_attended = self.cross_attn_dropout(
                self.cross_attn_norm(cross_attn_out + short_term_self_encoded)
            )
            
            # Step 4: Feed-Forward Network
            ffn_out = self.cross_attn_ffn(short_term_attended)
            short_term_state = self.cross_attn_dropout(
                self.cross_attn_norm2(ffn_out + short_term_attended)
            )

        # User enhancement
        # Step 1: Get static and dynamic representations (all modes need these)
        user_embed_static = self.embedding.user_embed(user_id)  # h_static
        embedding = torch.unsqueeze(self.embedding(short_term_features), 0)
        output, _ = self.lstm(embedding)
        short_term_enhance = torch.squeeze(output)
        user_embed_dynamic = self.tryone_line2(torch.mean(short_term_enhance, dim=0))  # h_dynamic

        # Step 2: Mode-specific fusion
        if settings.user_enhance_mode == 'lhuc':
            # Mode 1: LHUC (Learning Hidden Unit Contributions)
            # 2.1 Generate query and retrieve memory
            query = self.query_net(user_embed_dynamic)  # q = f_Q(h_dynamic)
            similarity = torch.matmul(query, self.memory_keys.T) / (query.size(-1) ** 0.5)  # s_i
            attention_weights = F.softmax(similarity, dim=-1)  # alpha_i
            memory_repr = torch.matmul(attention_weights, self.memory_values)  # h_memory
            
            # 2.2 Content aggregation
            h_content = user_embed_dynamic + memory_repr  # h_content = h_dynamic + h_memory
            
            # 2.3 Generate scaling factor: g = 2·σ(W·h_static + b)
            scaling_factor = 2.0 * torch.sigmoid(self.lhuc_weight(user_embed_static))
            
            # 2.4 Modulation fusion: h_enhanced = g ⊙ h_content
            user_embed = scaling_factor * h_content
        
        elif settings.user_enhance_mode == 'memory':
            # Mode 2: Memory Network (gated fusion of h_static and h_memory)
            # 2.1 Generate query and retrieve memory
            query = self.query_net(user_embed_dynamic)  # q = f_Q(h_dynamic)
            similarity = torch.matmul(query, self.memory_keys.T) / (query.size(-1) ** 0.5)  # s_i
            attention_weights = F.softmax(similarity, dim=-1)  # alpha_i
            memory_repr = torch.matmul(attention_weights, self.memory_values)  # h_memory
            
            # 2.2 Gated fusion
            combined = torch.cat([user_embed_static, memory_repr], dim=-1)
            gate_s = torch.sigmoid(self.gate_static(combined))  # g_s
            gate_m = torch.sigmoid(self.gate_memory(combined))  # g_m
            gate_sum = gate_s + gate_m
            
            # 2.3 Final fusion
            user_embed = (gate_s * user_embed_static + gate_m * memory_repr) / gate_sum
        
        elif settings.user_enhance_mode == 'original':  # 'original'
            # Mode 3: Original (linear interpolation of h_static and h_dynamic)
            user_embed = self.enhance_val * user_embed_static + (1 - self.enhance_val) * user_embed_dynamic
        else:
            user_embed = user_embed_static
        # SSL
        if len(neg_sample_list) > 0:
            neg_short_term_states = []
            for neg_day_sample in neg_sample_list:
                neg_trajectory_features = neg_day_sample[0]
                # Extract spatiotemporal info for negative samples if using fused RoPE
                neg_latitudes = None
                neg_longitudes = None
                neg_timestamps = None
                if self.use_fused_rope_3d and len(neg_day_sample) >= 5:
                    neg_latitudes = neg_day_sample[2]
                    neg_longitudes = neg_day_sample[3]
                    neg_timestamps = neg_day_sample[4]
                neg_short_term_state = self.encoder(
                    feature_seq=neg_trajectory_features,
                    latitudes=neg_latitudes,
                    longitudes=neg_longitudes,
                    timestamps=neg_timestamps
                )
                neg_short_term_state = torch.mean(neg_short_term_state, dim=0)
                neg_short_term_states.append(neg_short_term_state)

            short_embed_mean = torch.mean(short_term_state, dim=0)
            long_embed_mean = torch.mean(long_term_catted, dim=0)
            neg_embed_mean = torch.mean(torch.stack(neg_short_term_states), dim=0)
            
            # 提取时空信息（如果启用时空感知SSL）
            if settings.enable_spatiotemporal_ssl and self.use_fused_rope_3d:
                # 短期序列的最后一个POI的时空信息
                short_time = short_term_timestamps[-1] if short_term_timestamps is not None else None
                short_loc = torch.stack([
                    short_term_latitudes[-1], 
                    short_term_longitudes[-1]
                ]) if short_term_latitudes is not None else None
                
                # 长期序列的最后一个POI的时空信息
                # 注意：long_term_catted是拼接后的，需要获取原始的时空信息
                if not self.enable_cross_day_attention:
                    # 独立编码模式：取最后一天的最后一个POI
                    last_seq = long_term_sequences[-1]
                    # #region agent log
                    if not hasattr(self, '_long_seq_debug'):
                        import json, time
                        log_data = {'location':'CLSPRec.py:1048','message':'长期序列结构','data':{'last_seq_len':len(last_seq),'last_seq_type':str(type(last_seq)),'has_spatiotemporal':len(last_seq)>=5},'timestamp':int(time.time()*1000),'hypothesisId':'E'}
                        with open('/data/xwx/code/CLSPRec/.cursor/debug.log','a') as f: f.write(json.dumps(log_data)+'\n')
                        self._long_seq_debug = True
                    # #endregion
                    if len(last_seq) >= 5:
                        long_time = last_seq[4][-1]
                        long_loc = torch.stack([last_seq[2][-1], last_seq[3][-1]])
                    else:
                        long_time = None
                        long_loc = None
                else:
                    # 跨天编码模式：取拼接后的最后一个POI
                    long_time = concat_timestamps[-1] if concat_timestamps is not None else None
                    long_loc = torch.stack([
                        concat_latitudes[-1], 
                        concat_longitudes[-1]
                    ]) if concat_latitudes is not None else None
                
                # 负样本的最后一个POI的时空信息（平均）
                if len(neg_sample_list) > 0 and len(neg_sample_list[0]) >= 5:
                    neg_times = torch.stack([neg[4][-1] for neg in neg_sample_list])
                    neg_lats = torch.stack([neg[2][-1] for neg in neg_sample_list])
                    neg_lons = torch.stack([neg[3][-1] for neg in neg_sample_list])
                    neg_time = torch.mean(neg_times)
                    neg_loc = torch.stack([torch.mean(neg_lats), torch.mean(neg_lons)])
                else:
                    neg_time = None
                    neg_loc = None
                
                # #region agent log
                if not hasattr(self, '_ssl_params_debug'):
                    import json, time
                    log_data = {'location':'CLSPRec.py:1074','message':'SSL参数','data':{'short_time_is_none':short_time is None,'long_time_is_none':long_time is None,'neg_time_is_none':neg_time is None,'short_loc_is_none':short_loc is None,'long_loc_is_none':long_loc is None,'neg_loc_is_none':neg_loc is None},'timestamp':int(time.time()*1000),'hypothesisId':'E'}
                    with open('/data/xwx/code/CLSPRec/.cursor/debug.log','a') as f: f.write(json.dumps(log_data)+'\n')
                    self._ssl_params_debug = True
                # #endregion
                
                ssl_loss = self.ssl(
                    short_embed_mean, long_embed_mean, neg_embed_mean,
                    time1=short_time, time2=long_time, neg_times=neg_time,
                    loc1=short_loc, loc2=long_loc, neg_locs=neg_loc
                )
            else:
                # 原始版本（不使用时空信息）
                ssl_loss = self.ssl(short_embed_mean, long_embed_mean, neg_embed_mean)
        else:
            ssl_loss = torch.tensor(0.0).to(device)

        # Final predict
        h_all = torch.cat((short_term_state, long_term_catted))
        final_att = self.final_attention(user_embed, h_all, h_all)
        output = self.out_linear(final_att)
        output_cat = self.out_linear_cat(final_att)
        output_hour = self.out_linear_hour(final_att)

        label = torch.unsqueeze(target, 0)
        pred = torch.unsqueeze(output, 0)
        label_cat = torch.unsqueeze(target_cat, 0)
        pred_cat = torch.unsqueeze(output_cat, 0)
        label_hour = torch.unsqueeze(target_hour, 0)
        pred_hour = torch.unsqueeze(output_hour, 0)

        pred_loss = self.loss_func(pred, label)
        pred_loss_cat = self.loss_func(pred_cat, label_cat)
        pred_loss_hour = self.loss_func(pred_hour, label_hour)
        loss = pred_loss + (pred_loss_cat + pred_loss_hour) * settings.aux_weight + ssl_loss * settings.neg_weight
        return loss, output

    def predict(self, sample, neg_sample_list):
        _, pred_raw = self.forward(sample, neg_sample_list)
        ranking = torch.sort(pred_raw, descending=True)[1]
        target = sample[-1][0][0, -1]

        return ranking, target
