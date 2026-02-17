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
    def __init__(self, embed_size, heads, use_rope=False, max_seq_len=100):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads
        self.use_rope = use_rope

        assert (
                self.head_dim * self.heads == self.embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size)
        
        # RoPE: 预计算cos/sin，注册为buffer（不可训练）
        if use_rope:
            cos, sin = precompute_rope_params(self.head_dim, max_seq_len)
            self.register_buffer('rope_cos', cos)
            self.register_buffer('rope_sin', sin)

    def forward(self, values, keys, query):
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
    def __init__(self, embed_size, heads, use_rope=False, max_seq_len=100):
        super(HSTUAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope

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
        else:
            # RoPE: 预计算cos/sin，注册为buffer（不可训练）
            cos, sin = precompute_rope_params(self.head_dim, max_seq_len)
            self.register_buffer('rope_cos', cos)
            self.register_buffer('rope_sin', sin)
        
        # φ2: SiLU activation for attention scores
        self.phi2_activation = nn.SiLU()
        
        # f2: final projection
        self.f2 = nn.Linear(self.embed_size, self.embed_size)
        
        # Layer norm
        self.norm = nn.LayerNorm(self.embed_size)

    def forward(self, values, keys, query):
        """
        For encoder self-attention: values = keys = query
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
            # rope_cos/sin are [seq_len, head_dim], need to unsqueeze for heads dimension
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
    def __init__(self, embed_size, heads, dropout, forward_expansion, use_hstu=False, max_seq_len=100):
        super(EncoderBlock, self).__init__()
        self.embed_size = embed_size
        self.use_hstu = use_hstu
        
        # Choose attention mechanism
        if use_hstu:
            self.attention = HSTUAttention(self.embed_size, heads, use_rope=settings.use_rope, max_seq_len=max_seq_len)
        else:
            self.attention = SelfAttention(self.embed_size, heads, use_rope=settings.use_rope, max_seq_len=max_seq_len)
        
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, forward_expansion * self.embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * self.embed_size, self.embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)  # [len * embed_size]

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
            max_seq_len=100,
    ):
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = embedding_layer
        self.add_module('embedding', self.embedding_layer)
        self.use_hstu = use_hstu

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    use_hstu=use_hstu,
                    max_seq_len=max_seq_len,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, feature_seq):
        embedding = self.embedding_layer(feature_seq)  # [len, embedding]
        out = self.dropout(embedding)

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case
        for layer in self.layers:
            out = layer(out, out, out)

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
            max_seq_len=100,
            enable_cross_day_attention=False,
            enable_long_short_cross_attention=False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.total_embed_size = f_embed_size * 5
        self.enable_long_short_cross_attention = enable_long_short_cross_attention

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
        self.enhance_val = nn.Parameter(torch.tensor(0.5))
        self.enable_cross_day_attention = enable_cross_day_attention

        # Memory Network for user enhancement
        if settings.use_memory_network:
            # Memory bank: K and V
            self.memory_keys = nn.Parameter(torch.randn(settings.memory_size, f_embed_size) * 0.1)
            self.memory_values = nn.Parameter(torch.randn(settings.memory_size, f_embed_size) * 0.1)
            
            # Query network: f_Q
            self.query_net = nn.Sequential(
                nn.Linear(f_embed_size, f_embed_size),
                nn.ReLU(),
                nn.Linear(f_embed_size, f_embed_size)
            )
            
            # Gated fusion: g_s, g_m (only static and memory, no dynamic)
            self.gate_static = nn.Linear(f_embed_size * 2, 1)
            self.gate_memory = nn.Linear(f_embed_size * 2, 1)

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

            masked_sequences.append((feature_seq, day_nums))
        return masked_sequences

    def ssl(self, embedding_1, embedding_2, neg_embedding):
        def score(x1, x2):
            return torch.mean(torch.mul(x1, x2))

        def single_infoNCE_loss_simple(embedding1, embedding2, neg_embedding):
            pos = score(embedding1, embedding2)
            neg1 = score(embedding1, neg_embedding)
            neg2 = score(embedding2, neg_embedding)
            neg = (neg1 + neg2) / 2
            one = torch.cuda.FloatTensor([1], device=device)
            con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg))))
            return con_loss

        ssl_loss = single_infoNCE_loss_simple(embedding_1, embedding_2, neg_embedding)
        return ssl_loss

    def forward(self, sample, neg_sample_list):
        # Process input sample
        long_term_sequences = sample[:-1]
        short_term_sequence = sample[-1]
        short_term_features = short_term_sequence[0][:, :- 1]
        target = short_term_sequence[0][0, -1]
        target_cat = short_term_sequence[0][1, -1]
        target_hour = short_term_sequence[0][3, -1]
        user_id = short_term_sequence[0][2, 0]

        # Random mask long-term sequences
        long_term_sequences = self.feature_mask(long_term_sequences, settings.mask_prop)

        # Long-term
        if not self.enable_cross_day_attention:
            # 方式1: 每天独立编码（天内attention）
            long_term_out = []
            for seq in long_term_sequences:
                output = self.encoder(feature_seq=seq[0])
                long_term_out.append(output)
            long_term_catted = torch.cat(long_term_out, dim=0)
        else:
            # 方式2: 跨天attention - 先拼接所有天的特征，然后一起编码
            # 收集所有长期序列的特征
            all_long_term_features = []
            for seq in long_term_sequences:
                # seq[0]: [5, seq_len] - 某一天的特征
                all_long_term_features.append(seq[0])
            
            # 在时间步维度（dim=1）上拼接所有天的特征
            # 结果: [5, total_seq_len] 其中 total_seq_len = sum of all days' seq_len
            long_term_features_concat = torch.cat(all_long_term_features, dim=1)
            
            # 对拼接后的所有POI一起做attention（跨天交互）
            long_term_catted = self.encoder(feature_seq=long_term_features_concat)
            

        # Short-term
        if not self.enable_long_short_cross_attention:
            # 原始方式: 短期序列独立编码（Self-Attention）
            short_term_state = self.encoder(feature_seq=short_term_features)
        else:
            # 新方式: 短期序列先自编码，然后通过Cross-Attention查询长期信息
            # Step 1: 短期序列自编码（Self-Attention）
            short_term_self_encoded = self.encoder(feature_seq=short_term_features)
            
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
        # Step 1: Get static and dynamic representations
        user_embed_static = self.embedding.user_embed(user_id)  # h_static
        embedding = torch.unsqueeze(self.embedding(short_term_features), 0)
        output, _ = self.lstm(embedding)
        short_term_enhance = torch.squeeze(output)
        user_embed_dynamic = self.tryone_line2(torch.mean(short_term_enhance, dim=0))  # h_dynamic

        if settings.use_memory_network:
            # Step 2: Generate query
            query = self.query_net(user_embed_dynamic)  # q = f_Q(h_dynamic)
            
            # Step 3: Attention retrieval
            similarity = torch.matmul(query, self.memory_keys.T) / (query.size(-1) ** 0.5)  # s_i
            attention_weights = F.softmax(similarity, dim=-1)  # alpha_i
            memory_repr = torch.matmul(attention_weights, self.memory_values)  # h_memory
            
            # Step 4: Gated fusion (only h_static and h_memory, no h_dynamic)
            combined = torch.cat([user_embed_static, memory_repr], dim=-1)
            gate_s = torch.sigmoid(self.gate_static(combined))  # g_s
            gate_m = torch.sigmoid(self.gate_memory(combined))  # g_m
            gate_sum = gate_s + gate_m
            
            # Step 5: Final fusion
            user_embed = (gate_s * user_embed_static + gate_m * memory_repr) / gate_sum
        else:
            # Original method
            user_embed = self.enhance_val * user_embed_static + (1 - self.enhance_val) * user_embed_dynamic

        # SSL
        if len(neg_sample_list) > 0:
            neg_short_term_states = []
            for neg_day_sample in neg_sample_list:
                neg_trajectory_features = neg_day_sample[0]
                neg_short_term_state = self.encoder(feature_seq=neg_trajectory_features)
                neg_short_term_state = torch.mean(neg_short_term_state, dim=0)
                neg_short_term_states.append(neg_short_term_state)

            short_embed_mean = torch.mean(short_term_state, dim=0)
            long_embed_mean = torch.mean(long_term_catted, dim=0)
            neg_embed_mean = torch.mean(torch.stack(neg_short_term_states), dim=0)
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
