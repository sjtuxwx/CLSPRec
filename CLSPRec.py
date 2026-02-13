import torch
from torch import nn

import settings

device = settings.gpuId if torch.cuda.is_available() else 'cpu'


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
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads

        assert (
                self.head_dim * self.heads == self.embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, query, keys_values_projected=False):
        value_len, key_len, query_len = values.shape[0], keys.shape[0], query.shape[0]

        # 如果K/V已经投影，跳过投影步骤
        if keys_values_projected:
            # values和keys已经是 [seq_len, heads, head_dim]
            pass
        else:
            values = self.values(values)
            keys = self.keys(keys)
            values = values.reshape(value_len, self.heads, self.head_dim)
            keys = keys.reshape(key_len, self.heads, self.head_dim)
        
        queries = self.queries(query)
        queries = queries.reshape(query_len, self.heads, self.head_dim)

        energy = torch.einsum("qhd,khd->hqk", [queries, keys])

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)

        out = torch.einsum("hql,lhd->qhd", [attention, values]).reshape(
            query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        return out
    
    def get_keys(self, x):
        """生成投影后的keys"""
        keys = self.keys(x)
        key_len = keys.shape[0]
        keys = keys.reshape(key_len, self.heads, self.head_dim)
        return keys
    
    def get_values(self, x):
        """生成投影后的values"""
        values = self.values(x)
        value_len = values.shape[0]
        values = values.reshape(value_len, self.heads, self.head_dim)
        return values


class HSTUAttention(nn.Module):
    """
    HSTU Attention mechanism based on the formula:
    U(X), V(X), Q(X), K(X) = Split(φ1(f1(X)))
    A(X)V(X) = φ2(Q(X)K(X)^T + rab^{p,t})V(X)
    Y(X) = f2(Norm(A(X)V(X)) ⊙ U(X))
    """
    def __init__(self, embed_size, heads, max_seq_len=100):
        super(HSTUAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads
        self.max_seq_len = max_seq_len

        assert (
                self.head_dim * self.heads == self.embed_size
        ), "Embedding size needs to be divisible by heads"

        # φ1: Linear projection + SiLU activation
        self.phi1_linear = nn.Linear(self.embed_size, 4 * self.embed_size, bias=False)
        self.phi1_activation = nn.SiLU()  # SiLU/Swish activation
        
        # Relative position bias rab^{p,t}: 包含位置和时间信息
        # [heads, max_seq_len, max_seq_len] - 每个head学习不同的位置-时间偏置模式
        self.relative_position_bias = nn.Parameter(
            torch.zeros(self.heads, max_seq_len, max_seq_len)
        )
        
        # φ2: SiLU activation for attention scores
        self.phi2_activation = nn.SiLU()
        
        # f2: final projection
        self.f2 = nn.Linear(self.embed_size, self.embed_size)
        
        # Layer norm
        self.norm = nn.LayerNorm(self.embed_size)

    def forward(self, values, keys, query, keys_values_projected=False):
        """
        For encoder self-attention: values = keys = query
        keys_values_projected: if True, keys and values are already projected [seq_len, heads, head_dim]
        """
        seq_len = query.shape[0]
        
        # Apply φ1 (Linear + SiLU) to query and split into U, V, Q, K
        uvqk = self.phi1_linear(query)  # [seq_len, 4 * embed_size]
        uvqk = self.phi1_activation(uvqk)  # Apply SiLU activation
        u, v_query, q, k_query = torch.chunk(uvqk, 4, dim=-1)  # each: [seq_len, embed_size]
        
        # Reshape U and Q
        u = u.reshape(seq_len, self.heads, self.head_dim)
        q = q.reshape(seq_len, self.heads, self.head_dim)
        
        # 如果K/V已经投影，使用传入的；否则使用从query生成的
        if keys_values_projected:
            # keys和values已经是 [seq_len, heads, head_dim]
            k = keys
            v = values
        else:
            # 使用从query生成的K/V（self-attention情况）
            v = v_query.reshape(seq_len, self.heads, self.head_dim)
            k = k_query.reshape(seq_len, self.heads, self.head_dim)
        
        # Transpose for matrix multiplication: [heads, seq_len, head_dim]
        q = q.permute(1, 0, 2)  # [heads, query_len, head_dim]
        k = k.permute(1, 0, 2)  # [heads, kv_len, head_dim]
        v = v.permute(1, 0, 2)  # [heads, kv_len, head_dim]
        
        # Compute Q(X)K(X)^T: [heads, query_len, head_dim] @ [heads, head_dim, kv_len] -> [heads, query_len, kv_len]
        energy = torch.bmm(q, k.transpose(1, 2))  # [heads, query_len, kv_len]
        
        # Scale by sqrt(head_dim) for numerical stability
        energy = energy / (self.head_dim ** 0.5)
        
        # Add relative position-time bias rab^{p,t}
        # Extract the corresponding bias for current sequence length
        kv_len = k.shape[1]  # Get actual kv sequence length
        rel_bias = self.relative_position_bias[:, :seq_len, :kv_len]  # [heads, query_len, kv_len]
        energy = energy + rel_bias
        
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
    
    def get_keys(self, x):
        """生成投影后的keys"""
        seq_len = x.shape[0]
        uvqk = self.phi1_linear(x)
        uvqk = self.phi1_activation(uvqk)
        _, _, _, k = torch.chunk(uvqk, 4, dim=-1)
        k = k.reshape(seq_len, self.heads, self.head_dim)
        return k
    
    def get_values(self, x):
        """生成投影后的values"""
        seq_len = x.shape[0]
        uvqk = self.phi1_linear(x)
        uvqk = self.phi1_activation(uvqk)
        _, v, _, _ = torch.chunk(uvqk, 4, dim=-1)
        v = v.reshape(seq_len, self.heads, self.head_dim)
        return v


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, use_hstu=False, max_seq_len=100):
        super(EncoderBlock, self).__init__()
        self.embed_size = embed_size
        self.use_hstu = use_hstu
        
        # Choose attention mechanism
        if use_hstu:
            self.attention = HSTUAttention(self.embed_size, heads, max_seq_len=max_seq_len)
        else:
            self.attention = SelfAttention(self.embed_size, heads)
        
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, forward_expansion * self.embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * self.embed_size, self.embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, keys_values_projected=False):
        attention = self.attention(value, key, query, keys_values_projected=keys_values_projected)  # [len * embed_size]

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

    def forward(self, feature_seq, return_kv_cache=False, long_term_kv_cache=None):
        embedding = self.embedding_layer(feature_seq)  # [len, embedding]
        out = self.dropout(embedding)
        
        new_kv_cache = [] if return_kv_cache else None

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case
        for i, layer in enumerate(self.layers):
            # 如果需要返回K/V cache（长期序列），保存当前层的投影K/V
            if return_kv_cache:
                k = layer.attention.get_keys(out)
                v = layer.attention.get_values(out)
                new_kv_cache.append((k, v))
            
            # 如果提供了长期K/V cache（短期序列），使用已投影的K/V
            if long_term_kv_cache is not None:
                k_long, v_long = long_term_kv_cache[i]
                
                # 生成短期的K/V
                k_short = layer.attention.get_keys(out)
                v_short = layer.attention.get_values(out)
                
                # 拼接
                k_combined = torch.cat([k_short, k_long], dim=0)
                v_combined = torch.cat([v_short, v_long], dim=0)
                
                # 传入已投影的K/V
                out = layer(v_combined, k_combined, out, keys_values_projected=True)
            else:
                # 正常self-attention
                out = layer(out, out, out, keys_values_projected=False)
        
        if return_kv_cache:
            return out, new_kv_cache
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
            enable_long_short_cross_attention=False,
            enable_layerwise_cross_attention=False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.total_embed_size = f_embed_size * 5
        self.enable_long_short_cross_attention = enable_long_short_cross_attention
        self.enable_layerwise_cross_attention = enable_layerwise_cross_attention

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

        self.loss_func = nn.CrossEntropyLoss()

        self.tryone_line2 = nn.Linear(self.total_embed_size, f_embed_size)
        self.enhance_val = nn.Parameter(torch.tensor(0.5))
        self.enable_cross_day_attention = enable_cross_day_attention

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
        user_id = short_term_sequence[0][2, 0]

        # Random mask long-term sequences
        long_term_sequences = self.feature_mask(long_term_sequences, settings.mask_prop)

        # Long-term
        if not self.enable_cross_day_attention:
            # 方式1: 每天独立编码（天内attention）
            long_term_out = []
            long_term_kv_cache = None if not self.enable_layerwise_cross_attention else []
            
            for seq in long_term_sequences:
                if self.enable_layerwise_cross_attention:
                    output, kv_cache = self.encoder(feature_seq=seq[0], return_kv_cache=True)
                    long_term_out.append(output)
                    
                    # 累积K/V
                    if len(long_term_kv_cache) == 0:
                        long_term_kv_cache = kv_cache
                    else:
                        for i in range(len(kv_cache)):
                            k_new, v_new = kv_cache[i]
                            k_old, v_old = long_term_kv_cache[i]
                            long_term_kv_cache[i] = (
                                torch.cat([k_old, k_new], dim=0),
                                torch.cat([v_old, v_new], dim=0)
                            )
                else:
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
            if self.enable_layerwise_cross_attention:
                long_term_catted, long_term_kv_cache = self.encoder(
                    feature_seq=long_term_features_concat,
                    return_kv_cache=True
                )
            else:
                long_term_catted = self.encoder(feature_seq=long_term_features_concat)
                long_term_kv_cache = None
            

        # Short-term
        if self.enable_layerwise_cross_attention:
            # 层级交互
            short_term_state = self.encoder(
                feature_seq=short_term_features,
                long_term_kv_cache=long_term_kv_cache
            )
        elif self.enable_long_short_cross_attention:
            # 最后一层交互（原有逻辑）
            short_term_self_encoded = self.encoder(feature_seq=short_term_features)
            cross_attn_out = self.long_short_cross_attention(
                values=long_term_catted,
                keys=long_term_catted,
                query=short_term_self_encoded
            )
            short_term_attended = self.cross_attn_dropout(
                self.cross_attn_norm(cross_attn_out + short_term_self_encoded)
            )
            ffn_out = self.cross_attn_ffn(short_term_attended)
            short_term_state = self.cross_attn_dropout(
                self.cross_attn_norm2(ffn_out + short_term_attended)
            )
        else:
            # 独立编码
            short_term_state = self.encoder(feature_seq=short_term_features)

        # User enhancement
        user_embed = self.embedding.user_embed(user_id)
        embedding = torch.unsqueeze(self.embedding(short_term_features), 0)
        output, _ = self.lstm(embedding)
        short_term_enhance = torch.squeeze(output)
        user_embed = self.enhance_val * user_embed + (1 - self.enhance_val) * self.tryone_line2(
            torch.mean(short_term_enhance, dim=0))

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

        label = torch.unsqueeze(target, 0)
        pred = torch.unsqueeze(output, 0)

        pred_loss = self.loss_func(pred, label)
        loss = pred_loss + ssl_loss * settings.neg_weight
        return loss, output

    def predict(self, sample, neg_sample_list):
        _, pred_raw = self.forward(sample, neg_sample_list)
        ranking = torch.sort(pred_raw, descending=True)[1]
        target = sample[-1][0][0, -1]

        return ranking, target
