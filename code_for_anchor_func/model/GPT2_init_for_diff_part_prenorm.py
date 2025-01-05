import torch
import torch.utils.data as Data
from torch import nn
import numpy as np
import torch.nn.init as init
import math


def get_attn_pad_mask(seq_q, seq_k):
    '''
        seq_q: [batch_size, seq_len]
        seq_k: [batch_size, seq_len]
        seq_len could be src_len or it could be tgt_len
        seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

def get_attn_subsequence_mask(seq, device):
    """
    seq: [batch_size, tgt_len]
    """
    batch_size, tgt_len = seq.size()
    attn_shape = (batch_size, tgt_len, tgt_len)
    subsequence_mask = torch.triu(torch.ones(attn_shape, dtype=torch.uint8, device=device), diagonal=1)
    return subsequence_mask

def attn_mask(X_input, device):
    '''
        X_input: [batch_size, tgt_len]
    '''
    dec_self_attn_pad_mask = get_attn_pad_mask(X_input, X_input) # [batch_size, tgt_len, d_model] 
    dec_self_attn_subsequence_mask = get_attn_subsequence_mask(X_input, device) # [batch_size, tgt_len, d_model]

    dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0) # [batch_size, tgt_len, d_model] 

    return dec_self_attn_mask

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

class GeLU(nn.Module):
    def forward(self, x):
        return gelu(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()

        self.n_head = args.n_heads
        self.d_k = args.d_k
        self.d_v = args.d_v
        self.d_model = args.d_model
    
        self.W_Q = nn.Linear(args.d_model, args.d_k * args.n_heads)
        init.normal_(self.W_Q.weight, mean=0.0, std=(args.d_model)**(-args.qk_std))

        self.W_K = nn.Linear(args.d_model, args.d_k * args.n_heads)
        init.normal_(self.W_K.weight, mean=0.0, std=(args.d_model)**(-args.qk_std))

        self.W_V = nn.Linear(args.d_model, args.d_v * args.n_heads)
        init.normal_(self.W_V.weight, mean=0.0, std=(args.d_model)**(-args.vo_std))

        self.fc = nn.Linear(args.n_heads * args.d_v, args.d_model)
        init.normal_(self.fc.weight, mean=0.0, std=(args.d_model)**(-args.vo_std)/math.sqrt(2 * args.n_layers))
        self.layernorm = nn.LayerNorm(args.d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        input_Q=self.layernorm(input_Q)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # scores : [batch_size, n_heads, len_q, len_k]
        attn = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        masked_attn = attn.masked_fill(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        softmax_attn = nn.Softmax(dim=-1)(masked_attn)
        qkv = torch.matmul(softmax_attn, V)  # [batch_size, n_heads, len_q, d_v] 

        qkv_out = qkv.transpose(1, 2).reshape(batch_size, -1, self.n_head * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(qkv_out)  # [batch_size, len_q, d_model]

        return output + residual, softmax_attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(args.d_model, args.d_feedforward),
            nn.GELU(),
            nn.Linear(args.d_feedforward, args.d_model)
        )


        self.layernorm=nn.LayerNorm(args.d_model)

        # Initialize weights of Linear layers
        for i, layer in enumerate(self.fc):
            if isinstance(layer, nn.Linear):
                if i == len(self.fc) - 1:  # For the last Linear layer
                    init.normal_(layer.weight, mean=0.0, std=(args.d_model)**(-args.mlp_std) / math.sqrt(2 * args.n_layers))
                else:
                    init.normal_(layer.weight, mean=0.0, std=(args.d_model)**(-args.mlp_std))


    def forward(self, hidden_state):
        '''
        hidden_state: [batch_size, seq_len, d_model]
        '''
        residual = hidden_state
        hidden_state=self.layernorm(hidden_state)
        output = self.fc(hidden_state)
        return output + residual # [batch_size, seq_len, d_model]

class DecoderLayer(nn.Module):
    def __init__(self, args):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self, hidden_state, dec_self_attn_mask):
        '''
            hidden_state: [batch_size, tgt_len, d_model]
            dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        '''
        # Attention层
        # hidden_state: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        hidden_state, dec_self_attn = self.dec_self_attn(hidden_state, hidden_state, hidden_state, dec_self_attn_mask)

        # 非线性层
        hidden_state = self.pos_ffn(hidden_state)  # [batch_size, tgt_len, d_model]
        return hidden_state, dec_self_attn


class Decoder(nn.Module):
    def __init__(self, args, device):
        super(Decoder, self).__init__()
        self.device = device
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layers)])

    def forward(self, hidden_state, dec_self_attn_mask):
        '''
            hidden_state: [batch_size, tgt_len]
        '''
        dec_self_attns = []
        for layer in self.layers:
            # hidden_state: [batch_size, tgt_len, d_model]
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
            hidden_state, dec_self_attn = layer(hidden_state, dec_self_attn_mask)
   
            dec_self_attns.append(dec_self_attn)

        return hidden_state, dec_self_attns

class Embedding(nn.Module):
    def __init__ (self, args, device):
        super(Embedding, self).__init__()
        self.device = device
        self.tgt_emb = nn.Embedding(args.vocab_size, args.d_model)
        self.pos_emb = nn.Embedding(args.max_pos, args.d_model)

        init.normal_(self.tgt_emb.weight, mean=0.0, std=(args.d_model)**(-args.embedding_std))
        init.normal_(self.pos_emb.weight, mean=0.0, std=(args.d_model)**(-args.embedding_std))



    def forward(self, X_input):
        seq_len = X_input.size(1)
        pos = torch.arange(seq_len, dtype = torch.long, device = self.device)
        pos = pos.unsqueeze(0).expand_as(X_input)

        tgt_emb = self.tgt_emb(X_input)
        pos_emb = self.pos_emb(pos)
        emb = tgt_emb + pos_emb

        return emb

class myGPT2_init_for_diff_part_prenorm(nn.Module):
    def __init__(self, args, device):
        super(myGPT2_init_for_diff_part_prenorm, self).__init__()

        self.device = device
        self.embedding = Embedding(args, device)
        self.decoder = Decoder(args, device)
        self.projection = nn.Linear(args.d_model, args.vocab_size)
        self.layernorm = nn.LayerNorm(args.d_model)

        init.normal_(self.projection.weight, mean=0.0, std=(args.d_model)**(-args.embedding_std))
        # init.normal_(self.projection.bias, mean=0.0, std=1e-3)


    

    def forward(self, X_input):
        """
            dec_inputs: [batch_size, tgt_len]
        """
        hidden_state = self.embedding(X_input)

        dec_self_attn_mask = attn_mask(X_input, self.device)

        hidden_state, dec_self_attns = self.decoder(hidden_state, dec_self_attn_mask)
        hidden_state=self.layernorm(hidden_state)
        dec_logits = self.projection(hidden_state)
        
        return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns
    

    def greedy_decoder(self,dec_input):

        projected, _ = self.forward(dec_input)

        projected = projected[-1,:].argmax()
        next_word = projected.item() 

        return next_word


    def test(self,sentence):
        dec_input = torch.tensor(sentence, dtype=torch.long, device=self.device).unsqueeze(0)

        output = self.greedy_decoder(dec_input)

        return output




