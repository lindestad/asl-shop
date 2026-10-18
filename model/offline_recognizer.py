"""Offline (non-causal) OpenFS recogniser plus the paper's auxiliary attention losses.

Architecture and loss shapes follow arXiv 2602.22949 (OpenFS, CVPR 2026) as published:
3 encoder + 3 decoder layers, d_model 256, 8 heads, feed-forward 2048, GELU, dual-level
sinusoidal encodings (real frame positions plus hand-segment identities) added to the pose
embedding.  Nothing here imports or copies the research-licensed upstream source; the
cross-attention is computed explicitly because the signing-hand-focus and soft-monotonicity
losses need the head-averaged softmax weights, which the scaled-dot-product fast path never
materialises.  CTC is not used; decoding is autoregressive greedy.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.neural import VOCAB

EOS=0  # VOCAB[0] is the empty string; the CTC blank doubles as the autoregressive stop token.
BOS=len(VOCAB)
PAD_CHAR=len(VOCAB)+1
PAD_POSE=-2  # marks padded pose tokens, mirroring the upstream -2 sentinel.
PAD_SEG=32
PAD_FRAME=1024


def encode_text(text):
    return [BOS,*[VOCAB.index(character) for character in text],EOS]


def decode_text(indices):
    result=[]
    for index in indices:
        index=int(index)
        if index in (EOS,PAD_CHAR,BOS):
            break
        result.append(VOCAB[index])
    return ''.join(result)


def positional_encoding_from_idx(indices,d_model):
    """Interleaved sin/cos encoding of explicit position values, so irregular frame gaps
    and hand-segment identities reach the network instead of compressed token order."""
    position=torch.as_tensor(indices,dtype=torch.float32).unsqueeze(-1)
    bands=torch.exp(torch.arange(0,d_model,2,dtype=torch.float32,device=position.device)
                    *(-math.log(10000.)/d_model))
    angles=position*bands
    return torch.stack((angles.sin(),angles.cos()),dim=-1).flatten(-2)


class Attention(nn.Module):
    """Multi-head attention kept explicit: returns the softmax weights averaged over heads
    (pre-dropout, like the reference implementation) next to the projected output."""
    def __init__(self,d_model,heads,dropout):
        super().__init__()
        self.heads=heads
        self.query=nn.Linear(d_model,d_model)
        self.key=nn.Linear(d_model,d_model)
        self.value=nn.Linear(d_model,d_model)
        self.output=nn.Linear(d_model,d_model)
        self.attention_dropout=nn.Dropout(dropout)

    def forward(self,queries,keys,key_padding=None):
        heads=self.heads
        width=queries.shape[-1]//heads
        q=self.query(queries).unflatten(-1,(heads,width)).transpose(1,2)
        k=self.key(keys).unflatten(-1,(heads,width)).transpose(1,2)
        v=self.value(keys).unflatten(-1,(heads,width)).transpose(1,2)
        scores=(q@k.transpose(-2,-1))*(width**-.5)
        if key_padding is not None:
            scores=scores.masked_fill(key_padding.unsqueeze(1).unsqueeze(1),-torch.inf)
        weights=F.softmax(scores,dim=-1)
        attended=(self.attention_dropout(weights)@v).transpose(1,2).flatten(-2)
        return self.output(attended),weights.mean(1)


class DecoderLayer(nn.Module):
    """Post-norm decoder layer in the upstream convention; hands back the cross-attention."""
    def __init__(self,d_model,heads,feedforward,dropout):
        super().__init__()
        self.self_attention=nn.MultiheadAttention(d_model,heads,dropout=dropout,batch_first=True)
        self.cross_attention=Attention(d_model,heads,dropout)
        self.linear1=nn.Linear(d_model,feedforward)
        self.linear2=nn.Linear(feedforward,d_model)
        self.norm1,self.norm2,self.norm3=nn.LayerNorm(d_model),nn.LayerNorm(d_model),nn.LayerNorm(d_model)
        self.attention_dropout=nn.Dropout(dropout)
        self.feedforward_dropout=nn.Dropout(dropout)

    def forward(self,target,memory,causal,target_padding,memory_padding):
        attended,_=self.self_attention(target,target,target,attn_mask=causal,
                                       key_padding_mask=target_padding,need_weights=False)
        x=self.norm1(target+self.attention_dropout(attended))
        attended,weights=self.cross_attention(x,memory,memory_padding)
        x=self.norm2(x+self.attention_dropout(attended))
        return self.norm3(x+self.feedforward_dropout(self.linear2(F.gelu(self.linear1(x))))),weights


class Recognizer(nn.Module):
    """Non-causal pose encoder plus an autoregressive character decoder with cross-attention.
    forward returns teacher-forced logits and (B,layers,chars,tokens) head-averaged
    cross-attention weights; generate decodes greedily."""
    def __init__(self,input_dim=42,d_model=256,heads=8,layers=3,feedforward=2048,dropout=.1):
        super().__init__()
        self.d_model=d_model
        self.pose_embedding=nn.Sequential(nn.Linear(input_dim,d_model),nn.LayerNorm(d_model),nn.GELU(),
                                          nn.Linear(d_model,d_model),nn.LayerNorm(d_model),nn.GELU())
        self.encoder=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model,heads,feedforward,dropout,'gelu',batch_first=True),layers)
        self.char_embedding=nn.Embedding(len(VOCAB)+2,d_model,padding_idx=PAD_CHAR)
        self.decoder=nn.ModuleList([DecoderLayer(d_model,heads,feedforward,dropout) for _ in range(layers)])
        self.lm_head=nn.Sequential(nn.Linear(d_model,128),nn.GELU(),nn.Linear(128,len(VOCAB)))

    def encode(self,poses,segments,positions):
        padding=segments==PAD_SEG
        embedding=(self.pose_embedding(poses)
                   +positional_encoding_from_idx(positions,self.d_model)
                   +positional_encoding_from_idx(segments,self.d_model))
        return self.encoder(embedding,src_key_padding_mask=padding),padding

    def decode(self,memory,padding,inputs):
        length=inputs.shape[1]
        causal=torch.triu(torch.ones(length,length,dtype=torch.bool,device=inputs.device),diagonal=1)
        x=(self.char_embedding(inputs)
           +positional_encoding_from_idx(torch.arange(length,device=inputs.device),self.d_model).unsqueeze(0))
        weights=[]
        for layer in self.decoder:
            x,layer_weights=layer(x,memory,causal,inputs==PAD_CHAR,padding)
            weights.append(layer_weights)
        return self.lm_head(x),torch.stack(weights,dim=1)

    def forward(self,poses,segments,positions,inputs):
        memory,padding=self.encode(poses,segments,positions)
        return self.decode(memory,padding,inputs)

    def generate(self,poses,segments,positions,max_chars=48):
        memory,padding=self.encode(poses,segments,positions)
        steps=torch.full((len(poses),1),BOS,dtype=torch.long,device=memory.device)
        done=torch.zeros(len(poses),dtype=torch.bool,device=memory.device)
        for _ in range(max_chars):
            logits,_=self.decode(memory,padding,steps)
            following=logits[:,-1].argmax(-1)
            following=torch.where(done,torch.full_like(following,PAD_CHAR),following)
            steps=torch.cat((steps,following.unsqueeze(1)),dim=1)
            done=done|(following==EOS)
            if bool(done.all()):
                break
        return steps[:,1:]


def attention_entropy_loss(cross_attns,segments,target_ids):
    """Signing-hand focus (paper S2): mean per-character entropy of cross-attention mass pooled
    into hand-identity segments; low entropy means each emitted character commits to one hand.
    cross_attns is (B,layers,chars,tokens), already head-averaged.  Padded tokens carry seg id
    PAD_SEG but receive zero attention mass once encoder pads are masked, so their entropy term
    is 0 and drops out of the mean; padded character steps are masked by target_ids == -100."""
    average=cross_attns.mean(dim=1).float()
    classes=int(segments.max())+1
    one_hot=F.one_hot(segments,num_classes=classes).float()
    mass=torch.einsum('bts,bsk->btk',average,one_hot)
    entropy=-(mass*(mass+1e-12).log())
    entropy=entropy*(target_ids[:,1:]!=-100).unsqueeze(-1)
    selected=entropy[entropy!=0]
    if selected.numel()==0:
        return average.sum()*0  # keep the autograd path alive when every term is zero
    return selected.mean()


def soft_monotonicity_loss(cross_attns,segments,target_ids):
    """Soft monotonicity (paper S2): penalise the cumulative cross-attention mass moving
    backwards in the input timeline from one emitted character to the next.  ReLU keeps the
    penalty one-sided: forward jumps are free, backward drift costs."""
    steps=target_ids[:,1:]!=-100
    combined=steps.unsqueeze(-1)&(segments!=PAD_SEG).unsqueeze(1)
    valid=combined.sum()
    if not bool(valid):
        return cross_attns.sum()*0
    loss=0.
    for layer in cross_attns.transpose(0,1):
        cumulative=torch.cumsum(layer.float(),dim=2)
        backward=F.relu(cumulative[:,1:,:]-cumulative[:,:-1,:])
        loss=loss+(backward*combined[:,1:]).sum()/valid
    return loss
