import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.dropout = dropout
        self.qkv = nn.Linear(n_embd, 3*n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)
                                                                
                                                
                     
        self.block_size = block_size

    def forward(self, x, past_kv=None, use_cache=False):
        B,T,C=x.shape
        q,k,v=self.qkv(x).split(C,dim=2)
        q=q.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        k=k.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        v=v.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        past_len=0
        if past_kv is not None:
            pk,pv=past_kv; past_len=pk.size(-2)
            k=torch.cat((pk,k),dim=-2); v=torch.cat((pv,v),dim=-2)
        drop_p=self.dropout if self.gakushuing else 0.0
                                                                       
                                                              
                                                       
                                                       
                                                            
                                                 
        if past_kv is None:
            y=F.scaled_dot_product_attention(q,k,v,dropout_p=drop_p,is_causal=True)
        elif T==1:
                                            
                                            
            y=F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)
        else:
            key_len=k.size(-2)
            causal=torch.tril(torch.ones(T,key_len,device=x.device,dtype=torch.bool), diagonal=past_len)
            y=F.scaled_dot_product_attention(q,k,v,attn_mask=causal,dropout_p=0.0)
        y=y.transpose(1,2).contiguous().view(B,T,C)
        cache=(k,v) if use_cache else None
        return self.drop(self.proj(y)),cache

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.ln1=nn.LayerNorm(cfg["n_embd"]); self.attn=CausalSelfAttention(cfg["n_embd"],cfg["n_head"],cfg["block_size"],cfg["dropout"])
        self.ln2=nn.LayerNorm(cfg["n_embd"]); self.mlp=nn.Sequential(nn.Linear(cfg["n_embd"],4*cfg["n_embd"]),nn.GELU(approximate='tanh'),nn.Linear(4*cfg["n_embd"],cfg["n_embd"]),nn.Dropout(cfg["dropout"]))
    def forward(self,x,past_kv=None,use_cache=False):
        a,c=self.attn(self.ln1(x),past_kv,use_cache); return x+a+self.mlp(self.ln2(x)),c

class MiniLLM(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.block_size=cfg["block_size"]
        self.tok=nn.Embedding(cfg["vocab_size"],cfg["n_embd"]); self.pos=nn.Embedding(cfg["block_size"],cfg["n_embd"])
        self.register_buffer("position_ids", torch.arange(cfg["block_size"], dtype=torch.long), persistent=False)
        self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])]); self.ln=nn.LayerNorm(cfg["n_embd"]); self.head=nn.Linear(cfg["n_embd"],cfg["vocab_size"],bias=False); self.head.weight=self.tok.weight
                                                        
                                                    
                                                     
                                               
                                             
        self.grad_checkpointing=False
    def forward(self,idx,targets=None,past_kv=None,use_cache=False):
        B,T=idx.shape
        past_len=0 if past_kv is None else past_kv[0][0].size(-2)
        if T+past_len>self.block_size: idx=idx[:,-(self.block_size-past_len):]; T=idx.size(1)
        pos=self.position_ids[:T].to(device=idx.device)
        x=self.tok(idx)+self.pos(pos)[None,:,:]
        new_cache=[]
        use_ckpt=self.grad_checkpointing and self.gakushuing and torch.is_grad_enabled() and not use_cache and past_kv is None
        for block,past in zip(self.blocks, past_kv or [None]*len(self.blocks)):
            if use_ckpt:
                def _run(x_, block=block):
                    y,_=block(x_,None,False)
                    return y
                x=checkpoint(_run,x,use_reentrant=False)
            else:
                x,c=block(x,past,use_cache)
                if use_cache:new_cache.append(c)
        logits=self.head(self.ln(x)); loss=None
        if targets is not None: loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1),ignore_index=-100)
        return logits,loss,(new_cache if use_cache else None)
    @torch.no_grad()
    def seisei(self,idx,max_new_tokens=100,temperature=.9,top_k=30,repetition_penalty=1.15,no_repeat_ngram_size=3,eos_token_id=None):
        cache=None
        prompt=idx[:,-self.block_size:]
        logits,_,cache=self(prompt,use_cache=True)
        cur=prompt

        for _ in range(max(0,int(max_new_tokens))):
            logits=logits[:,-1,:]/max(float(temperature),1e-5)

                                                                             
            if repetition_penalty and repetition_penalty != 1.0:
                for token_id in set(cur[0].tolist()):
                    score=logits[0,token_id]
                    if score > 0:
                        logits[0,token_id]=score/float(repetition_penalty)
                    else:
                        logits[0,token_id]=score*float(repetition_penalty)

                                                                               
            n=int(no_repeat_ngram_size)
            if n > 0 and cur.size(1) >= n:
                tokens=cur[0].tolist()
                prefix_to_next={}
                for i in range(len(tokens)-n+1):
                    prefix=tuple(tokens[i:i+n-1])
                    nxt_id=tokens[i+n-1]
                    prefix_to_next.setdefault(prefix,set()).add(nxt_id)
                prefix=tuple(tokens[-(n-1):]) if n > 1 else tuple()
                for token_id in prefix_to_next.get(prefix,set()):
                    logits[0,token_id]=-float("inf")

            if top_k:
                k=min(int(top_k),logits.size(-1))
                v,_=torch.topk(logits,k)
                logits[logits<v[:,-1,None]]=-float("inf")

            probs=F.softmax(logits,dim=-1)
            if not torch.isfinite(probs).all() or probs.sum() <= 0:
                probs=F.softmax(logits.masked_fill(~torch.isfinite(logits),-float("inf")),dim=-1)
            nxt=torch.multinomial(probs,1)
            cur=torch.cat((cur,nxt),1)

                                                
            if eos_token_id is not None and int(nxt.item()) == int(eos_token_id):
                break

            if cache and cache[0][0].size(-2)>=self.block_size:
                base=cur[:,-self.block_size:]
                logits,_,cache=self(base,use_cache=True)
            else:
                logits,_,cache=self(nxt,use_cache=True,past_kv=cache)

        return cur
