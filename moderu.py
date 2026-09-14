import math 
import torch 
import torch .nn as nn 
import torch .nn .functional as F 
from torch .utils .checkpoint import checkpoint 

class RMSNorm (nn .Module ):
    def __init__ (self ,dim ,eps =1e-6 ):
        super ().__init__ ()
        self .weight =nn .Parameter (torch .ones (dim ))
        self .eps =eps 
    def forward (self ,x ):
        return x *torch .rsqrt (x .pow (2 ).mean (-1 ,keepdim =True )+self .eps )*self .weight 

class CausalSelfAttention (nn .Module ):
    def __init__ (self ,n_embd ,n_head ,block_size ,dropout ):
        super ().__init__ ()
        assert n_embd %n_head ==0 
        self .n_head =n_head 
        self .head_dim =n_embd //n_head 
        self .dropout =dropout 
        self .qkv =nn .Linear (n_embd ,3 *n_embd ,bias =False )
        self .proj =nn .Linear (n_embd ,n_embd ,bias =False )
        self .q_norm =RMSNorm (self .head_dim )
        self .k_norm =RMSNorm (self .head_dim )
        self .out_gate =nn .Linear (n_embd ,n_embd ,bias =True )
        self .drop =nn .Dropout (dropout )
        self .block_size =block_size 
        half =self .head_dim //2 
        inv =1.0 /(10000.0 **(torch .arange (0 ,half ,dtype =torch .float32 )/max (1 ,half )))
        pos =torch .arange (block_size ,dtype =torch .float32 )[:,None ]
        freq =pos *inv [None ,:]
        self .register_buffer ('rope_cos',freq .cos (),persistent =False )
        self .register_buffer ('rope_sin',freq .sin (),persistent =False )

    def _rope (self ,x ,start ):
        if self .head_dim %2 :
            return x 
        t =x .size (-2 )
        cos =self .rope_cos [start :start +t ].to (device =x .device ,dtype =x .dtype )[None ,None ,:,:]
        sin =self .rope_sin [start :start +t ].to (device =x .device ,dtype =x .dtype )[None ,None ,:,:]
        a ,b =x [...,::2 ],x [...,1 ::2 ]
        out =torch .empty_like (x )
        out [...,::2 ]=a *cos -b *sin 
        out [...,1 ::2 ]=a *sin +b *cos 
        return out 

    def forward (self ,x ,past_kv =None ,use_cache =False ):
        B ,T ,C =x .shape 
        q ,k ,v =self .qkv (x ).split (C ,dim =2 )
        q =q .view (B ,T ,self .n_head ,self .head_dim ).transpose (1 ,2 )
        k =k .view (B ,T ,self .n_head ,self .head_dim ).transpose (1 ,2 )
        v =v .view (B ,T ,self .n_head ,self .head_dim ).transpose (1 ,2 )
        past_len =0 
        if past_kv is not None :
            pk ,pv =past_kv 
            past_len =pk .size (-2 )
        q =self .q_norm (q )
        k =self .k_norm (k )
        q =self ._rope (q ,past_len )
        k =self ._rope (k ,past_len )
        if past_kv is not None :
            k =torch .cat ((pk ,k ),dim =-2 )
            v =torch .cat ((pv ,v ),dim =-2 )
        drop_p =self .dropout if self .training else 0.0 
        if past_kv is None :
            y =F .scaled_dot_product_attention (q ,k ,v ,dropout_p =drop_p ,is_causal =True )
        elif T ==1 :
            y =F .scaled_dot_product_attention (q ,k ,v ,dropout_p =0.0 ,is_causal =False )
        else :
            key_len =k .size (-2 )
            causal =torch .tril (torch .ones (T ,key_len ,device =x .device ,dtype =torch .bool ),diagonal =past_len )
            y =F .scaled_dot_product_attention (q ,k ,v ,attn_mask =causal ,dropout_p =0.0 )
        y =y .transpose (1 ,2 ).contiguous ().view (B ,T ,C )
        y =y *torch .sigmoid (self .out_gate (x ))
        cache =(k ,v )if use_cache else None 
        return self .drop (self .proj (y )),cache 

class SwiGLU (nn .Module ):
    def __init__ (self ,n_embd ):
        super ().__init__ ()
        hidden =max (256 ,int (8 *n_embd /3 ))
        hidden =((hidden +255 )//256 )*256 
        self .gate =nn .Linear (n_embd ,hidden ,bias =False )
        self .up =nn .Linear (n_embd ,hidden ,bias =False )
        self .down =nn .Linear (hidden ,n_embd ,bias =False )
    def forward (self ,x ):
        return self .down (F .silu (self .gate (x ))*self .up (x ))

class Block (nn .Module ):
    def __init__ (self ,cfg ):
        super ().__init__ ()
        d =cfg ['n_embd']
        self .ln1 =RMSNorm (d )
        self .attn =CausalSelfAttention (d ,cfg ['n_head'],cfg ['block_size'],cfg ['dropout'])
        self .ln2 =RMSNorm (d )
        self .mlp =SwiGLU (d )
        self .drop =nn .Dropout (cfg ['dropout'])
    def forward (self ,x ,past_kv =None ,use_cache =False ):
        a ,c =self .attn (self .ln1 (x ),past_kv ,use_cache )
        x =x +a 
        x =x +self .drop (self .mlp (self .ln2 (x )))
        return x ,c 

class PrefixMemory (nn .Module ):
    def __init__ (self ,dim):
        super ().__init__ ()
        self .norm =RMSNorm (dim )
        self .proj =nn .Linear (dim ,dim ,bias =False )
        self .gate =nn .Linear (dim *2 ,dim ,bias =True )
    def forward (self ,x ):
        t=x .size (1 )
        if t<=1 :
            return x
        acc=x .cumsum (1 )
        denom=torch .arange (1 ,t +1 ,device =x .device ,dtype =x .dtype ).view (1 ,-1 ,1 )
        mem=self .proj (self .norm (acc /denom ))
        g=torch .sigmoid (self .gate (torch .cat ((x ,mem ),-1 )))
        return x +g *mem

class ThinkingRefiner (nn .Module ):
    def __init__ (self ,dim ):
        super ().__init__ ()
        self .norm =RMSNorm (dim )
        self .ff =SwiGLU (dim )
        self .gate =nn .Linear (dim ,dim ,bias =True )
    def forward (self ,x ):
        z=self .ff (self .norm (x ))
        return x +torch .sigmoid (self .gate (x ))*z

class LatentWorkspace(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.norm=RMSNorm(dim)
        self.query=nn.Parameter(torch.randn(1,1,dim)*0.02)
        self.k=nn.Linear(dim,dim,bias=False)
        self.v=nn.Linear(dim,dim,bias=False)
        self.gate=nn.Linear(dim*2,dim,bias=True)
        self.refine=SwiGLU(dim)
        self.out=nn.Linear(dim,dim,bias=False)
    def forward(self,x,state=None):
        z=self.norm(x)
        q=self.query.expand(x.size(0),-1,-1) if state is None else state
        score=torch.matmul(q,self.k(z).transpose(-1,-2))/math.sqrt(x.size(-1))
        att=F.softmax(score,dim=-1)
        pooled=torch.matmul(att,self.v(z))
        pooled=self.out(pooled)
        gate=torch.sigmoid(self.gate(torch.cat((q,pooled),-1)))
        state=q+gate*pooled
        state=state+self.refine(self.norm(state))
        mix=state.expand(-1,x.size(1),-1)
        gx=torch.sigmoid(self.gate(torch.cat((x,mix),-1)))
        return x+gx*mix,state

class MiniLLM (nn .Module ):
    architecture ='rope-rms-qk-gated-swiglu-prefixmemory-refine-v3'
    def __init__ (self ,cfg ):
        super ().__init__ ()
        self .block_size =cfg ['block_size']
        self .tok =nn .Embedding (cfg ['vocab_size'],cfg ['n_embd'])
        self .blocks =nn .ModuleList ([Block (cfg )for _ in range (cfg ['n_layer'])])
        self .ln =RMSNorm (cfg ['n_embd'])
        self .head =nn .Linear (cfg ['n_embd'],cfg ['vocab_size'],bias =False )
        self .head .weight =self .tok .weight 
        self .grad_checkpointing =False 
        self .reasoning_aux_weight =float (cfg .get ('reasoning_aux_weight',0.12 ))
        self .reasoning_horizon =int (cfg .get ('reasoning_horizon',2 ))
        self .thinking_passes =max (1 ,int (cfg .get ('thinking_passes',2 )))
        self .prefix_memory =PrefixMemory (cfg ['n_embd'])
        self .refiner =ThinkingRefiner (cfg ['n_embd'])
        self .workspace =LatentWorkspace (cfg ['n_embd'])
        self .semantic_aux_weight=float (cfg .get ('semantic_aux_weight',0.08))
        self .semantic_head=nn.Linear(cfg ['n_embd'],cfg ['n_embd'],bias=False)

    def forward (self ,idx ,targets =None ,past_kv =None ,use_cache =False ):
        B ,T =idx .shape 
        past_len =0 if past_kv is None else past_kv [0 ][0 ].size (-2 )
        if T +past_len >self .block_size :
            idx =idx [:,-(self .block_size -past_len ):]
            T =idx .size (1 )
        pos_start =past_len 
        x =self .tok (idx )
        new_cache =[]
        use_ckpt =self .grad_checkpointing and self .training and torch .is_grad_enabled ()and not use_cache and past_kv is None 
        for block ,past in zip (self .blocks ,past_kv or [None ]*len (self .blocks )):
            if use_ckpt :
                def _run (x_ ,block =block ):
                    y ,_ =block (x_ ,None ,False )
                    return y 
                x =checkpoint (_run ,x ,use_reentrant =False )
            else :
                x ,c =block (x ,past ,use_cache )
                if use_cache :new_cache .append (c )
        x=self .prefix_memory (x )
        for _ in range (self .thinking_passes -1 ):
            x=self .refiner (x )
        logits =self .head (self .ln (x ))
        loss =None 
        if targets is not None :
            main =F .cross_entropy (logits .reshape (-1 ,logits .size (-1 )),targets .reshape (-1 ),ignore_index =-100 ,label_smoothing =float (self .reasoning_aux_weight *0.08 ))
            loss =main 
            w =self .reasoning_aux_weight 
            h =min (max (2 ,self .reasoning_horizon ),T -1 )
            if w >0 and h >=2 :
                for n in range (2 ,h +1 ):
                    a =logits [:,:T -n +1 ].reshape (-1 ,logits .size (-1 ))
                    b =targets [:,n -1 :T ].reshape (-1 )
                    aux =F .cross_entropy (a ,b ,ignore_index =-100 )
                    loss =loss +w /(h -1 )*aux 
        return logits ,loss ,(new_cache if use_cache else None )

    @torch .no_grad ()
    def generate_with_score (self ,idx ,max_new_tokens =100 ,temperature =.9 ,top_k =30 ,repetition_penalty =1.15 ,no_repeat_ngram_size =3 ,eos_token_id =None ):
        cache =None
        prompt =idx [:,-self .block_size :]
        logits ,_ ,cache =self (prompt ,use_cache =True )
        cur =prompt
        score_sum =0.0
        generated =0
        seen_tokens =set(cur[0].tolist())
        n =int(no_repeat_ngram_size)
        for _ in range(max(0,int(max_new_tokens))):
            logits =logits[:,-1,:]/max(float(temperature),1e-5)
            if repetition_penalty and repetition_penalty !=1.0 and seen_tokens:
                ids=torch.tensor(list(seen_tokens),device=logits.device,dtype=torch.long)
                vals=logits[0,ids]
                logits[0,ids]=torch.where(vals>0,vals/float(repetition_penalty),vals*float(repetition_penalty))
            if n>0 and cur.size(1)>=n:
                tail=cur[0].tolist()[-(n-1):] if n>1 else []
                tokens=cur[0].tolist()
                banned=set()
                start=max(0,len(tokens)-256-n)
                prefix=tuple(tail)
                for i in range(start,len(tokens)-n+1):
                    if tuple(tokens[i:i+n-1])==prefix:
                        banned.add(tokens[i+n-1])
                if banned:
                    ids=torch.tensor(list(banned),device=logits.device,dtype=torch.long)
                    logits[0,ids]=-float('inf')
            if top_k:
                k=min(int(top_k),logits.size(-1))
                v,_=torch.topk(logits,k)
                logits.masked_fill_(logits<v[:,-1,None],-float('inf'))
            probs=F.softmax(logits,dim=-1)
            if not torch.isfinite(probs).all() or float(probs.sum())<=0:
                probs=F.softmax(logits.masked_fill(~torch.isfinite(logits),-float('inf')),dim=-1)
            nxt=torch.multinomial(probs,1)
            score_sum += float(torch.log(probs.gather(1,nxt).clamp_min(1e-12)).item())
            generated += 1
            tid=int(nxt.item())
            seen_tokens.add(tid)
            cur=torch.cat((cur,nxt),1)
            if eos_token_id is not None and tid==int(eos_token_id):break
            if cache and cache[0][0].size(-2)>=self.block_size:
                base=cur[:,-self.block_size:]
                logits,_,cache=self(base,use_cache=True)
            else:
                logits,_,cache=self(nxt,use_cache=True,past_kv=cache)
        avg_score=score_sum/max(1,generated)
        return cur,avg_score

    @torch.no_grad()
    def generate (self ,idx ,max_new_tokens =100 ,temperature =.9 ,top_k =30 ,repetition_penalty =1.15 ,no_repeat_ngram_size =3 ,eos_token_id =None ):
        return self.generate_with_score(idx,max_new_tokens,temperature,top_k,repetition_penalty,no_repeat_ngram_size,eos_token_id)[0]
