import json ,os ,random ,math ,hashlib ,time ,gc ,contextlib ,socket ,concurrent .futures 
from pathlib import Path 


from cpu_kasoku import configure_environment ,configure_torch_threads ,IS_INTEL_CPU ,IPEX 
configure_environment ()
import torch 
from torch .utils .data import Dataset ,DataLoader 
from bunsho_bunkatsu import SentencePieceTokenizer 
from moderu import MiniLLM 


os .environ .setdefault ('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')




if not torch .cuda .is_available ():
    configure_torch_threads (torch )

CFG =json .load (open ('config.json',encoding ='utf8'))
CORPUS =Path ('data/pretrain.txt');WIKI =Path ('data/wikipedia.txt');DATA =Path ('data/initial_ja.jsonl')
DISCORD =Path ('data/discord.jsonl');CONV =Path ('data/discord_conversations.jsonl')
CKPT =Path ('data/moderu.pt');TOK =Path ('data/bunsho_bunkatsu.moderu');CACHE =Path ('data/token_cache.pt')
DISCORD_CACHE =Path ('data/discord_token_cache.pt')


def load_records ():
    out =[]
    for p in (DATA ,DISCORD ,CONV ):
        if not p .exists ():continue 
        for line in p .read_text (encoding ='utf8').splitlines ():
            try :
                o =json .loads (line );t =o .get ('text')or o .get ('input')
                if t :out .append ((str (t ),p .name ))
            except json .JSONDecodeError :pass 
    return out 


def load_corpus_text (max_chars =0 ):
    parts =[];remaining =max (0 ,int (max_chars ))
    for p in (CORPUS ,WIKI ):
        if not p .exists ():continue 
        if remaining :
            text =p .read_text (encoding ='utf8',errors ='ignore')[:remaining ]
            remaining -=len (text )
        else :
            text =p .read_text (encoding ='utf8',errors ='ignore')
        if text :parts .append (text )
        if remaining ==0 and max_chars :break 
    return '\n'.join (parts ).strip ()

def corpus_char_count ():
    return sum (p .stat ().st_size for p in (CORPUS ,WIKI )if p .exists ())

class ChunkDataset (Dataset ):
    def __init__ (self ,sequences ,block_size ,pack =True ):
        self .items =[]
        if not sequences :return 
        if pack :
            flat =[]
            for ids in sequences :
                if len (ids )>=3 :
                    flat .extend (ids )
            step =int (block_size )
            limit =len (flat )-1 
            for i in range (0 ,limit ,step ):
                chunk =flat [i :i +step +1 ]
                if len (chunk )<3 :break 
                if len (chunk )<step +1 :
                    break 
                x =torch .tensor (chunk [:-1 ],dtype =torch .long )
                y =torch .tensor (chunk [1 :],dtype =torch .long )
                self .items .append ((x ,y ))
        else :
            for ids in sequences :
                if len (ids )<3 :continue 
                for i in range (0 ,len (ids )-1 ,block_size ):
                    chunk =ids [i :i +block_size +1 ]
                    if len (chunk )>=3 :
                        x =torch .zeros (block_size ,dtype =torch .long );y =torch .full ((block_size ,),-100 ,dtype =torch .long )
                        n =min (block_size ,len (chunk )-1 );x [:n ]=torch .as_tensor (chunk [:n ]);y [:n ]=torch .as_tensor (chunk [1 :n +1 ])
                        self .items .append ((x ,y ))

class _ListChunkDataset (Dataset ):
    def __init__ (self ,items ):self .items =items 
    def __len__ (self ):return len (self .items )
    def __getitem__ (self ,i ):
        item =self .items [i ]
        if isinstance (item ,tuple ):return item 
        a =item 
        x =torch .as_tensor (a [:-1 ],dtype =torch .long )
        y =torch .as_tensor (a [1 :],dtype =torch .long )
        return x ,y 

def collate (batch ):

    return torch .stack ([b [0 ]for b in batch ]),torch .stack ([b [1 ]for b in batch ])

def _load_cache (tok ):
    if not CACHE .exists ():return None 
    try :
        c =torch .load (CACHE ,map_location ='cpu',weights_only =False )
        if c .get ('corpus_hash')==hashlib .sha1 (load_corpus_text ().encode ('utf8')).hexdigest ()and c .get ('vocab_size')==tok .vocab_size :
            return c .get ('seqs')
    except Exception :pass 
    return None 


def _load_discord_token_cache (tok ,records ):
    if not DISCORD_CACHE .exists ():return None 
    try :
        c =torch .load (DISCORD_CACHE ,map_location ='cpu',weights_only =False )
        texts =[t for t ,_ in records ]
        digest =hashlib .sha1 (('\n'.join (texts )).encode ('utf8')).hexdigest ()
        if c .get ('records_hash')==digest and c .get ('vocab_size')==tok .vocab_size :
            return c .get ('seqs')
    except Exception :
        pass 
    return None 

def _tokenize_parallel (tok ,records ):
    texts =[t for t ,_ in records ]
    if not texts :return []



    workers =max (1 ,min (4 ,len (texts ),int (os .environ .get ('TOKENIZER_WORKERS','4'))))
    if workers ==1 :
        return [tok .encode (t )for t in texts ]
    with concurrent .futures .ThreadPoolExecutor (max_workers =workers ,thread_name_prefix ='tok')as ex :
        return list (ex .map (tok .encode ,texts ))

def _tokenizer_info ():
    p =Path ('data/bunsho_bunkatsu.json')
    if not p .exists ():return {}
    try :return json .loads (p .read_text (encoding ='utf-8'))
    except Exception :return {}


def _free_tcp_port ():
    with socket .socket (socket .AF_INET ,socket .SOCK_STREAM )as s :
        s .bind (('127.0.0.1',0 ))
        return s .getsockname ()[1 ]


def _maybe_compile (model ,cfg ):
    if not cfg .get ('compile_model',True ):
        return model 
    if not torch .cuda .is_available ()and IPEX is not None and cfg .get ('ipex',{}).get ('enabled',True ):
        print ('[train] IPEX active: IPEX/oneDNN owns CPU graph optimization',flush =True )
        return model 
    if torch .cuda .is_available ():
        try :
            major ,_ =torch .cuda .get_device_capability ()
            if major <8 :
                print ('[train] GPU is pre-Ampere (compute capability < 8); skipping torch.compile to save VRAM',flush =True )
                return model 
        except Exception :
            pass 
    try :
        import torch ._dynamo as _torch_dynamo 
        _torch_dynamo .config .suppress_errors =True 
        if not torch .cuda .is_available ():
            try :
                return torch .compile (model ,mode ='max-autotune',dynamic =False )
            except Exception :
                pass 
        return torch .compile (model )
    except Exception as e :
        print (f'[train] torch.compile unavailable ({e }); using eager mode',flush =True )
        return model 


def _maybe_ipex_optimize (model ,optimizer ,cfg ,device ):
    if device !='cpu':
        return model ,optimizer ,False 
    enabled =bool (cfg .get ('ipex',{}).get ('enabled',True ))
    return __import__ ('cpu_kasoku').optimize_training (model ,optimizer ,torch ,enabled =enabled ,bf16 =bool (cfg .get ('ipex',{}).get ('bf16',True )))


def _make_optimizer (params ,cfg ,use_bnb ,device ='cuda',log_prefix ='[train]'):
    params =list (params )
    if device =='cpu':



        try :
            opt =torch .optim .AdamW (params ,lr =float (cfg .get ('learning_rate',3e-4 )),
            weight_decay =float (cfg .get ('weight_decay',.01 )),
            foreach =True )
            kind ='adamw_fp32_cpu_foreach'
        except TypeError :
            opt =torch .optim .AdamW (params ,lr =float (cfg .get ('learning_rate',3e-4 )),
            weight_decay =float (cfg .get ('weight_decay',.01 )))
            kind ='adamw_fp32_cpu'
        if log_prefix :print (f'{log_prefix } using {kind }',flush =True )
        return opt ,kind 
    if use_bnb :
        try :
            import bitsandbytes as bnb 
            opt =bnb .optim .AdamW8bit (params ,lr =float (cfg .get ('learning_rate',3e-4 )),weight_decay =float (cfg .get ('weight_decay',.01 )))
            if log_prefix :print (f'{log_prefix } using bitsandbytes AdamW8bit',flush =True )
            return opt ,'adamw8bit'
        except Exception as e :
            if log_prefix :print (f'{log_prefix } bitsandbytes unavailable ({e }); check that bitsandbytes is actually '
            f'installed for this CUDA/torch build (pip show bitsandbytes / python -c "import bitsandbytes").',flush =True )
    if log_prefix :print (f'{log_prefix } falling back to momentum-SGD (memory-safe, but converges slower than Adam) '
    f'to avoid OOM from fp32 AdamW optimizer states',flush =True )
    opt =torch .optim .SGD (params ,lr =float (cfg .get ('learning_rate',3e-4 )),momentum =0.9 ,weight_decay =float (cfg .get ('weight_decay',.01 )))
    return opt ,'sgd_fallback'


def _ddp_worker (rank ,world_size ,master_port ,cfg ,ckpt_path_str ,pretrain_chunks ,discord_chunks ,
bs ,grad_acc ,epochs ,log_every ,tokens_seen ,result_q ):
    import torch .distributed as dist 
    from torch .nn .parallel import DistributedDataParallel as DDP 
    from torch .utils .data .distributed import DistributedSampler 
    os .environ ['MASTER_ADDR']='127.0.0.1'
    os .environ ['MASTER_PORT']=str (master_port )
    try :
        dist .init_process_group ('nccl',rank =rank ,world_size =world_size )
        torch .cuda .set_device (rank )
        device =f'cuda:{rank }'
        model =MiniLLM (cfg )
        ckpt_path =Path (ckpt_path_str )



        if rank ==0 and ckpt_path .exists ():
            old =torch .load (ckpt_path ,map_location ='cpu',weights_only =False )
            if old .get ('vocab_size')==cfg ['vocab_size']and old .get ('model_config',{}).get ('n_embd',cfg ['n_embd'])==cfg ['n_embd']:
                model .load_state_dict (old ['model'])
        model .grad_checkpointing =bool (cfg .get ('gradient_checkpointing',True ))
        model .to (device )


        compiled =_maybe_compile (model ,cfg )
        ddp_model =DDP (compiled ,device_ids =[rank ],find_unused_parameters =False ,
        static_graph =bool (cfg .get ('ddp_static_graph',False )))

        ds =_ListChunkDataset (pretrain_chunks +discord_chunks )
        sampler =DistributedSampler (ds ,num_replicas =world_size ,rank =rank ,shuffle =True ,drop_last =True )
        loader =DataLoader (ds ,batch_size =bs ,sampler =sampler ,collate_fn =collate ,num_workers =2 ,
        pin_memory =True ,persistent_workers =False )

        use_bnb =bool (cfg .get ('use_8bit_optimizer',True ))
        opt ,opt_kind =_make_optimizer (model .parameters (),cfg ,use_bnb ,log_prefix ='[train][DDP]'if rank ==0 else '')

        scaler =torch .amp .GradScaler ('cuda',enabled =True )
        losses =[]
        n_steps =len (loader )
        for epoch in range (epochs ):
            sampler .set_epoch (epoch )
            opt .zero_grad (set_to_none =True )
            for step ,(x ,y )in enumerate (loader ):
                x ,y =x .to (device ,non_blocking =True ),y .to (device ,non_blocking =True )
                boundary =(step +1 )%grad_acc ==0 


                sync_ctx =contextlib .nullcontext ()if boundary else ddp_model .no_sync ()
                with sync_ctx :
                    with torch .autocast (device_type ='cuda',dtype =torch .float16 ,enabled =True ):
                        _ ,loss ,_ =ddp_model (x ,y )
                    scaler .scale (loss /grad_acc ).backward ()
                if boundary :
                    scaler .unscale_ (opt );torch .nn .utils .clip_grad_norm_ (model .parameters (),1.0 )
                    scaler .step (opt );scaler .update ();opt .zero_grad (set_to_none =True )
                losses .append (float (loss .detach ()))
                if rank ==0 and (step +1 )%max (1 ,log_every )==0 :
                    w =losses [-log_every :]
                    print (f"[train][DDP] epoch={epoch +1 }/{epochs } step={step +1 }/{n_steps } loss={sum (w )/len (w ):.4f}",flush =True )
        dist .barrier ()
        if rank ==0 :
            ckpt_path .parent .mkdir (parents =True ,exist_ok =True )
            torch .save ({'model':model .state_dict (),'optimizer':opt .state_dict (),'vocab_size':cfg ['vocab_size'],
            'model_config':{k :cfg [k ]for k in ('n_embd','n_head','n_layer','block_size','dropout')},
            'tokens_seen':tokens_seen ,'trained_at':time .time ()},ckpt_path )
            result_q .put (('ok',sum (losses )/max (1 ,len (losses ))))
    except torch .cuda .OutOfMemoryError as e :
        result_q .put (('oom',f'rank{rank }: {e }'))
    except Exception as e :
        import traceback ;traceback .print_exc ()
        result_q .put (('error',f'rank{rank }: {e }'))
    finally :
        if dist .is_initialized ():dist .destroy_process_group ()


def _train_multi_gpu (n_gpus ,cfg ,pretrain_chunks ,discord_chunks ,bs ,grad_acc ,epochs ,log_every ,tokens_seen ):
    import torch .multiprocessing as mp 




    ctx =mp .get_context ('spawn')
    result_q =ctx .SimpleQueue ()
    port =_free_tcp_port ()
    try :
        mp .spawn (_ddp_worker ,
        args =(n_gpus ,port ,cfg ,str (CKPT ),pretrain_chunks ,discord_chunks ,bs ,grad_acc ,epochs ,log_every ,tokens_seen ,result_q ),
        nprocs =n_gpus ,join =True ,start_method ='spawn')
    except Exception as e :



        msg =str (e )
        if not result_q .empty ():
            status ,payload =result_q .get ()
            if status =='oom':raise torch .cuda .OutOfMemoryError (payload )
            if status =='error':raise RuntimeError (f'[train] DDP学習に失敗: {payload }')from e 
        if 'out of memory'in msg .lower ()or 'cuda oom'in msg .lower ():
            raise torch .cuda .OutOfMemoryError (msg )from e 
        raise 
    if result_q .empty ():
        raise RuntimeError ('[train] DDPワーカーが結果を返す前に落ちました(ログを確認してください)')
    status ,payload =result_q .get ()
    if status =='oom':
        raise torch .cuda .OutOfMemoryError (payload )
    if status =='error':
        raise RuntimeError (f'[train] DDP学習に失敗: {payload }')
    return payload 


def _free_cuda ():
    gc .collect ()
    if torch .cuda .is_available ():
        torch .cuda .empty_cache ()
        torch .cuda .synchronize ()


def train_once (incremental =False ):

    _free_cuda ()
    try :
        _train_once_impl (incremental )
    finally :
        _free_cuda ()


def _text_quality (text ,source ):
    text =' '.join (str (text ).split ())
    if len (text )<80 :
        return None 
    if len (set (text ))<12 :
        return None 
    bad =('http://','https://','BEGIN:VCARD','-----BEGIN','<script','javascript:')
    if any (x in text .lower ()for x in bad ):
        return None 
    if text .count ('ｗ')+text .count ('w')>len (text )*0.45 :
        return None 
    if source =='wikipedia'and len (text )<180 :
        return None 
    return text 


def _documents (path ,source ,max_chars =0 ):
    if not path .exists ():
        return []
    docs =[];buf =[];total =0 ;limit =max (0 ,int (max_chars ))
    with path .open ('r',encoding ='utf8',errors ='ignore')as f :
        in_doc =False 
        for line in f :
            if '<|document|>'in line :
                in_doc =True ;buf =[];continue 
            if '<|end_document|>'in line :
                q =_text_quality (' '.join (buf ),source )
                if q :
                    docs .append (q );total +=len (q )
                buf =[];in_doc =False 
                if limit and total >=limit :
                    break 
                continue 
            if in_doc :
                buf .append (line .strip ())
        if buf and in_doc and (not limit or total <limit ):
            q =_text_quality (' '.join (buf ),source )
            if q :docs .append (q )
    if not docs :

        with path .open ('r',encoding ='utf8',errors ='ignore')as f :
            text =f .read (limit if limit else 8_000_000 )
        q =_text_quality (text ,source )
        if q :docs =[q ]
    return docs 


def _cache_key_docs (docs ,vocab_size ):
    h =hashlib .sha256 ()
    for d in docs :
        h .update (d .encode ('utf8',errors ='ignore'))
        h .update (b'\\0')
    h .update (str (vocab_size ).encode ())
    return h .hexdigest ()


def _load_or_make_seq_cache (path ,tok ,docs ,label ):
    key =_cache_key_docs (docs ,tok .vocab_size )
    if path .exists ():
        try :
            c =torch .load (path ,map_location ='cpu',weights_only =False )
            if c .get ('key')==key and c .get ('vocab_size')==tok .vocab_size :
                return c .get ('seqs',[])
        except Exception :
            pass 
    seqs =_tokenize_parallel (tok ,[(d ,label )for d in docs ])
    seqs =[x for x in seqs if len (x )>=16 ]
    path .parent .mkdir (parents =True ,exist_ok =True )
    torch .save ({'key':key ,'vocab_size':tok .vocab_size ,'seqs':seqs },path )
    return seqs 


def _pack_sequences (seqs ,block_size ,token_budget =0 ,seed =1337 ):
    if not seqs :
        return []
    rng =random .Random (seed )
    ordered =list (seqs )
    rng .shuffle (ordered )
    flat =[]
    budget =max (0 ,int (token_budget ))
    used =0 
    for ids in ordered :
        if len (ids )<16 :
            continue 
        take =len (ids )
        if budget and used +take >budget :
            take =max (0 ,budget -used )
        if take >=16 :
            flat .extend (ids [:take ])
            used +=take 
        if budget and used >=budget :
            break 
    if len (flat )<block_size +1 :
        return []
    n =(len (flat )-1 )//block_size 
    flat =flat [:n *block_size +1 ]
    return [(torch .tensor (flat [i *block_size :(i +1 )*block_size ],dtype =torch .long ),
    torch .tensor (flat [i *block_size +1 :(i +1 )*block_size +1 ],dtype =torch .long ))
    for i in range (n )]


def _sample_chunks (chunks ,count ,seed ):
    if count <=0 or len (chunks )<=count :
        return list (chunks )
    rng =random .Random (seed )
    return rng .sample (chunks ,count )


def _save_stage (path ,model ,opt ,tok ,cfg ,tokens_seen ,stage ,base_pretrained_at =0.0 ):
    path .parent .mkdir (parents =True ,exist_ok =True )
    torch .save ({
    'model':model .state_dict (),
    'optimizer':opt .state_dict ()if opt is not None else None ,
    'vocab_size':tok .vocab_size ,
    'model_config':{k :cfg [k ]for k in ('n_embd','n_head','n_layer','block_size','dropout')},
    'architecture':getattr (model ,'architecture',''),
    'tokens_seen':int (tokens_seen ),
    'stage':stage ,
    'base_stage':'pretrain'if stage =='pretrain'else 'pretrain',
    'base_pretrained_at':float (base_pretrained_at ),
    'trained_at':time .time (),
    },path )


def _load_model_state (model ,path ,cfg ,tok ):
    if not path .exists ():
        return {},False 
    try :
        old =torch .load (path ,map_location ='cpu',weights_only =False )
        mc =old .get ('model_config',{})
        ok =(old .get ('vocab_size')==tok .vocab_size and 
        int (mc .get ('n_embd',cfg ['n_embd']))==int (cfg ['n_embd'])and 
        int (mc .get ('n_layer',cfg ['n_layer']))==int (cfg ['n_layer'])and int (mc .get ('n_head',cfg ['n_head']))==int (cfg ['n_head'])and int (mc .get ('block_size',cfg ['block_size']))==int (cfg ['block_size'])and old .get ('architecture')==getattr (model ,'architecture',''))
        if ok :
            model .load_state_dict (old ['model'],strict =True )
            return old ,True 
    except Exception as e :
        print (f'[train] checkpoint load skipped: {e }',flush =True )
    return {},False 


def _make_scheduler (opt ,total_steps ,cfg ):
    warmup_ratio =float (cfg .get ("training",{}).get ("warmup_ratio",0.03 ))
    min_ratio =float (cfg .get ("training",{}).get ("min_lr_ratio",0.1 ))
    total =max (1 ,int (total_steps ))
    warm =max (1 ,int (total *warmup_ratio ))if total >1 else 0 
    def f (step ):
        if warm and step <warm :
            return max (1e-6 ,(step +1 )/warm )
        if total <=warm :
            return min_ratio 
        progress =min (1.0 ,max (0.0 ,(step -warm )/(total -warm )))
        return min_ratio +(1.0 -min_ratio )*0.5 *(1.0 +math .cos (math .pi *progress ))
    return torch .optim .lr_scheduler .LambdaLR (opt ,f )


def _run_cpu_stage (model ,chunks ,cfg ,tok ,ckpt_path ,stage ,token_budget ,resume_path =None ,replay =None ):
    if not chunks :
        print (f'[train] {stage }: no chunks',flush =True )
        return 0.0 ,0 
    tr =cfg .get ('training',{})
    bs =max (1 ,int (tr .get ('cpu_batch_size',cfg .get ('batch_size',4 ))))
    grad_acc =max (1 ,int (tr .get ('cpu_gradient_accumulation_steps',cfg .get ('gradient_accumulation_steps',8 ))))
    epochs =max (1 ,int (tr .get ('epochs_per_update',1 )))
    lr =float (tr .get ('pretrain_learning_rate'if stage =='pretrain'else 'chat_learning_rate',cfg .get ('learning_rate',2e-4 )))
    wd =float (cfg .get ('weight_decay',0.01 ))
    max_steps =int (tr .get ('max_optimizer_steps',0 ))
    seed =int (tr .get ('seed',1337 ))
    model .grad_checkpointing =False 
    model .to ('cpu').train ()
    opt ,_ =_make_optimizer (model .parameters (),dict (cfg ,learning_rate =lr ),False ,device ='cpu',log_prefix =f'[train][{stage }]')
    if resume_path and Path (resume_path ).exists ():
        old ,_ok =_load_model_state (model ,Path (resume_path ),cfg ,tok )
        if old .get ('optimizer')and bool (tr .get ('resume_optimizer',True )):
            try :opt .load_state_dict (old ['optimizer'])
            except Exception :pass 
    model ,opt ,ipex_active =_maybe_ipex_optimize (model ,opt ,cfg ,'cpu')
    estimated_steps =max (1 ,math .ceil (math .ceil (len (chunks )/bs )/grad_acc )*epochs )
    total_sched_steps =min (max_steps ,estimated_steps )if max_steps else estimated_steps 
    scheduler =_make_scheduler (opt ,total_sched_steps ,cfg )
    if replay :
        replay_n =max (0 ,int (tr .get ('replay_chunks',0 )))
        if replay_n :
            chunks =list (chunks )+_sample_chunks (replay ,replay_n ,seed +991 )
    rng =random .Random (seed +int (time .time ()//86400 ))
    losses =[];step_count =0 ;seen =0 
    bf16 =bool (cfg .get ('ipex',{}).get ('bf16',True ))and __import__ ('cpu_kasoku').supports_bf16 ()
    amp_dtype =torch .bfloat16 if bf16 else torch .float32 
    print (f'[train][{stage }] CPU chunks={len (chunks ):,} batch={bs } grad_acc={grad_acc } bf16={bf16 } token_budget={token_budget :,}',flush =True )
    for epoch in range (epochs ):
        order =list (range (len (chunks )));rng .shuffle (order )
        opt .zero_grad (set_to_none =True )
        accum =0 
        for pos in range (0 ,len (order ),bs ):
            batch =[chunks [i ]for i in order [pos :pos +bs ]]
            x =torch .stack ([a for a ,_ in batch ]).contiguous ()
            y =torch .stack ([b for _ ,b in batch ]).contiguous ()
            with torch .autocast (device_type ='cpu',dtype =amp_dtype ,enabled =bf16 ):
                _ ,loss ,_ =model (x ,y )
            (loss /grad_acc ).backward ()
            accum +=1 
            seen +=x .numel ()
            boundary =(accum >=grad_acc or pos +bs >=len (order ))
            if boundary :
                torch .nn .utils .clip_grad_norm_ (model .parameters (),1.0 )
                opt .step ();opt .zero_grad (set_to_none =True )
                scheduler .step ();step_count +=1 ;accum =0 
                losses .append (float (loss .detach ()))
                if step_count %max (1 ,int (cfg .get ('log_every_steps',200 )))==0 :
                    avg =sum (losses [-50 :])/len (losses [-50 :])
                    print (f'[train][{stage }] epoch={epoch +1 }/{epochs } opt_step={step_count } loss={avg :.4f} tokens={seen :,}',flush =True )
                if max_steps and step_count >=max_steps :
                    break 
            del x ,y ,loss 
        if max_steps and step_count >=max_steps :
            break 
    base_time =float (cfg .get ('_base_pretrained_at',0.0 ))if stage =='discord'else 0.0 
    _save_stage (Path (ckpt_path ),model ,opt ,tok ,cfg ,seen ,stage ,base_time )
    avg =sum (losses )/max (1 ,len (losses ))
    print (f'[train][{stage }] saved={ckpt_path } avg_loss={avg :.4f} tokens={seen :,}',flush =True )
    return avg ,seen 


def _train_once_impl (incremental =False ):
    print (f'[train] starting two-stage pipeline incremental={incremental }',flush =True )
    cfg =dict (CFG )
    tr =cfg .setdefault ('training',{})
    ds_cfg =cfg .setdefault ('dataset',{})
    stage_cfg =cfg .setdefault ('stages',{})
    PRETRAIN_CKPT =Path ('data/pregakushued.pt')
    FINAL_CKPT =Path ('data/moderu.pt')
    AOZ_CACHE =Path ('data/aozora_token_cache.pt')
    WIKI_CACHE =Path ('data/wikipedia_token_cache.pt')

    records =load_records ()
    corpus_chars =corpus_char_count ()
    if corpus_chars <=0 and not records :
        print ('[train] 学習データがありません。',flush =True );return 

    cur_chars =corpus_chars +sum (len (r [0 ])for r in records )
    target_vocab =int (cfg .get ('vocab_size',32000 ))
    info =_tokenizer_info ();prior_chars =int (info .get ('trained_on_chars',0 ))
    need_tok =not TOK .exists ()
    if TOK .exists ():
        tok =SentencePieceTokenizer (TOK )
        stale =(tok .vocab_size <target_vocab *0.5 and cur_chars >max (100_000 ,prior_chars *2 ))
        need_tok =stale 
    if need_tok :
        corpus_seed =load_corpus_text (max_chars =int (cfg .get ('dataset',{}).get ('tokenizer_seed_chars',8_000_000 )))
        seed_text =(corpus_seed +'\n'+'\n'.join (r [0 ]for r in records )).strip ()
        tok =SentencePieceTokenizer .train ([seed_text ]if seed_text else [r [0 ]for r in records ],TOK ,target_vocab )
        tok .save_info ('data/bunsho_bunkatsu.json',trained_on_chars =cur_chars )
        for f in (Path ('data/token_cache.pt'),AOZ_CACHE ,WIKI_CACHE ,DISCORD_CACHE ):f .unlink (missing_ok =True )
    else :
        tok =SentencePieceTokenizer (TOK )

    block =int (cfg .get ('block_size',512 ))
    pre_budget =int (stage_cfg .get ('pretrain_token_budget',tr .get ('pretrain_token_budget',300_000_000 )))
    aoz_ratio =float (stage_cfg .get ('aozora_ratio',0.30 ));wiki_ratio =float (stage_cfg .get ('wikipedia_ratio',0.70 ))
    aoz_docs =_documents (CORPUS ,'aozora',max_chars =int (pre_budget *aoz_ratio *2.0 ))
    wiki_docs =_documents (WIKI ,'wikipedia',max_chars =int (pre_budget *wiki_ratio *2.0 ))
    aoz =_load_or_make_seq_cache (AOZ_CACHE ,tok ,aoz_docs ,'aozora')if aoz_docs else []
    wiki =_load_or_make_seq_cache (WIKI_CACHE ,tok ,wiki_docs ,'wikipedia')if wiki_docs else []
    records =load_records ()
    discord =_load_discord_token_cache (tok ,records )
    if discord is None :
        discord =_tokenize_parallel (tok ,records )
        try :
            digest =hashlib .sha1 ('\n'.join (t for t ,_ in records ).encode ('utf8')).hexdigest ()
            torch .save ({'records_hash':digest ,'vocab_size':tok .vocab_size ,'seqs':discord },DISCORD_CACHE )
        except Exception :pass 

    pre_total =max (block +1 ,pre_budget )
    aoz_budget =int (pre_total *aoz_ratio );wiki_budget =int (pre_total *wiki_ratio )
    pre_chunks =_pack_sequences (aoz ,block ,aoz_budget ,1337 )+_pack_sequences (wiki ,block ,wiki_budget ,7331 )
    print (f'[train] pretrain sources: Aozora={len (aoz ):,} seqs Wikipedia={len (wiki ):,} seqs -> {len (pre_chunks ):,} chunks',flush =True )


    if not PRETRAIN_CKPT .exists ()or bool (stage_cfg .get ('force_pretrain',False )):
        model =MiniLLM (cfg )
        _run_cpu_stage (model ,pre_chunks ,cfg ,tok ,PRETRAIN_CKPT ,'pretrain',pre_total )
        del model ;gc .collect ()
    else :
        print ('[train] pretrained checkpoint already exists; skipping full pretraining',flush =True )


    if not discord :
        print ('[train] Discord経験がまだないため、事前学習だけ完了しました。',flush =True )
        return 
    chat_budget =int (stage_cfg .get ('discord_token_budget_per_update',tr .get ('discord_token_budget_per_update',2_000_000 )))
    discord_chunks =_pack_sequences (discord ,block ,chat_budget ,99991 )
    replay =int (tr .get ('replay_chunks',max (64 ,len (discord_chunks )//10 )))
    replay_chunks =_sample_chunks (pre_chunks ,replay ,4242 )if pre_chunks else []
    model =MiniLLM (cfg )
    pre_meta ={}
    try :
        pre_meta =torch .load (PRETRAIN_CKPT ,map_location ='cpu',weights_only =False )
    except Exception :
        pass 
    old ,ok =_load_model_state (model ,FINAL_CKPT ,cfg ,tok )
    final_base =float (old .get ('base_pretrained_at',0 ))if ok else 0.0 
    pre_time =float (pre_meta .get ('trained_at',0 ))if pre_meta else 0.0 
    if not ok or final_base <pre_time :
        old ,ok =_load_model_state (model ,PRETRAIN_CKPT ,cfg ,tok )
    if not ok :
        print ('[train] valid pretrained checkpointがありません。事前学習を先に実行します。',flush =True )
        del model ;return 
    cfg ['_base_pretrained_at']=pre_time 
    _run_cpu_stage (model ,discord_chunks ,cfg ,tok ,FINAL_CKPT ,'discord',chat_budget ,replay =replay_chunks )
    del model ;gc .collect ()


def main ():train_once (CKPT .exists ())
if __name__ =='__main__':main ()
