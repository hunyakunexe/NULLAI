import os 
import platform 


def _cpuinfo_text ():
    try :
        with open ('/proc/cpuinfo','r',encoding ='utf-8',errors ='ignore')as f :
            return f .read (800000 ).lower ()
    except Exception :
        return ''


def _cpu_flags ():
    flags =set ()
    text =_cpuinfo_text ()
    for line in text .splitlines ():
        if line .startswith ('flags')or line .startswith ('features'):
            flags .update (line .split (':',1 )[-1 ].split ())
    return flags 


def _intel_cpu ()->bool :
    text =_cpuinfo_text ()
    return 'genuineintel'in text or 'intel'in platform .processor ().lower ()


IS_INTEL_CPU =_intel_cpu ()
IPEX =None 
IPEX_VERSION =None 


def configure_environment ()->None :
    if not IS_INTEL_CPU :
        return 


    allowed =len (os .sched_getaffinity (0 ))if hasattr (os ,'sched_getaffinity')else (os .cpu_count ()or 1 )
    physical =physical_core_count ()
    threads =max (1 ,min (physical ,allowed ))

    env ={
    'OMP_NUM_THREADS':str (threads ),
    'MKL_NUM_THREADS':str (threads ),
    'OMP_PROC_BIND':'TRUE',
    'OMP_PLACES':'cores',
    'OMP_DYNAMIC':'FALSE',
    'MKL_DYNAMIC':'FALSE',
    'KMP_AFFINITY':'granularity=fine,compact,1,0',

    'KMP_BLOCKTIME':'0',

    'DNNL_PRIMITIVE_CACHE_CAPACITY':'32768',
    'DNNL_GRAPH_CACHE_SIZE':'8192',
    }
    for k ,v in env .items ():
        os .environ .setdefault (k ,v )





def supports_avx512_bf16 ():
    return 'avx512_bf16'in _cpu_flags ()


def supports_amx_bf16 ():
    return 'amx_bf16'in _cpu_flags ()


def supports_avx512_vnni ():
    flags =_cpu_flags ()
    return 'avx512_vnni'in flags or 'avx_vnni'in flags 


def supports_bf16 ():
    return supports_avx512_bf16 ()or supports_amx_bf16 ()


def physical_core_count ()->int :
    allowed =set (os .sched_getaffinity (0 ))if hasattr (os ,'sched_getaffinity')else None 
    pairs =set ()
    base ='/sys/devices/system/cpu'
    try :
        for cpu in os .listdir (base ):
            if not cpu .startswith ('cpu')or not cpu [3 :].isdigit ():
                continue 
            n =int (cpu [3 :])
            if allowed is not None and n not in allowed :
                continue 
            topo =os .path .join (base ,cpu ,'topology')
            with open (os .path .join (topo ,'physical_package_id'))as f :
                pkg =f .read ().strip ()
            with open (os .path .join (topo ,'core_id'))as f :
                core =f .read ().strip ()
            pairs .add ((pkg ,core ))
    except Exception :
        pairs .clear ()
    if pairs :
        return len (pairs )
    return max (1 ,((len (allowed )if allowed is not None else (os .cpu_count ()or 1 ))+1 )//2 )



configure_environment ()


def configure_torch_threads (torch ,log_prefix ='[train]')->int :
    cores =physical_core_count ()
    allowed =len (os .sched_getaffinity (0 ))if hasattr (os ,'sched_getaffinity')else (os .cpu_count ()or 1 )
    threads =max (1 ,min (int (os .environ .get ('OMP_NUM_THREADS',cores )),cores ,allowed ))
    os .environ ['OMP_NUM_THREADS']=str (threads )
    os .environ ['MKL_NUM_THREADS']=str (threads )
    torch .set_num_threads (threads )
    try :


        torch .set_num_interop_threads (max (1 ,int (os .environ .get ('CPU_INTEROP_THREADS','1'))))
    except RuntimeError :
        pass 
    try :
        torch .backends .mkldnn .enabled =True 
    except Exception :
        pass 
    try :
        torch .set_float32_matmul_precision ('high')
    except Exception :
        pass 
    try :
        torch .set_flush_denormal (True )
    except Exception :
        pass 
    if log_prefix :
        mode =f'IPEX {IPEX_VERSION }'if IPEX is not None else 'PyTorch/oneDNN fallback'
        print (
        f'{log_prefix } Intel CPU EXTREME: {mode }; physical_cores={cores }; '
        f'affinity_cpus={allowed }; threads={threads }; '
        f'AMX_BF16={supports_amx_bf16 ()}; AVX512_BF16={supports_avx512_bf16 ()}; '
        f'AVX512/VNNI={supports_avx512_vnni ()}',flush =True )
    return cores 


def optimize_training (model ,optimizer ,torch ,enabled =True ,bf16 =True ,log_prefix ='[train]'):
    if not enabled or IPEX is None :
        return model ,optimizer ,False 
    try :
        dtype =torch .bfloat16 if (bf16 and supports_bf16 ())else torch .float32 
        kwargs =dict (optimizer =optimizer ,dtype =dtype ,inplace =True ,weights_prepack =True )
        model ,optimizer =IPEX .optimize (model ,**kwargs )
        print (f'{log_prefix } IPEX optimize(training)=ON; dtype={dtype }; weight_prepack=ON',flush =True )
        return model ,optimizer ,True 
    except Exception as e :
        print (f'{log_prefix } IPEX optimize(training) unavailable ({e }); using PyTorch/oneDNN',flush =True )
        return model ,optimizer ,False 


def optimize_inference (model ,torch ,enabled =True ,log_prefix ='[inference]'):
    if not enabled or IPEX is None :
        return model ,False 
    try :
        dtype =torch .bfloat16 if supports_bf16 ()else torch .float32 
        model ,_ =IPEX .optimize (model ,dtype =dtype ,inplace =True ,weights_prepack =True )
        model .eval ()
        print (f'{log_prefix } IPEX optimize=ON; dtype={dtype }; weight_prepack=ON',flush =True )
        return model ,True 
    except Exception as e :
        print (f'{log_prefix } IPEX inference optimize unavailable ({e }); using PyTorch/oneDNN',flush =True )
        return model ,False 
