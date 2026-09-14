import os 
from cpu_kasoku import configure_environment ,configure_torch_threads ,IS_INTEL_CPU ,IPEX ,optimize_inference 
configure_environment ()
import json ,torch 
from bunsho_bunkatsu import SentencePieceTokenizer 
from moderu import MiniLLM 
try :from torch .ao .quantization import quantize_dynamic 
except Exception :quantize_dynamic =None 
cfg =json .load (open ("config.json",encoding ="utf-8"));tok =SentencePieceTokenizer ("data/bunsho_bunkatsu.moderu");cfg ["vocab_size"]=tok .vocab_size 
model =MiniLLM (cfg );ck =torch .load ("data/moderu.pt",map_location ="cpu",weights_only =False );model .load_state_dict (ck ["model"]);model .eval ()
if IS_INTEL_CPU and IPEX is not None and cfg .get ("ipex",{}).get ("enabled",True ):
    try :
        model =IPEX .optimize (model ,dtype =torch .bfloat16 ,inplace =True ,weights_prepack =True )
        model .eval ()
        print ("[inference] IPEX optimize=ON",flush =True )
    except Exception as e :
        print (f"[inference] IPEX optimize unavailable ({e })",flush =True )
if cfg .get ("inference",{}).get ("quantized",False )and quantize_dynamic and not (IS_INTEL_CPU and IPEX is not None ):model =quantize_dynamic (model ,{torch .nn .Linear },dtype =torch .qint8 );model .eval ()
def generate (prompt ):
    ids =torch .tensor ([tok .encode (prompt )],dtype =torch .long );out =model .generate (ids ,max_new_tokens =int (cfg .get ("inference",{}).get ("max_new_tokens",120 )),temperature =.85 ,top_k =30 )

    return tok .decode (out [0 ].tolist ()[len (ids [0 ]):])
if __name__ =="__main__":print (generate (input (">>> ")))
