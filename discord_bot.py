import discord ,asyncio ,json ,os ,time ,random ,torch 
from keiken_core import ExperienceCore 
from gurafu_kioku import GraphMemory 
from imi_engine import analyze 
from bunsho_bunkatsu import SentencePieceTokenizer 
from moderu import MiniLLM 
from cpu_kasoku import configure_environment,configure_torch_threads,IS_INTEL_CPU,IPEX,optimize_inference
try :
 from torch .ao .quantization import quantize_dynamic 
except Exception :quantize_dynamic =None 
class Bot (discord .Client ):
 def __init__ (self ,cfg ):



  configure_environment ();configure_torch_threads(torch,log_prefix="[LLM]")
  intents =discord .Intents .default ();intents .message_content =True 
  super ().__init__ (intents =intents );self .cfg =cfg ;self .core =ExperienceCore ();self .graph =GraphMemory ();self ._last_text ={};self .pending ={};self .model =None ;self .tok =None ;self ._train_lock =asyncio .Lock ();self ._experience_count =0 ;self ._new_message_count =0 
  self ._reload_model ()

 def _reload_model (self ):
  try :
   tok =SentencePieceTokenizer ('data/bunsho_bunkatsu.moderu')
   mc =dict (self .cfg );mc ['vocab_size']=tok .vocab_size 
   model =MiniLLM (mc )
   ck =torch .load ('data/moderu.pt',map_location ='cpu',weights_only =False )
   model .load_state_dict (ck ['model']);model .eval ()
   inf=self .cfg .get('inference',{})
   if IS_INTEL_CPU and IPEX is not None and self .cfg .get('ipex',{}).get('enabled',True):
    try:model=optimize_inference(model,torch,True,log_prefix='[LLM]')[0]
    except Exception:pass
   elif inf.get('quantized',True) and quantize_dynamic:
    try:model=quantize_dynamic(model,{torch.nn.Linear},dtype=torch.qint8);model.eval()
    except Exception:pass
   try:
    if self .cfg .get('cpu',{}).get('inductor_fallback',True) and not (IS_INTEL_CPU and IPEX is not None):
     model=torch.compile(model,mode='reduce-overhead',fullgraph=False,dynamic=False)
   except Exception:pass
   self .tok =tok ;self .model =model 
   print ('[LLM] model (re)loaded; KV cache enabled')
  except Exception as e :print ('[LLM] custom model unavailable:',e )
 async def on_ready (self ):
  print ('READY',self .user ,flush =True )
 def _save_config (self ):
  with open ('config.json','w',encoding ='utf-8')as f :json .dump (self .cfg ,f ,ensure_ascii =False ,indent =2 )
 def allowed (self ,m ):
  p =self .cfg ['privacy'];b =self .cfg ['bot'];
  if m .author .bot and not p ['collect_bot_messages']:return False 
  if isinstance (m .channel ,discord .DMChannel )and not p ['collect_dms']:return False 
  if p ['allowed_channel_ids']and str (m .channel .id )not in map (str ,p ['allowed_channel_ids']):return False 
  if str (m .channel .id )in map (str ,p ['denied_channel_ids']):return False 
  return True 
 async def on_message (self ,m ):
  if not self .allowed (m ):return 
  if m .author .bot :return 

  parts =m .content .strip ().split ()
  if len (parts )>=2 and parts [0 ].lower ()=='/ep'and parts [1 ].lower ()=='channel':
   if not isinstance (m .author ,discord .Member )or not m .author .guild_permissions .administrator :
    await m .channel .send ('このコマンドは管理者専用です。');return 
   b =self .cfg ['bot'];ignored =b .setdefault ('ignored_channel_ids',[])
   action =parts [2 ].lower ()if len (parts )>=3 else 'list'
   if action =='list':
    if not ignored :
     await m .channel .send ('返信しないチャンネルは設定されていません。');return 
    names =[]
    for cid in ignored :
     ch =m .guild .get_channel (int (cid ))if m .guild else None 
     names .append (f'{ch .mention } (`{cid }`)'if ch else f'`{cid }`')
    await m .channel .send ('返信しないチャンネル:\n'+'\n'.join (names ));return 
   if action in ('deny','add','off'):
    cid =parts [3 ]if len (parts )>=4 else str (m .channel .id )
    cid =cid .strip ('<>#&!')
    if not cid .isdigit ():
     await m .channel .send ('チャンネルIDを指定してください。例: `/ep channel deny 123456789012345678`');return 
    if cid not in map (str ,ignored ):
     ignored .append (int (cid ))
     await asyncio .to_thread (self ._save_config )
    await m .channel .send (f'このチャンネル (`{cid }`) を返信対象外にしました。');return 
   if action in ('allow','remove','on'):
    cid =parts [3 ]if len (parts )>=4 else str (m .channel .id )
    cid =cid .strip ('<>#&!')
    if not cid .isdigit ():
     await m .channel .send ('チャンネルIDを指定してください。');return 
    b ['ignored_channel_ids']=[x for x in ignored if str (x )!=cid ]
    await asyncio .to_thread (self ._save_config )
    await m .channel .send (f'このチャンネル (`{cid }`) を返信対象に戻しました。');return 
   await m .channel .send ('使用法: `/ep channel deny [channel_id]` / `/ep channel allow [channel_id]` / `/ep channel list`');return 
  if str (m .channel .id )in map (str ,self .cfg ['bot'].get ('ignored_channel_ids',[])):
   return 
  if len (parts )>=2 and parts [0 ].lower ()=='/ep'and parts [1 ].lower ()=='probability':
   if not isinstance (m .author ,discord .Member )or not m .author .guild_permissions .administrator :
    await m .channel .send ('このコマンドは管理者専用です。');return 
   b =self .cfg ['bot']
   if len (parts )==2 :
    await m .channel .send (f"通常メッセージへの割り込み返信確率: {float (b .get ('occasional_reply_probability',0.08 ))*100 :.1f}%")
    return 
   try :
    value =float (parts [2 ])
   except ValueError :
    await m .channel .send ('使用法: /ep probability <0〜100>');return 
   if not 0 <=value <=100 :
    await m .channel .send ('確率は0〜100%で指定してください。');return 
   b ['occasional_reply_probability']=value /100.0 
   await asyncio .to_thread (self ._save_config )
   await m .channel .send (f'通常メッセージへの割り込み返信確率を {value :g}% に設定しました。')
   return 
  a =analyze (m .content ,self .core .known ());self .core .add_message (id =str (m .id ),channel =str (m .channel .id ),guild =str (m .guild .id )if m .guild else '',author =str (m .author .id ),text =m .content ,ts =m .created_at .timestamp ())
  cid =str (m .channel .id )




  self .graph .add (m .author .id ,m .content ,previous_text =self ._last_text .get (cid ))
  self ._last_text [cid ]=m .content 


  import re 
  text =m .content .strip ()
  q =re .match (r'^[「『]([^「」『』=]{1,80})[」』]\s*(?:=|は)\s*(.{1,300})$',text )
  if not q :
   q =re .match (r'^([^「」『』=\s]{1,80})=\s*(.{1,300})$',text )
  if q :
   self .core .add_term (q .group (1 ).strip (),q .group (2 ).strip ())
   if self .user and self .user .mentioned_in (m ):await m .channel .send ('覚えました。');return 
   return 
  b =self .cfg ['bot'];mentioned =self .user and self .user .mentioned_in (m )

  if not mentioned :
   if b .get ('reply_only_when_mentioned',False ):
    if not b .get ('occasional_reply',True ):return 
    if random .random ()>=float (b .get ('occasional_reply_probability',0.08 )):return 
   elif not b .get ('occasional_reply',True ):return 
  if not b ['auto_reply']:return 
  ctx ={'user':str (m .author .id ),'channel':str (m .channel .id ),'guild':str (m .guild .id )if m .guild else ''}
  ex =self .core .search (a ['semantic'],ctx ,b ['candidate_limit'],query =m .content )


  ask_prob =float (b .get ('unknown_term_question_probability',0.3 ))
  if a ['unknown']and random .random ()<ask_prob :
   reply =f'「{a ["unknown"][0 ]}」ってどういう意味？'
  else :
   history =[x .content async for x in m .channel .history (limit =b ['history_messages'],before =m )]
   inf=self .cfg .get ('inference',{})
   query_text=m .content
   def hist_score(text,pos,total):
    a=set(query_text[i:i+2] for i in range(max(0,len(query_text)-1)))
    z=set(text[i:i+2] for i in range(max(0,len(text)-1)))
    overlap=len(a&z)/max(1,len(a)) if a else 0.0
    rec=(pos+1)/max(1,total)
    return float(inf.get('history_relevance_weight',.72))*overlap+float(inf.get('history_recency_weight',.28))*rec
   ranked=sorted(enumerate(history),key=lambda x:hist_score(x[1],len(history)-x[0],len(history)),reverse=True)
   history=[x[1] for x in ranked[:int(b['history_messages'])]]
   memory =[]
   for e in ex [:8 ]:memory .append (f"発言:{e ['input']}\n返答:{e ['response']}\n結果:{e ['result']}")
   graph_candidates =self .graph .next_candidates (m .content ,limit =3 )
   graph_text ='\n'.join (f'候補:{r [0 ]}'for r in graph_candidates )
   prompt ='''役割: Discordの会話相手。\n方針: 現在の発言に直接答える。会話の流れを優先する。事実を勝手に作らない。質問には質問の内容を保ったまま答える。\n会話履歴:\n'''+'\n'.join (reversed (history ))+f'\n現在の発言:{m .content }\n関連する過去の経験:\n'+'\n'.join (memory )+f'\n会話グラフ候補:\n{graph_text }\n返答:'
   reply =None
   if self .model :
    inf=self .cfg .get ('inference',{})
    max_new=int (inf .get ('max_new_tokens',80 ))
    eos=self .tok .sp .eos_id ()
    def clean(t):
     t=re.sub(r'^(返答|回答|最終返答)[:：]?\s*','',t.strip ())
     t=t.split('\n')[0].strip ()
     return t[:int (b ['max_reply_chars'])]
    def score_candidate(c,gen_score):
     if not c:return -1e9
     q=re.sub(r'\s+','',m.content);r=re.sub(r'\s+','',c)
     if not r:return -1e9
     grams=set(q[i:i+2] for i in range(max(0,len(q)-1)))
     rgrams=set(r[i:i+2] for i in range(max(0,len(r)-1)))
     overlap=len(grams&rgrams)/max(1,len(grams)) if grams else 0.0
     echo=1.0 if r==q else 0.0
     contradiction=sum(1 for x in ('違う','不明','わからない','できない') if x in r) if '?' in q or '？' in q else 0
     anchors=re.findall(r'[A-Za-z0-9一-龯ぁ-んァ-ヶ]{2,}',m.content)
     evidence=sum(1 for z in anchors if z in c)/len(anchors) if anchors else 0.0
     return float(inf.get('rerank_weight',.55))*gen_score+float(inf.get('relevance_weight',.25))*overlap-float(inf.get('novelty_weight',.10))*echo-float(inf.get('contradiction_weight',.10))*contradiction+0.10*evidence
    candidates=[]
    hard=bool(inf.get('adaptive_compute',True)) and (len(m.content)>=45 or '?' in m.content or '？' in m.content or any(z in m.content for z in ('なぜ','どうして','理由','比較','違い','条件','なら','すると','つまり','説明')))
    base_count=int(inf.get('candidate_count',3));hard_count=int(inf.get('hard_candidate_count',5))
    count=max(1,hard_count if hard else base_count) if bool(inf.get('rerank',True)) else 1
    prompt_ids=torch.tensor([self.tok.encode(prompt)],dtype=torch.long)
    for i in range(count):
     temp=float(inf.get('candidate_temperature',inf.get('temperature',.72))) if count>1 else float(inf.get('temperature',.72))
     with torch.no_grad():
      y,gen_score=self.model.generate_with_score(prompt_ids,max_new_tokens=max_new,temperature=temp,top_k=int(inf.get('top_k',40)),repetition_penalty=float(inf.get('repetition_penalty',1.16)),no_repeat_ngram_size=int(inf.get('no_repeat_ngram_size',3)),eos_token_id=eos)
     draft=clean(self.tok.decode(y[0].tolist()[len(prompt_ids[0]):]))
     if draft and all(draft!=x[0] for x in candidates):candidates.append((draft,gen_score))
    if candidates:
     scored=sorted(((score_candidate(c,s),c,s) for c,s in candidates),reverse=True)
     reply=scored[0][1]
     margin=scored[0][0]-(scored[1][0] if len(scored)>1 else scored[0][0])
     need_verify=bool(inf.get('verify_reply',True)) and (hard or margin<float(inf.get('verify_margin',.04)))
     if need_verify:
      verify='元の発言に答えているか確認する。\n元の発言:'+m.content+f'\n候補:{reply}\n矛盾や作り話を避け、必要なら候補を修正して、最終返答だけを書く。\n最終返答:'
      z=torch.tensor([self.tok.encode(verify)],dtype=torch.long)
      with torch.no_grad():v=self.model.generate(z,max_new_tokens=max_new,temperature=float(inf.get('verify_temperature',.42)),top_k=int(inf.get('verify_top_k',20)),repetition_penalty=float(inf.get('repetition_penalty',1.15)),no_repeat_ngram_size=3,eos_token_id=eos)
      checked=clean(self.tok.decode(v[0].tolist()[len(z[0]):]))
      if checked:reply=checked
   if not reply :
    graph_candidates =self .graph .next_candidates (m .content ,limit =1 )
    reply =ex [0 ]['response']if ex else (graph_candidates [0 ]['dst_text']if graph_candidates else ('どういう意味？'if a ['semantic']=='question'else 'なるほど。'))
  sent =await m .channel .send (reply [:b ['max_reply_chars']]);self .pending [sent .id ]=(m ,a ,reply ,ctx );asyncio .create_task (self .observe (sent .id ))

 async def _train_background (self ):
  async with self ._train_lock :
   try :
    from gakushu import train_once 
    await asyncio .to_thread (train_once ,True )



    await asyncio .to_thread (self ._reload_model )
   except Exception as e :
    print ('[train] background update failed:',e )
 async def observe (self ,sid ):
  m ,a ,reply ,ctx =self .pending [sid ];await asyncio .sleep (self .cfg ['bot']['observation_delay_seconds']);msgs =[]
  async for x in m .channel .history (limit =self .cfg ['bot']['observation_max_messages'],after =m ):
   if x .id !=sid and not x .author .bot :msgs .append (x .content )

  positive =('いいね','それな','了解','わかる','助かる','ありがとう','正解','完璧','なるほど','その通り','草','w')
  negative =('違う','いや、それは','間違い','間違ってる','無理','ダメ','違います','訂正','じゃない')
  pos =sum (sum (t .lower ().count (z .lower ())for z in positive )for t in msgs )
  neg =sum (sum (t .lower ().count (z .lower ())for z in negative )for t in msgs )

  score =max (-1.0 ,min (1.0 ,(pos -neg *1.5 )/max (1 ,len (msgs ))))
  result ='successful'if score >=.45 else 'failed'if score <=-.45 else 'neutral'
  confidence =.75 if any (any (z in t for z in negative )for t in msgs )else (.65 if any (any (z in t .lower ()for z in positive )for t in msgs )else (.45 if msgs else .25 ))
  self .core .add_experience (semantic =a ['semantic'],input =m .content ,context =ctx ,response =reply ,reactions =msgs ,next_messages =msgs ,result =result ,score =score ,confidence =confidence ,fingerprint =a ['fingerprint'])
  self ._experience_count +=1 
  self ._new_message_count +=1 

  if self ._experience_count %int (self .cfg .get ('training',{}).get ('export_every_experiences',5 ))==0 :
   try :
    self .core .export (min_score =float (self .cfg .get ('experience',{}).get ('export_score',.45 )),min_conf =float (self .cfg .get ('experience',{}).get ('export_confidence',.45 )))
    self .core .export_conversation_windows (window =int (self .cfg .get ('training',{}).get ('conversation_window',12 )))
   except Exception as e :print ('[experience] export failed:',e )
  every =int (self .cfg .get ('training',{}).get ('train_every_experiences',100 ))
  if every >0 and self ._experience_count %every ==0 and not self ._train_lock .locked ():
   asyncio .create_task (self ._train_background ())
