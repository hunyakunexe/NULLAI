import discord,asyncio,json,os,time,random,torch
from keiken_core import ExperienceCore
from gurafu_kioku import GraphMemory
from imi_engine import analyze
from bunsho_bunkatsu import SentencePieceTokenizer
from moderu import MiniLLM
try:
 from torch.ao.quantization import quantize_dynamic
except Exception: quantize_dynamic=None
class Bot(discord.Client):
 def __init__(self,cfg):
                                                   
                                                 
                                               
  intents=discord.Intents.default();intents.message_content=True
  super().__init__(intents=intents);self.cfg=cfg;self.core=ExperienceCore();self.graph=GraphMemory();self._last_text={};self.pending={};self.moderu=None;self.tok=None;self._gakushu_lock=asyncio.Lock();self._experience_count=0;self._new_message_count=0
  self._reload_moderu()

 def _reload_moderu(self):
  try:
   tok=SentencePieceTokenizer('data/bunsho_bunkatsu.moderu')
   mc=dict(self.cfg);mc['vocab_size']=tok.vocab_size
   moderu=MiniLLM(mc)
   ck=torch.load('data/moderu.pt',map_location='cpu',weights_only=False)
   moderu.load_state_dict(ck['moderu']); moderu.eval()
   if self.cfg.get('inference',{}).get('quantized',True) and quantize_dynamic:
    moderu=quantize_dynamic(moderu,{torch.nn.Linear},dtype=torch.qint8);moderu.eval()
   self.tok=tok; self.moderu=moderu
   print('[LLM] moderu (re)loaded; KV cache enabled')
  except Exception as e: print('[LLM] custom moderu unavailable:',e)
 async def on_ready(self):
  print('READY',self.user,flush=True)
  if not getattr(self,'_meintenance_started',False):
   self._meintenance_started=True
   asyncio.create_task(self._startup_meintenance())
 async def _startup_meintenance(self):
  print('[startup] meintenance task started',flush=True)
  async with self._gakushu_lock:
   try:
    from startup_meintenance import run as run_startup_meintenance
    await asyncio.to_thread(run_startup_meintenance, self.cfg)
    print('[startup] meintenance task finished',flush=True)
   except Exception:
    import traceback
    print('[startup] meintenance failed:',flush=True)
    traceback.print_exc()
 def _save_config(self):
  with open('config.json','w',encoding='utf-8') as f: json.dump(self.cfg,f,ensure_ascii=False,indent=2)
 def allowed(self,m):
  p=self.cfg['privacy'];b=self.cfg['bot'];
  if m.author.bot and not p['collect_bot_messages']:return False
  if isinstance(m.channel,discord.DMChannel) and not p['collect_dms']:return False
  if p['allowed_channel_ids'] and str(m.channel.id) not in map(str,p['allowed_channel_ids']):return False
  if str(m.channel.id) in map(str,p['denied_channel_ids']):return False
  return True
 async def on_message(self,m):
  if not self.allowed(m):return
  if m.author.bot:return
            
  parts=m.content.strip().split()
  if len(parts) >= 2 and parts[0].lower() == '/ep' and parts[1].lower() == 'channel':
   if not isinstance(m.author, discord.Member) or not m.author.guild_permissions.administrator:
    await m.channel.send('このコマンドは管理者専用です。');return
   b=self.cfg['bot'];ignored=b.setdefault('ignored_channel_ids', [])
   action=parts[2].lower() if len(parts) >= 3 else 'list'
   if action == 'list':
    if not ignored:
     await m.channel.send('返信しないチャンネルは設定されていません。');return
    names=[]
    for cid in ignored:
     ch=m.guild.get_channel(int(cid)) if m.guild else None
     names.append(f'{ch.mention} (`{cid}`)' if ch else f'`{cid}`')
    await m.channel.send('返信しないチャンネル:\n'+'\n'.join(names));return
   if action in ('deny','add','off'):
    cid=parts[3] if len(parts) >= 4 else str(m.channel.id)
    cid=cid.strip('<>#&!')
    if not cid.isdigit():
     await m.channel.send('チャンネルIDを指定してください。例: `/ep channel deny 123456789012345678`');return
    if cid not in map(str, ignored):
     ignored.append(int(cid))
     await asyncio.to_thread(self._save_config)
    await m.channel.send(f'このチャンネル (`{cid}`) を返信対象外にしました。');return
   if action in ('allow','remove','on'):
    cid=parts[3] if len(parts) >= 4 else str(m.channel.id)
    cid=cid.strip('<>#&!')
    if not cid.isdigit():
     await m.channel.send('チャンネルIDを指定してください。');return
    b['ignored_channel_ids']=[x for x in ignored if str(x) != cid]
    await asyncio.to_thread(self._save_config)
    await m.channel.send(f'このチャンネル (`{cid}`) を返信対象に戻しました。');return
   await m.channel.send('使用法: `/ep channel deny [channel_id]` / `/ep channel allow [channel_id]` / `/ep channel list`');return
  if str(m.channel.id) in map(str, self.cfg['bot'].get('ignored_channel_ids', [])):
   return
  if len(parts) >= 2 and parts[0].lower() == '/ep' and parts[1].lower() == 'probability':
   if not isinstance(m.author, discord.Member) or not m.author.guild_permissions.administrator:
    await m.channel.send('このコマンドは管理者専用です。');return
   b=self.cfg['bot']
   if len(parts) == 2:
    await m.channel.send(f"通常メッセージへの割り込み返信確率: {float(b.get('occasional_reply_probability', 0.08))*100:.1f}%")
    return
   try:
    value=float(parts[2])
   except ValueError:
    await m.channel.send('使用法: /ep probability <0〜100>');return
   if not 0 <= value <= 100:
    await m.channel.send('確率は0〜100%で指定してください。');return
   b['occasional_reply_probability']=value/100.0
   await asyncio.to_thread(self._save_config)
   await m.channel.send(f'通常メッセージへの割り込み返信確率を {value:g}% に設定しました。')
   return
  a=analyze(m.content,self.core.known());self.core.add_message(id=str(m.id),channel=str(m.channel.id),guild=str(m.guild.id) if m.guild else '',author=str(m.author.id),text=m.content,ts=m.created_at.timestamp())
  cid=str(m.channel.id)
                                                    
                                        
                                       
                                  
  self.graph.add(m.author.id,m.content,previous_text=self._last_text.get(cid))
  self._last_text[cid]=m.content
                                                                                           
                                           
  import re
  text=m.content.strip()
  q=re.match(r'^[「『]([^「」『』=]{1,80})[」』]\s*(?:=|は)\s*(.{1,300})$',text)
  if not q:
   q=re.match(r'^([^「」『』=\s]{1,80})=\s*(.{1,300})$',text)
  if q:
   self.core.add_term(q.group(1).strip(),q.group(2).strip())
   if self.user and self.user.mentioned_in(m): await m.channel.send('覚えました。'); return
   return
  b=self.cfg['bot'];mentioned=self.user and self.user.mentioned_in(m)
                                    
  if not mentioned:
   if b.get('reply_only_when_mentioned', False):
    if not b.get('occasional_reply', True):return
    if random.random() >= float(b.get('occasional_reply_probability', 0.08)):return
   elif not b.get('occasional_reply', True):return
  if not b['auto_reply']:return
  ctx={'user':str(m.author.id),'channel':str(m.channel.id),'guild':str(m.guild.id) if m.guild else ''}
  ex=self.core.search(a['semantic'],ctx,b['candidate_limit'])
                                  
                                   
  ask_prob=float(b.get('unknown_term_question_probability',0.3))
  if a['unknown'] and random.random()<ask_prob:
   reply=f'「{a["unknown"][0]}」ってどういう意味？'
  else:
   history=[x.content async for x in m.channel.history(limit=b['history_messages'],before=m)]
   prompt='Discord会話に自然に短く返信。\n会話:\n'+'\n'.join(reversed(history))+f'\n現在:{m.content}\n経験:\n'+''.join(f"{e['input']} -> {e['response']} [{e['result']}]\n" for e in ex[:8])+'返信:'
   reply=None
   if self.moderu:
    x=torch.tensor([self.tok.encode(prompt)],dtype=torch.long)
    with torch.no_grad(): y=self.moderu.seisei(x,max_new_tokens=int(self.cfg.get('inference',{}).get('max_new_tokens',80)),temperature=float(self.cfg.get('inference',{}).get('temperature',.75)),top_k=int(self.cfg.get('inference',{}).get('top_k',30)),repetition_penalty=float(self.cfg.get('inference',{}).get('repetition_penalty',1.3)),no_repeat_ngram_size=int(self.cfg.get('inference',{}).get('no_repeat_ngram_size',3)),eos_token_id=self.tok.sp.eos_id())
    reply=self.tok.decode(y[0].tolist()[len(x[0]):]).strip()
   if not reply:
    graph_candidates=self.graph.next_candidates(m.content,limit=1)
    reply=ex[0]['response'] if ex else (graph_candidates[0]['dst_text'] if graph_candidates else ('どういう意味？' if a['semantic']=='question' else 'なるほど。'))
  sent=await m.channel.send(reply[:b['max_reply_chars']]);self.pending[sent.id]=(m,a,reply,ctx);asyncio.create_task(self.observe(sent.id))

 async def _gakushu_background(self):
  async with self._gakushu_lock:
   try:
    from gakushu import gakushu_once
    await asyncio.to_thread(gakushu_once, True)
                                             
                                                  
                                  
    await asyncio.to_thread(self._reload_moderu)
   except Exception as e:
    print('[gakushu] background update failed:', e)
 async def observe(self,sid):
  m,a,reply,ctx=self.pending[sid];await asyncio.sleep(self.cfg['bot']['observation_delay_seconds']);msgs=[]
  async for x in m.channel.history(limit=self.cfg['bot']['observation_max_messages'],after=m):
   if x.id!=sid and not x.author.bot:msgs.append(x.content)
                                                                                    
  positive=('いいね','それな','了解','わかる','助かる','ありがとう','正解','完璧','なるほど','その通り','草','w')
  negative=('違う','いや、それは','間違い','間違ってる','無理','ダメ','違います','訂正','じゃない')
  pos=sum(sum(t.lower().count(z.lower()) for z in positive) for t in msgs)
  neg=sum(sum(t.lower().count(z.lower()) for z in negative) for t in msgs)
                                                                     
  score=max(-1.0,min(1.0,(pos-neg*1.5)/max(1,len(msgs))))
  result='successful' if score>=.45 else 'failed' if score<=-.45 else 'neutral'
  confidence=.75 if any(any(z in t for z in negative) for t in msgs) else (.65 if any(any(z in t.lower() for z in positive) for t in msgs) else (.45 if msgs else .25))
  self.core.add_experience(semantic=a['semantic'],input=m.content,context=ctx,response=reply,reactions=msgs,next_messages=msgs,result=result,score=score,confidence=confidence,fingerprint=a['fingerprint'])
  self._experience_count += 1
  self._new_message_count += 1
                                                                      
  if self._experience_count % int(self.cfg.get('gakushuing',{}).get('export_every_experiences',5)) == 0:
   try:
    self.core.export(min_score=float(self.cfg.get('experience',{}).get('export_score',.45)), min_conf=float(self.cfg.get('experience',{}).get('export_confidence',.45)))
    self.core.export_conversation_windows(window=int(self.cfg.get('gakushuing',{}).get('conversation_window',12)))
   except Exception as e: print('[experience] export failed:',e)
  every=int(self.cfg.get('gakushuing',{}).get('gakushu_every_experiences',100))
  if every > 0 and self._experience_count % every == 0 and not self._gakushu_lock.locked():
   asyncio.create_task(self._gakushu_background())
