import sqlite3 ,time ,json ,math ,hashlib ,re 
from pathlib import Path 

class ExperienceCore :
    def __init__ (self ,path ='data/experience.db'):
        Path (path ).parent .mkdir (parents =True ,exist_ok =True )
        self .db =sqlite3 .connect (path ,check_same_thread =False )
        self .db .row_factory =sqlite3 .Row 
        self .init ()

    def init (self ):
        self .db .executescript ('''
        CREATE TABLE IF NOT EXISTS experiences(
          id INTEGER PRIMARY KEY, semantic TEXT,input TEXT,context TEXT,response TEXT,
          reactions TEXT,next_messages TEXT,result TEXT,score REAL,frequency INTEGER,
          confidence REAL,ts REAL,fingerprint TEXT);
        CREATE INDEX IF NOT EXISTS ex_sem ON experiences(semantic);
        CREATE INDEX IF NOT EXISTS ex_fp ON experiences(fingerprint);
        CREATE TABLE IF NOT EXISTS terms(term TEXT PRIMARY KEY,meaning TEXT,semantic TEXT,confidence REAL,source TEXT,ts REAL);
        CREATE TABLE IF NOT EXISTS messages(id TEXT PRIMARY KEY,channel TEXT,guild TEXT,author TEXT,text TEXT,ts REAL);
        CREATE INDEX IF NOT EXISTS msg_channel_ts ON messages(channel,ts);
        CREATE TABLE IF NOT EXISTS profiles(kind TEXT,key TEXT PRIMARY KEY,stats TEXT,ts REAL);
        CREATE TABLE IF NOT EXISTS training_meta(key TEXT PRIMARY KEY,value TEXT);
        ''')
        self .db .commit ()

    def add_term (self ,t ,m ,s ='learned',c =.9 ,source ='human'):
        self .db .execute ('INSERT OR REPLACE INTO terms VALUES(?,?,?,?,?,?)',(t ,m ,s ,c ,source ,time .time ()))
        self .db .commit ()

    def known (self ):
        return {r ['term']for r in self .db .execute ('SELECT term FROM terms')}

    def add_message (self ,**x ):
        self .db .execute ('INSERT OR REPLACE INTO messages VALUES(?,?,?,?,?,?)',(
        x ['id'],x .get ('channel'),x .get ('guild'),x .get ('author'),x .get ('text'),x .get ('ts',time .time ())))
        self .db .commit ()

    def add_experience (self ,**x ):
        fp =x .get ('fingerprint')or hashlib .sha1 ((x ['input']+'\n'+x ['response']).encode ('utf8')).hexdigest ()
        old =self .db .execute ('SELECT id,frequency FROM experiences WHERE fingerprint=? ORDER BY id DESC LIMIT 1',(fp ,)).fetchone ()
        if old :
            self .db .execute ('UPDATE experiences SET frequency=frequency+1,score=?,confidence=?,ts=?,reactions=?,next_messages=?,result=? WHERE id=?',(
            x ['score'],x ['confidence'],time .time (),json .dumps (x .get ('reactions',[]),ensure_ascii =False ),
            json .dumps (x .get ('next_messages',[]),ensure_ascii =False ),x ['result'],old ['id']))
        else :
            self .db .execute ('INSERT INTO experiences(semantic,input,context,response,reactions,next_messages,result,score,frequency,confidence,ts,fingerprint) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(
            x ['semantic'],x ['input'],json .dumps (x .get ('context',{}),ensure_ascii =False ),x ['response'],
            json .dumps (x .get ('reactions',[]),ensure_ascii =False ),json .dumps (x .get ('next_messages',[]),ensure_ascii =False ),
            x ['result'],x ['score'],1 ,x ['confidence'],time .time (),fp ))
        self .db .commit ()

    def search (self ,semantic ,ctx ,limit =16 ,query =None ):
        rows =self .db .execute ('SELECT * FROM experiences ORDER BY ts DESC LIMIT ?',(max (limit *30 ,100 ),)).fetchall ()
        now =time .time ();out =[]
        query_text=re.sub(r'\s+','',query or '')
        qgrams=set(query_text[i:i+2] for i in range(max(0,len(query_text)-1)))
        qgrams|=set(query_text[i:i+3] for i in range(max(0,len(query_text)-2)))
        for r in rows :
            age =(now -r ['ts'])/86400 
            decay =math .exp (-max (0 ,age )/120 )
            c =json .loads (r ['context']or '{}')
            sem =1.0 if r ['semantic']==semantic else .15 
            match =.20 *(c .get ('channel')==ctx .get ('channel'))+.15 *(c .get ('guild')==ctx .get ('guild'))+.15 *(c .get ('user')==ctx .get ('user'))
            text=re.sub(r'\s+','',r['input']+' '+r['response'])
            tgrams=set(text[i:i+2] for i in range(max(0,len(text)-1)))
            tgrams|=set(text[i:i+3] for i in range(max(0,len(text)-2)))
            overlap=len(qgrams&tgrams)/max(1,len(qgrams)) if qgrams else 0.0
            score=(r['score']*.30+r['confidence']*.18+min(1,r['frequency']/10)*.12+sem*.16+match+overlap*.29)*decay 
            out .append ((score ,dict (r )))
        return [x [1 ]for x in sorted (out ,key =lambda x :x [0 ],reverse =True )[:limit ]]

    def export (self ,path ='data/curated_experiences.jsonl',min_score =.45 ,min_conf =.45 ,discord_path ='data/discord.jsonl'):
        rows =self .db .execute ('SELECT * FROM experiences ORDER BY ts ASC').fetchall ()
        Path (path ).parent .mkdir (parents =True ,exist_ok =True )
        Path (discord_path ).parent .mkdir (parents =True ,exist_ok =True )
        selected =[r for r in rows if r ['confidence']>=min_conf and (r ['score']>=min_score or r ['result']in ('neutral','successful'))]
        with open (path ,'w',encoding ='utf8')as f :
            for r in selected :
                f .write (json .dumps ({'instruction':'Discord会話に自然に返信する','input':r ['input'],
                'context':json .loads (r ['context']or '{}'),'output':r ['response'],'result':r ['result'],
                'score':r ['score'],'confidence':r ['confidence']},ensure_ascii =False )+'\n')
        with open (discord_path ,'w',encoding ='utf8')as f :
            for r in selected :
                text =f"ユーザー: {r ['input']}\nAI: {r ['response']}"
                f .write (json .dumps ({'text':text ,'input':r ['input'],'response':r ['response'],
                'result':r ['result'],'score':r ['score'],'confidence':r ['confidence'],'timestamp':r ['ts']},ensure_ascii =False )+'\n')
        return len (selected )

    def export_conversation_windows (self ,path ='data/discord_conversations.jsonl',window =12 ):
        Path (path ).parent .mkdir (parents =True ,exist_ok =True )
        rows =self .db .execute ('SELECT * FROM messages ORDER BY channel,ts ASC').fetchall ()
        written =0 ;seen =set ()
        with open (path ,'w',encoding ='utf8')as f :
            by ={}
            for r in rows :by .setdefault (r ['channel'],[]).append (r )
            for channel ,items in by .items ():
                for i in range (0 ,len (items ),max (1 ,window //2 )):
                    chunk =items [i :i +window ]
                    if len (chunk )<2 :continue 
                    lines =[f"ユーザー{j +1 }: {x ['text']}"for j ,x in enumerate (chunk )if x ['text'].strip ()]
                    if len (lines )<2 :continue 
                    text ='\n'.join (lines )
                    fp =hashlib .sha1 (text .encode ('utf8')).hexdigest ()
                    if fp in seen :continue 
                    seen .add (fp )
                    f .write (json .dumps ({'text':text ,'channel':channel ,'timestamp':chunk [-1 ]['ts'],'type':'conversation_window'},ensure_ascii =False )+'\n')
                    written +=1 
        return written 

    def export_all_training_data (self ):
        self .export ()
        return self .export_conversation_windows ()

    def forget_channel (self ,ch ):
        self .db .execute ('DELETE FROM messages WHERE channel=?',(str (ch ),));self .db .commit ()
