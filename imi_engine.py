import re ,hashlib 
RULES =[('greeting',r'^(おはよ|こんにちは|こんばんは|やあ|おつ)'),('game_invite',r'(ゲーム|マイクラ|minecraft|遊ぶ|やる).*(する|やろ|やる|行く)'),('question',r'(？|\?)$|^(なん|なに|どう|いつ|どこ|誰|なぜ|なんで)'),('agreement',r'^(うん|はい|そう|それな|わかる|了解|いいよ|OK|おけ)'),('disagreement',r'(違う|いや|無理|ダメ|そうじゃない)'),('joke',r'(w+|草|笑|www|ネタ|冗談)')]

URL_RE =re .compile (r'https?://\S+|www\.\S+')




STOPWORDS ={
'http','https','www','com','net','org','jp','co','html','htm','php','id','url',
'png','jpg','jpeg','gif','webp','pdf','mp4','mp3','gz','zip','txt',
'minecraft','discord','python','llm','ai','bot','ok','okay','lol','ww','www2',
}

def norm (s ):return re .sub (r'\s+',' ',s .lower ().strip ())

def analyze (s ,known =set ()):
 t =norm (s )
 t_no_url =URL_RE .sub (' ',t )
 sem ='general_chat'
 for n ,p in RULES :
  if re .search (p ,t ,re .I ):sem =n ;break 
 words =re .findall (r'[一-龯々ぁ-んァ-ヶA-Za-z0-9_]+',t_no_url )
 unknown =[w for w in words if w not in known and re .fullmatch (r'[A-Za-z_][A-Za-z0-9_]{2,}',w )and w .lower ()not in STOPWORDS ]
 return {'semantic':sem ,'unknown':unknown ,'fingerprint':hashlib .sha256 (t .encode ()).hexdigest ()[:24 ]}
