from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from textblob import TextBlob
import os
import json
import nltk

# Vercel-specific NLTK path
nltk_data_path = os.path.join('/tmp', 'nltk_data')
os.environ['NLTK_DATA'] = nltk_data_path

def init_resources():
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)
    
    try:
        nltk.data.find('tokenizers/punkt', paths=[nltk_data_path])
    except (LookupError, Exception):
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True)

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger', paths=[nltk_data_path])
    except (LookupError, Exception):
        nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path, quiet=True)
    
    with app.app_context():
        db.create_all()

# Setup directories
base_dir = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(os.path.dirname(base_dir), 'frontend', 'templates')

if os.environ.get('VERCEL'):
    db_path = '/tmp/emo.db'
else:
    db_path = os.path.join(base_dir, 'emo.db')

app = Flask(__name__, template_folder=template_dir)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)

db = SQLAlchemy(app)

# Resource initialization flag
resources_initialized = False

@app.before_request
def initialize():
    global resources_initialized
    if not resources_initialized:
        init_resources()
        resources_initialized = True

# Database Models
class DiaryEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    polarity = db.Column(db.Float)
    subjectivity = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_capsule = db.Column(db.Boolean, default=False)
    unlock_date = db.Column(db.DateTime, nullable=True)
    ai_tags = db.Column(db.String(100), default='Emotion') # Comma separated
    token = db.relationship('MusicToken', backref='entry', uselist=False)

class MusicToken(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    entry_id = db.Column(db.Integer, db.ForeignKey('diary_entry.id'), unique=True)
    token_hash = db.Column(db.String(64), unique=True)
    music_params = db.Column(db.Text) # JSON string
    price = db.Column(db.Float, default=0.01)
    owner = db.Column(db.String(50), default='Original Artist')
    is_for_sale = db.Column(db.Boolean, default=True)

class PaymentIntent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    amount = db.Column(db.Float, default=1.0)
    token_id = db.Column(db.Integer, db.ForeignKey('music_token.id'))
    status = db.Column(db.String(20), default='intent_clicked')

# Routes
@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'db_path': db_path,
        'template_dir': template_dir,
        'vercel': os.environ.get('VERCEL', 'no')
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/vault')
def vault():
    return render_template('vault.html')

@app.route('/diary')
def diary():
    return render_template('diary.html')

# Global/Mock Regional Sentiment Data
REGIONAL_SENTIMENT = {
    'Asia': {'happiness': 0.7, 'sadness': 0.2, 'energy': 0.8},
    'Europe': {'happiness': 0.5, 'sadness': 0.4, 'energy': 0.5},
    'Americas': {'happiness': 0.6, 'sadness': 0.3, 'energy': 0.7},
    'Global': {'happiness': 0.62, 'sadness': 0.31, 'energy': 0.68}
}

# Counselor Knowledge/Keywords
KEYWORDS = {
    'family': ['mom', 'dad', 'mother', 'father', 'parents', 'mama', 'papa', '妈妈', '爸爸', '父母', '家人'],
    'missing': ['miss', 'longing', 'missing', '想念', '思念'],
    'anger': ['angry', 'mad', 'furious', 'pissed', '气死', '生气', '愤怒', '火大'],
    'joy': ['happy', 'great', 'awesome', 'joy', '开心', '快乐', '太棒了', '舒服']
}

import requests

# LLM Config (Optional: User can set GROQ_API_KEY in Vercel/Local)
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

def get_llm_response(user_text, round_num, context, chat_history=[], mode='experiment'):
    if not GROQ_API_KEY:
        if mode == 'diary':
            return f"在这段文字里，我听到了你内心的{context}。今晚的月色很美，希望这段回响能带给你一丝安宁。"
        return f"作为你的心理陪伴者，我能感受到你此刻的{context}。这第 {round_num} 轮的旋律中，我为你加入了一丝深沉的基调。你能再深入聊聊那个让你产生这种感觉的具体瞬间吗？"
    
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        
        system_content = (
            "你是一位融合了人本主义和存在主义流派的高级心理咨询师。你的语言风格：专业、温暖、充满洞察力、富有诗意且精炼。 "
            "你的任务是通过对话： "
            "1. 展现极高的共情能力，精准捕捉用户言语背后的情感色彩。 "
            "2. 运用‘内容反应’和‘情感反应’技术，让用户感到被深度理解。 "
            "3. 语言要像深夜的耳语，含蓄而有力量。 "
            "4. 保持逻辑连贯。 "
            "5. 每次回复严控在 60 字以内。"
        )

        if mode == 'diary':
            user_prompt = (
                f"用户写下了一篇深夜日记：‘{user_text}’。情绪背景：{context}。 "
                "请以心理咨询师的身份，给出一个充满洞察力、温暖且能引发自我觉察的‘深夜回响’回复。 "
                "并以 JSON 格式附加在回复末尾（不要显示给用户，仅供机器读取）： "
                "{\"tags\": [\"标签1\", \"标签2\"], \"visual_prompt\": \"一段描述唯美、艺术化视觉意象的英文短语，用于文生图\"}"
            )
        else:
            user_prompt = f"对话轮次：{round_num}/5。情绪背景：{context}。用户说：‘{user_text}’。请以心理咨询师的身份给出回应。"

        messages = [{"role": "system", "content": system_content}]
        
        for turn in chat_history[-4:]:
            messages.append({"role": "user", "content": turn['user']})
            messages.append({"role": "assistant", "content": turn['ai']})
            
        messages.append({"role": "user", "content": user_prompt})
        
        data = {
            "model": "llama-3-8b-8192",
            "messages": messages,
            "max_tokens": 150,
            "temperature": 0.8
        }
        res = requests.post(url, headers=headers, json=data, timeout=7)
        return res.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"LLM Error: {e}")
        return "我听到了。在这一轮的波动中，旋律变得更加宽广。请继续告诉我，那对你意味着什么？"

# Music Theory Mappings (P3: 7th and 9th chords)
GENRES = {
    'ambient': {'tempo': [60, 80], 'synth': 'sine', 'reverb': 0.8},
    'lo-fi': {'tempo': [80, 95], 'synth': 'triangle', 'reverb': 0.4},
    'cinematic': {'tempo': [70, 90], 'synth': 'sawtooth', 'reverb': 0.9}
}

SCALES = {
    'bright': ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5', 'E5'], # C Major
    'peaceful': ['F3', 'G3', 'A3', 'Bb3', 'C4', 'D4', 'E4', 'F4', 'G4', 'A4'], # F Major/Lydian
    'nostalgic': ['A3', 'B3', 'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'], # A Minor
    'intense': ['D3', 'Eb3', 'F3', 'G3', 'A3', 'Bb3', 'C4', 'D4', 'Eb4', 'F4'] # D Phrygian
}

@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data.get('text', '').lower()
    round_num = data.get('round', 1)
    genre = data.get('genre', 'ambient')
    chat_history = data.get('history', [])
    mode = data.get('mode', 'experiment')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    
    # Context Detection
    context = 'general'
    for key, words in KEYWORDS.items():
        if any(w in text for w in words):
            context = key
            break

    # Determine Mode and Scale
    if context == 'anger' or polarity < -0.4: mood = 'intense'
    elif context == 'missing' or context == 'family' or polarity < 0: mood = 'nostalgic'
    elif context == 'joy' or polarity > 0.4: mood = 'bright'
    else: mode_key = 'peaceful' # Renamed to avoid conflict with 'mode' param

    scale = SCALES[mode_key if 'mode_key' in locals() else 'peaceful'] # Corrected
    if context == 'anger' or polarity < -0.4: scale = SCALES['intense']
    elif context == 'missing' or context == 'family' or polarity < 0: scale = SCALES['nostalgic']
    elif context == 'joy' or polarity > 0.4: scale = SCALES['bright']
    
    # Chord Progression Logic: Progressive addition
    progression_indices = [0, 3, 4, 5, 3, 0, 1, 4]
    base_idx = progression_indices[(round_num - 1) % len(progression_indices)]
    
    # Construct a complex chord (7th or 9th)
    root = scale[base_idx]
    third = scale[(base_idx + 2) % len(scale)]
    fifth = scale[(base_idx + 4) % len(scale)]
    seventh = scale[(base_idx + 6) % len(scale)]
    ninth = scale[(base_idx + 8) % len(scale)]
    
    # Add notes based on round complexity
    if round_num < 3:
        chord = [root, third, fifth, seventh]
    else:
        chord = [root, third, fifth, seventh, ninth]
    
    # Melody Generation Params (P2 Upgrade)
    melody_params = {
        'root_index': base_idx,
        'pitch_offset': int(polarity * 3),
        'rhythm_density': 0.3 + (abs(polarity) * 0.5),
        'note_length': '4n' if abs(polarity) < 0.5 else '8n',
        'expressivity': subjectivity
    }

    music_params = {
        'tempo': GENRES[genre]['tempo'][0] + (polarity * 10),
        'new_chord': chord,
        'mode': mode, # Keep for consistency but use mode_key for internal logic if needed
        'genre': genre,
        'energy_gain': abs(polarity) * 15 + 10,
        'melody': melody_params
    }
    
    ai_response = get_llm_response(text, round_num, context, chat_history, mode=mode)
    
    # Parse AI JSON if in diary mode
    parsed_ai = {'text': ai_response, 'tags': ['深夜', '觉察'], 'visual_url': 'https://images.unsplash.com/photo-1518173946687-a4c8a9ba332f'}
    if mode == 'diary' and '{' in ai_response:
        try:
            parts = ai_response.split('{')
            text_part = parts[0].strip()
            json_part = '{' + parts[1]
            ai_data = json.loads(json_part)
            parsed_ai['text'] = text_part
            parsed_ai['tags'] = ai_data.get('tags', [])
            v_prompt = ai_data.get('visual_prompt', 'calm sea night rain')
            parsed_ai['visual_url'] = f"https://source.unsplash.com/1600x900/?{v_prompt.replace(' ', ',')}"
        except:
            pass

    return jsonify({
        'polarity': polarity,
        'subjectivity': subjectivity,
        'music_params': music_params,
        'ai_response': parsed_ai['text'],
        'ai_tags': parsed_ai['tags'],
        'visual_url': parsed_ai['visual_url'],
        'regional_data': REGIONAL_SENTIMENT
    })

@app.route('/api/global_sentiment', methods=['GET'])
def get_global_sentiment():
    return jsonify(REGIONAL_SENTIMENT)

@app.route('/api/save', methods=['POST'])
def save_entry():
    data = request.json
    content = data.get('text', '')
    full_history = data.get('full_history', [])
    music_params = data.get('music_params', {})
    total_energy = data.get('total_energy', 0)
    
    # Capsule Data
    is_capsule = data.get('is_capsule', False)
    unlock_days = data.get('unlock_days', 0)
    unlock_date = datetime.utcnow() + timedelta(days=unlock_days) if is_capsule else None
    ai_tags = ",".join(data.get('ai_tags', ['Emotion']))

    combined_text = " ".join(full_history) if full_history else content
    analysis = TextBlob(combined_text)
    
    new_entry = DiaryEntry(
        content=combined_text,
        polarity=analysis.sentiment.polarity,
        subjectivity=analysis.sentiment.subjectivity,
        is_capsule=is_capsule,
        unlock_date=unlock_date,
        ai_tags=ai_tags
    )
    db.session.add(new_entry)
    db.session.commit()
    
    # Add energy and summary to music_params for storage
    music_params['total_energy'] = total_energy
    music_params['summary'] = data.get('summary', '这是一段珍贵的情绪燃料。')
    
    # Create Token
    token_hash = hex(hash(combined_text + str(datetime.now())))[2:]
    new_token = MusicToken(
        entry_id=new_entry.id,
        token_hash=token_hash,
        music_params=json.dumps(music_params),
        price=round(0.01 + abs(analysis.sentiment.polarity) * 0.1, 3)
    )
    db.session.add(new_token)
    db.session.commit()
    
    return jsonify({
        'status': 'success',
        'token_id': new_token.id,
        'token_hash': token_hash
    })

@app.route('/api/tokens', methods=['GET'])
def get_tokens():
    tokens = MusicToken.query.filter_by(is_for_sale=True).all()
    result = []
    for t in tokens:
        result.append({
            'id': t.id,
            'token_hash': t.token_hash,
            'price': t.price,
            'owner': t.owner,
            'music_params': json.loads(t.music_params),
            'timestamp': t.entry.timestamp.strftime("%Y-%m-%d %H:%M")
        })
    return jsonify(result)

@app.route('/api/buy', methods=['POST'])
def buy_token():
    data = request.json
    token_id = data.get('token_id')
    new_owner = data.get('buyer', 'Collector')
    
    token = MusicToken.query.get(token_id)
    if token:
        token.owner = new_owner
        token.is_for_sale = False
        db.session.commit()
        return jsonify({'status': 'purchased', 'token_id': token_id})
    return jsonify({'error': 'Token not found'}), 404

@app.route('/api/record_intent', methods=['POST'])
def record_intent():
    data = request.json
    new_intent = PaymentIntent(
        token_id=data.get('token_id'),
        amount=data.get('amount', 1.0),
        status='intent_confirmed'
    )
    db.session.add(new_intent)
    db.session.commit()
    
    # In a real app, this is where we'd trigger a 402 Payment Required or Mock Payment
    return jsonify({
        'status': 'intent_recorded',
        'message': 'We are currently in a high-fidelity validation phase. No real money has been charged.'
    })

@app.route('/api/vault', methods=['GET'])
def get_vault():
    now = datetime.utcnow()
    tokens = MusicToken.query.order_by(MusicToken.id.desc()).all()
    result = []
    for t in tokens:
        # Check if it's a locked capsule
        is_locked = False
        if t.entry.is_capsule and t.entry.unlock_date > now:
            is_locked = True
        
        result.append({
            'id': t.id,
            'token_hash': t.token_hash,
            'price': t.price,
            'owner': t.owner,
            'music_params': json.loads(t.music_params),
            'timestamp': t.entry.timestamp.strftime("%Y-%m-%d %H:%M"),
            'is_locked': is_locked,
            'unlock_date': t.entry.unlock_date.strftime("%Y-%m-%d %H:%M") if t.entry.unlock_date else None,
            'ai_tags': t.entry.ai_tags.split(',') if t.entry.ai_tags else []
        })
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
