from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
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
    token = db.relationship('MusicToken', backref='entry', uselist=False)

class MusicToken(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    entry_id = db.Column(db.Integer, db.ForeignKey('diary_entry.id'), unique=True)
    token_hash = db.Column(db.String(64), unique=True)
    music_params = db.Column(db.Text) # JSON string
    price = db.Column(db.Float, default=0.01)
    owner = db.Column(db.String(50), default='Original Artist')
    is_for_sale = db.Column(db.Boolean, default=True)

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

def get_llm_response(user_text, round_num, context):
    if not GROQ_API_KEY:
        # Fallback to a much smarter rule-based counselor if no API key
        return f"第 {round_num} 轮：我深刻理解这种 {context} 的感受。在这一步，我们让和弦更加深沉。你能谈谈这背后的具体瞬间吗？"
    
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        prompt = f"你是一位高级心理咨询师。用户正在参与一个‘情绪燃料实验’。这是第{round_num}轮对话。用户的情绪关键词是{context}。用户的输入是：‘{user_text}’。请用一段充满共情、富有诗意且专业的中文回复（50字以内），引导用户继续深入探索。"
        data = {
            "model": "llama-3-8b-8192",
            "messages": [{"role": "system", "content": "你是一位富有共情力的心理咨询师，语言优美且具有深度。"}, {"role": "user", "content": prompt}],
            "max_tokens": 100
        }
        res = requests.post(url, headers=headers, json=data, timeout=5)
        return res.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"第 {round_num} 轮：情感的波动已被捕捉。这一步的和弦充满了共鸣，请继续分享你的感受。"

# Music Theory Mappings
GENRES = {
    'ambient': {'tempo': [60, 80], 'synth': 'sine', 'reverb': 0.8},
    'lo-fi': {'tempo': [80, 95], 'synth': 'triangle', 'reverb': 0.4},
    'cinematic': {'tempo': [70, 90], 'synth': 'sawtooth', 'reverb': 0.9}
}

SCALES = {
    'bright': ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4'], # C Major
    'peaceful': ['F3', 'G3', 'A3', 'Bb3', 'C4', 'D4', 'E4'], # F Major/Lydian
    'nostalgic': ['A3', 'B3', 'C4', 'D4', 'E4', 'F4', 'G4'], # A Minor
    'intense': ['D3', 'Eb3', 'F3', 'G3', 'A3', 'Bb3', 'C4'] # D Phrygian
}

@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data.get('text', '').lower()
    round_num = data.get('round', 1)
    genre = data.get('genre', 'ambient')
    
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
    if context == 'anger' or polarity < -0.4: mode = 'intense'
    elif context == 'missing' or context == 'family' or polarity < 0: mode = 'nostalgic'
    elif context == 'joy' or polarity > 0.4: mode = 'bright'
    else: mode = 'peaceful'

    scale = SCALES[mode]
    
    # Chord Progression Logic: Progressive addition
    # Progression: I -> IV -> V -> vi -> IV -> I -> ...
    progression_indices = [0, 3, 4, 5, 3, 0, 1, 4]
    base_idx = progression_indices[(round_num - 1) % len(progression_indices)]
    
    # Construct a chord (Triad) from the scale
    root = scale[base_idx]
    third = scale[(base_idx + 2) % len(scale)]
    fifth = scale[(base_idx + 4) % len(scale)]
    chord = [root, third, fifth]

    music_params = {
        'tempo': GENRES[genre]['tempo'][0] + (polarity * 10),
        'new_chord': chord,
        'mode': mode,
        'genre': genre,
        'energy_gain': abs(polarity) * 15 + 10 
    }
    
    ai_response = get_llm_response(text, round_num, context)
    
    return jsonify({
        'polarity': polarity,
        'music_params': music_params,
        'ai_response': ai_response,
        'regional_data': REGIONAL_SENTIMENT
    })

@app.route('/api/global_sentiment', methods=['GET'])
def get_global_sentiment():
    return jsonify(REGIONAL_SENTIMENT)

@app.route('/api/save', methods=['POST'])
def save_entry():
    data = request.json
    content = data.get('text', '') # The last round text
    full_history = data.get('full_history', []) # List of all round texts
    music_params = data.get('music_params', {}) # All chords, genre, etc.
    total_energy = data.get('total_energy', 0)
    
    # Analyze the overall sentiment from the whole history
    combined_text = " ".join(full_history) if full_history else content
    analysis = TextBlob(combined_text)
    
    # Create Diary Entry
    new_entry = DiaryEntry(
        content=combined_text,
        polarity=analysis.sentiment.polarity,
        subjectivity=analysis.sentiment.subjectivity
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

@app.route('/api/vault', methods=['GET'])
def get_vault():
    # In a real app, filter by user. Here we just show all created tokens.
    tokens = MusicToken.query.order_by(MusicToken.id.desc()).all()
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

if __name__ == '__main__':
    app.run(debug=True, port=5001)
