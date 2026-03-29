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

@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data.get('text', '').lower()
    round_num = data.get('round', 1)
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    
    # Detect Keywords for Context
    context = 'general'
    for key, words in KEYWORDS.items():
        if any(w in text for w in words):
            context = key
            break

    # Advanced Music Parameters based on Context and Sentiment
    chord = []
    mode = 'peaceful'
    
    if context == 'anger' or polarity < -0.5:
        # Dissonant or heavy: Dm or Tritone influenced
        chord = ['D3', 'F3', 'Ab3'] if round_num % 2 == 0 else ['G2', 'B2', 'Db3']
        mode = 'intense'
    elif context == 'missing' or context == 'family' or polarity < 0:
        # Melancholic/Soft: Am or Em
        chord = ['A3', 'C4', 'E4'] if round_num % 2 == 0 else ['E3', 'G3', 'B3']
        mode = 'nostalgic'
    elif context == 'joy' or polarity > 0.3:
        # Bright: C Major or G Major
        chord = ['C4', 'E4', 'G4'] if round_num % 2 == 0 else ['G3', 'B3', 'D4']
        mode = 'bright'
    else:
        chord = ['F3', 'A3', 'C4'] if round_num % 2 == 0 else ['C4', 'E4', 'G4']
        mode = 'peaceful'

    music_params = {
        'tempo': 70 + (polarity * 30) if mode != 'intense' else 110,
        'scale': 'major' if polarity >= 0 else 'minor',
        'new_chord': chord,
        'mode': mode,
        'reverb': 0.7 + (subjectivity * 0.2),
        'energy_gain': abs(polarity) * 15 + 10 
    }
    
    # Context-Aware AI Counselor Responses
    responses = {
        'general': [
            f"第 {round_num} 轮：我听到了。这种感觉很真实，能再深入一点吗？",
            f"第 {round_num} 轮：这种情绪在你的和弦中留下了痕迹。继续说下去，我在听。",
            f"第 {round_num} 轮：沉静的力量在滋长。这一步很重要。"
        ],
        'anger': [
            f"第 {round_num} 轮：我感受到了你的愤怒，这股力量非常强烈。这种张力会被转化为更深沉的音符，你想谈谈愤怒背后的原因吗？",
            f"第 {round_num} 轮：愤怒往往是受伤的保护壳。让我们用这段不协和的和弦来释放它。继续说吧。"
        ],
        'missing': [
            f"第 {round_num} 轮：想念是一种温柔的痛。这段旋律加入了一些怀旧的色彩，那种思念此刻在你的心中是什么样子的？",
            f"第 {round_num} 轮：思念让和弦变得悠长。这种连接感是独一无二的。能告诉我关于她的一个细节吗？"
        ],
        'family': [
            f"第 {round_num} 轮：家人的羁绊总是最深。这段音乐里加入了一些温暖但略显厚重的底色。这种情感对你意味着什么？"
        ],
        'joy': [
            f"第 {round_num} 轮：这种光芒在你的文字中闪烁。明亮的音程已经加入。请尽情沉浸在这种喜悦中，再告诉我一些吧！"
        ]
    }
    
    # Pick a response based on context and round
    available_responses = responses.get(context, responses['general'])
    ai_response = available_responses[round_num % len(available_responses)]
    
    return jsonify({
        'polarity': polarity,
        'subjectivity': subjectivity,
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
    content = data.get('text')
    params = data.get('music_params')
    
    analysis = TextBlob(content)
    
    # Create Diary Entry
    new_entry = DiaryEntry(
        content=content,
        polarity=analysis.sentiment.polarity,
        subjectivity=analysis.sentiment.subjectivity
    )
    db.session.add(new_entry)
    db.session.commit()
    
    # Create Token
    token_hash = hex(hash(content + str(datetime.now())))[2:]
    new_token = MusicToken(
        entry_id=new_entry.id,
        token_hash=token_hash,
        music_params=json.dumps(params),
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
