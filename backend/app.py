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

@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    
    # Advanced Music Parameters
    # Positive -> Major, Faster, Bright
    # Negative -> Minor, Slower, Dark
    # Subjective -> More reverb, complex chords
    
    chords = []
    if polarity > 0.3:
        # Uplifting: I - V - vi - IV
        chords = [['C4', 'E4', 'G4'], ['G3', 'B3', 'D4'], ['A3', 'C4', 'E4'], ['F3', 'A3', 'C4']]
        mode = 'bright'
    elif polarity > -0.3:
        # Neutral/Peaceful: I - IV - I - V
        chords = [['C4', 'E4', 'G4'], ['F3', 'A3', 'C4'], ['C4', 'E4', 'G4'], ['G3', 'B3', 'D4']]
        mode = 'peaceful'
    else:
        # Melancholic: i - VI - III - VII
        chords = [['A3', 'C4', 'E4'], ['F3', 'A3', 'C4'], ['C4', 'E4', 'G4'], ['G3', 'B3', 'D4']]
        mode = 'melancholic'

    music_params = {
        'tempo': 60 + (polarity * 20), # 40-80 BPM for soothing
        'scale': 'major' if polarity >= 0 else 'minor',
        'chords': chords,
        'mode': mode,
        'reverb': 0.6 + (subjectivity * 0.3),
        'pad_volume': -25,
        'lead_volume': -15,
        'weather': 'snow' if polarity < -0.2 else 'rain' if subjectivity > 0.6 else 'clear'
    }
    
    # AI Counselor Simulation Response
    responses = {
        'bright': "我能感受到你文字中透出的光。这种积极的能量像是一段明亮的和弦，你愿意多聊聊让你开心的那个瞬间吗？",
        'peaceful': "你的内心此刻似乎很平静。这种宁静是极其宝贵的，像是清晨的微风。在这样的时刻，你通常会思考些什么？",
        'melancholic': "听起来你正在经历一段沉重的时光。没关系，每个人都有这样的时刻。就像阴雨天，它也是自然的一部分。你想释放掉这种情绪吗？"
    }
    
    return jsonify({
        'polarity': polarity,
        'subjectivity': subjectivity,
        'music_params': music_params,
        'ai_response': responses.get(mode, "我在听。继续说下去...")
    })

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
