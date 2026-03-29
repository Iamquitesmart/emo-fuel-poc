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

@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data.get('text', '')
    round_num = data.get('round', 1)
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    
    # Advanced Music Parameters based on current round
    # We generate a unique chord for THIS round to be added to the sequence
    if polarity > 0.3:
        chord = ['C4', 'E4', 'G4'] if round_num % 2 == 0 else ['G3', 'B3', 'D4']
        mode = 'bright'
    elif polarity > -0.3:
        chord = ['F3', 'A3', 'C4'] if round_num % 2 == 0 else ['C4', 'E4', 'G4']
        mode = 'peaceful'
    else:
        chord = ['A3', 'C4', 'E4'] if round_num % 2 == 0 else ['D3', 'F3', 'A3']
        mode = 'melancholic'

    music_params = {
        'tempo': 60 + (polarity * 20),
        'scale': 'major' if polarity >= 0 else 'minor',
        'new_chord': chord,
        'mode': mode,
        'reverb': 0.6 + (subjectivity * 0.3),
        'energy_gain': abs(polarity) * 10 + 5 # Electricity/Fuel conversion value
    }
    
    responses = {
        'bright': f"第 {round_num} 轮：我听到了你内心的欢愉。这段旋律中加入了一组明亮的和弦。继续分享，让我们完成这首曲子。",
        'peaceful': f"第 {round_num} 轮：沉静的力量在滋长。这一组和弦非常稳健。你还想表达什么？",
        'melancholic': f"第 {round_num} 轮：没关系，释放出这些忧伤。这组小调和弦会承接你的情绪。我们离完成还有几步。"
    }
    
    return jsonify({
        'polarity': polarity,
        'subjectivity': subjectivity,
        'music_params': music_params,
        'ai_response': responses.get(mode, "我在听。"),
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
