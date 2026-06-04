"""
Healthcare AI Chatbot - Professional Web Interface
Complete Flask Application with Modern UI
"""
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from document_retriever import DocumentRetriever, RetrievalConfig
import json
import uuid
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'
CORS(app)
retriever = None
conversations = {}
def init_retriever():
    global retriever
    try:
        config = RetrievalConfig(
            similarity_threshold=0.12,
            context_size=2,
            enable_caching=True,
            enable_keyword_boosting=True,
            enable_multiple_answers=True,
            max_answers=3
        )
        retriever = DocumentRetriever('healthcare_data.json', config)
        logger.info("Retriever initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        return False
@app.route('/')
def index():
    """Main chat interface"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    session_id = session['session_id']
    if session_id not in conversations:
        conversations[session_id] = []
    return render_template('index.html')
@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        session_id = session.get('session_id')
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        if not retriever:
            return jsonify({'error': 'System not ready'}), 503
        logger.info(f"Query from {session_id}: {user_message}")
        result, score, topic = retriever.find_best_answer(user_message)
        if result:
            if hasattr(result, 'answer'):
                answer_text = result.answer
                source_topic = result.topic if hasattr(result, 'topic') else topic
            else:
                answer_text = result.get('answer', '')
                source_topic = result.get('topic', topic)
            confidence = get_confidence_level(score)
            suggestions = get_suggestions(user_message)
            response_data = {
                'answer': answer_text,
                'topic': source_topic,
                'confidence': confidence,
                'score': round(score * 100, 1),
                'suggestions': suggestions,
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Fallback response
            fallback = retriever.get_fallback_response(user_message)
            response_data = {
                'answer': fallback,
                'topic': 'General',
                'confidence': 'Low',
                'score': 0,
                'suggestions': get_suggestions(user_message, fallback_mode=True),
                'timestamp': datetime.now().isoformat()
            }
        if session_id in conversations:
            conversations[session_id].append({
                'user': user_message,
                'bot': response_data,
                'timestamp': datetime.now().isoformat()
            })
            # Keep only last 50 messages
            if len(conversations[session_id]) > 50:
                conversations[session_id] = conversations[session_id][-50:]
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': str(e)}), 500
@app.route('/api/suggestions', methods=['GET'])
def get_suggestion_topics():
    """Get suggested topics for users"""
    suggestions = [
        "What are the symptoms of diabetes?",
        "How to prevent heart disease?",
        "What causes high blood pressure?",
        "Treatment options for asthma",
        "COVID-19 prevention tips",
        "Migraine relief methods",
        "Benefits of regular exercise",
        "Healthy diet recommendations"
    ]
    return jsonify({'suggestions': suggestions})
@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history"""
    session_id = session.get('session_id')
    if session_id in conversations:
        conversations[session_id] = []
    return jsonify({'status': 'success'})
@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    session_id = session.get('session_id')
    if session_id in conversations:
        return jsonify({'history': conversations[session_id]})
    return jsonify({'history': []})
def get_confidence_level(score):
    """Convert score to confidence level"""
    if score >= 0.6:
        return "Very High"
    elif score >= 0.4:
        return "High"
    elif score >= 0.25:
        return "Medium"
    elif score >= 0.15:
        return "Low"
    else:
        return "Very Low"
def get_suggestions(query, fallback_mode=False):
    """Generate contextual suggestions"""
    suggestions = []

    if "diabetes" in query.lower():
        suggestions = [
            "What are diabetes complications?",
            "How to manage blood sugar?",
            "Diabetes diet recommendations"
        ]
    elif "heart" in query.lower() or "cardio" in query.lower():
        suggestions = [
            "Heart disease risk factors",
            "Best exercises for heart health",
            "Foods for heart health"
        ]
    elif "blood pressure" in query.lower() or "hypertension" in query.lower():
        suggestions = [
            "Natural ways to lower blood pressure",
            "Hypertension medications",
            "Diet for high blood pressure"
        ]
    elif "asthma" in query.lower():
        suggestions = [
            "Asthma triggers to avoid",
            "Asthma inhaler types",
            "Emergency asthma treatment"
        ]
    elif fallback_mode:
        suggestions = [
            "What are the symptoms of diabetes?",
            "How to lower blood pressure naturally?",
            "Best treatment for asthma",
            "Heart disease prevention tips",
            "COVID-19 safety measures"
        ]
    return suggestions[:3]
if __name__ == '__main__':
    if init_retriever():
        print("\n" + "=" * 60)
        print("🚀 Healthcare AI Chatbot is running!")
        print("📍 Open your browser and go to: http://localhost:5000")
        print("=" * 60 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    else:
        print("Failed to initialize. Please check your healthcare_data.json file.")