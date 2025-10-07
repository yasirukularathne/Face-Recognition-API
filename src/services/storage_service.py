"""Model storage"""
import os
import pickle
import faiss
from datetime import datetime

def save_model(embeddings, labels, threshold, index, model_path=None):
    """Save the model inside FaceRecognition/models folder"""
    # Default paths inside FaceRecognition structure
    if model_path is None:
        model_path = "models/enhanced_face_model.pkl"
    
    # Ensure models directory exists
    models_dir = os.path.dirname(model_path)
    if models_dir and not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        print(f"📁 Created directory: {models_dir}")
    
    model_data = {
        'embeddings': embeddings,
        'labels': labels,
        'threshold': threshold,
        'created_at': datetime.now().isoformat()
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    # Save FAISS index in models folder
    if index:
        index_path = "models/enhanced_face_index.faiss"
        faiss.write_index(index, index_path)
        print(f"💾 Index saved: {index_path}")
    
    print(f"💾 Model saved: {model_path}")

def load_model(model_path=None):
    """Load the model from FaceRecognition/models folder"""
    # Default paths inside FaceRecognition structure
    if model_path is None:
        model_path = "models/enhanced_face_model.pkl"
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        embeddings = model_data['embeddings']
        labels = model_data['labels']
        threshold = model_data['threshold']
        
        # Load FAISS index from models folder
        index = None
        index_path = "models/enhanced_face_index.faiss"
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            print(f"📁 Index loaded: {index_path}")
        
        print(f"📁 Model loaded: {len(labels)} embeddings from {model_path}")
        return embeddings, labels, threshold, index, True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None, None, None, False