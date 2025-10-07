"""Embedding extraction"""
from insightface.app import FaceAnalysis

def init_insightface():
    """Initialize InsightFace with optimized settings for speed"""
    try:
        print("📦 Loading InsightFace models (optimized for speed)...")
        
        app = FaceAnalysis(
            allowed_modules=['detection', 'recognition'],
            providers=['CPUExecutionProvider']
        )
        
        app.prepare(ctx_id=-1, det_size=(160, 160))
        print("✅ InsightFace ready!")
        return app
        
    except Exception as e:
        print(f"❌ InsightFace initialization failed: {e}")
        print("💡 Try installing a lighter face recognition model")
        raise