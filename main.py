"""Main entry point"""

import os
import sys

# Get the directory where main.py is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Change working directory to the script directory
os.chdir(SCRIPT_DIR)
print(f"Working directory set to: {os.getcwd()}")

from config.config import dataset_config
from src.core.embedding_service import init_insightface
from src.services.dataset_loader import load_dataset
from src.services.index_manager import build_index, get_top_matches
from src.services.verification_service import verify_face
from src.services.camera_service import close_camera
from src.services.live_verification import capture_and_verify, live_verification_mode
from src.services.storage_service import save_model, load_model

def main():
    """Enhanced main function with camera options"""
    print("🚀 ENHANCED FACE VERIFICATION SYSTEM WITH CAMERA")
    print("="*60)
    
    app = init_insightface()
    embeddings = []
    labels = []
    index = None
    threshold = 0.6
    camera = None
    dataset_path = dataset_config.dataset_path
    
    # Check for model in FaceRecognition/models folder
    model_path = "models/enhanced_face_model.pkl"
    if os.path.exists(model_path):
        choice = input("📁 Found existing model. Load it? (y/n): ").lower().strip()
        if choice == 'y':
            embeddings, labels, threshold, index, success = load_model(model_path)
            if success:
                print("✅ Model loaded successfully!")
            else:
                print("⚠️ Failed to load model, will create new one")
                embeddings = []
                labels = []
                index = None
    
    if len(embeddings) == 0:
        print("\n📚 Building new model from dataset...")
        embeddings, labels, success = load_dataset(app, dataset_path)
        if success:
            index = build_index(embeddings)
            if index:
                save_model(embeddings, labels, threshold, index)
                print("✅ Model built and saved!")
            else:
                print("❌ Failed to build index")
                return
        else:
            print("❌ Failed to load dataset")
            return
    
    while True:
        print(f"\n{'='*60}")
        print("🎯 ENHANCED MENU:")
        print("1. 🔍 Verify single image file")
        print("2. 📊 Get top 5 matches")
        print("3. 📷 Capture & Verify (Camera)")
        print("4. 🎥 Live Verification Mode")
        print("5. ⚙️ Change threshold")
        print("6. 📈 System info")
        print("7. 🚪 Exit")
        print("="*60)
        
        choice = input("Choose option (1-7): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip().strip('"')
            if os.path.exists(image_path):
                verify_face(app, index, labels, image_path, threshold)
            else:
                print(f"❌ File not found: {image_path}")
        
        elif choice == '2':
            image_path = input("Enter image path: ").strip().strip('"')
            if os.path.exists(image_path):
                matches = get_top_matches(index, labels, image_path, app)
                print("\n🔍 Top matches:")
                for i, (label, score) in enumerate(matches, 1):
                    print(f"  {i}. {label}: {score:.3f}")
            else:
                print(f"❌ File not found: {image_path}")
        
        elif choice == '3':
            print("\n📷 Camera Capture & Verify Mode")
            try:
                label, score, camera = capture_and_verify(app, camera, index, labels, threshold)
                if label:
                    print(f"\n🎉 VERIFICATION RESULT: {label} (confidence: {score:.3f})")
                else:
                    print(f"\n❌ VERIFICATION FAILED: Unknown person")
            except Exception as e:
                print(f"❌ Camera error: {e}")
        
        elif choice == '4':
            print("\n🎥 Live Verification Mode")
            try:
                camera = live_verification_mode(app, camera, index, labels, threshold)
            except Exception as e:
                print(f"❌ Live mode error: {e}")
            finally:
                close_camera(camera)
                camera = None
        
        elif choice == '5':
            current = threshold
            print(f"Current threshold: {current:.3f}")
            try:
                new_threshold = float(input("Enter new threshold (0.0-1.0): "))
                if 0.0 <= new_threshold <= 1.0:
                    threshold = new_threshold
                    print(f"✅ Threshold updated to {new_threshold:.3f}")
                else:
                    print("❌ Threshold must be between 0.0 and 1.0")
            except ValueError:
                print("❌ Invalid number")
        
        elif choice == '6':
            print(f"\n📊 SYSTEM INFO:")
            print(f"Embeddings: {len(embeddings)}")
            print(f"Classes: {list(set(labels))}")
            print(f"Threshold: {threshold:.3f}")
            print(f"Index size: {index.ntotal if index else 0}")
            print(f"Camera status: {'Ready' if camera else 'Not initialized'}")
        
        elif choice == '7':
            print("👋 Goodbye!")
            close_camera(camera)
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()