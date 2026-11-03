"""
Inference script for Fruit Image Classifier
Loads trained CNN model checkpoint (.pth) and classifies any input fruit image.
"""

import os
import sys
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

import torch
from PIL import Image
from torchvision import transforms
from cnn_fruit_classifier import FruitCNN

def predict_single_image(image_path: str, model, class_names, device):
    """Loads an image, transforms it, and returns prediction probabilities."""
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    best_idx = probs.argmax()
    predicted_fruit = class_names[best_idx]
    confidence = probs[best_idx]

    prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    return predicted_fruit, confidence, prob_dict


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "fruit_cnn_model.pth")

    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}. Please run cnn_fruit_classifier.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = FruitCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("=" * 65)
    print("         FRUIT IMAGE CLASSIFIER - INFERENCE DEMO               ")
    print("=" * 65)
    print(f"Loaded FruitCNN with classes: {class_names}")

    # If user provided a specific image path
    if len(sys.argv) > 1:
        target_img = sys.argv[1]
        if os.path.exists(target_img):
            fruit, conf, dist = predict_single_image(target_img, model, class_names, device)
            print(f"\nTarget Image: {target_img}")
            print(f"--> Predicted Fruit: [{fruit.upper()}] (Confidence: {conf*100:.2f}%)")
            print("Class Probabilities:")
            for cls, p in dist.items():
                bar = "█" * int(p * 30)
                print(f"  - {cls:<10}: {p*100:5.1f}% | {bar}")
            return
        else:
            print(f"File not found: {target_img}")

    # Otherwise test on one image from each test directory
    print("\n[Running automated test inference on test set samples]")
    for fruit in class_names:
        fruit_test_folder = os.path.join(current_dir, "dataset", "test", fruit)
        if os.path.exists(fruit_test_folder):
            files = [f for f in os.listdir(fruit_test_folder) if f.endswith(".png")]
            if files:
                sample_file = os.path.join(fruit_test_folder, files[0])
                pred_fruit, conf, dist = predict_single_image(sample_file, model, class_names, device)
                match = "[CORRECT]" if pred_fruit.lower() == fruit.lower() else "[MISMATCH]"
                print(f"\nGround Truth: {fruit.capitalize()} ({files[0]})")
                print(f"--> Prediction: [{pred_fruit.upper()}] {match} (Confidence: {conf*100:.1f}%)")
                for cls, p in dist.items():
                    print(f"      {cls:<8}: {p*100:.1f}%")

if __name__ == "__main__":
    main()
