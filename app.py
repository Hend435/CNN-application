from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEIGHTS_PATH = 'model/final_intel_model.weights.h5'
IMG_SIZE = (224, 224)
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def create_model():
    """Recreate the exact model architecture"""
    from tensorflow import keras
    from tensorflow.keras import layers
    
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights=None
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(6, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

print("🔄 Creating model architecture...")
model = create_model()

print("🔄 Loading weights...")
model.load_weights(WEIGHTS_PATH)

print("🔄 Compiling model...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("✅ Model loaded successfully!")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
async def root():
    return {
        "status": "running",
        "model_loaded": True,
        "classes": classes
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content_type = file.content_type or ""
        
    
        contents = await file.read()
        
        if len(contents) == 0:
            return {"success": False, "error": "Empty file"}
        
        processed_img = preprocess_image(contents)
        
        prediction = model.predict(processed_img, verbose=0)
        class_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        
        return {
            "success": True,
            "prediction": classes[class_idx],
            "confidence": f"{confidence * 100:.2f}%",
            "confidence_score": round(confidence * 100, 2)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

print("🔥 Warming up model...")
dummy_input = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
model.predict(dummy_input, verbose=0)
print("✅ Ready to serve predictions!")