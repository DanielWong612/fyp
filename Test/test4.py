import tensorflow as tf
import numpy as np
import cv2

# 加載模型
model = tf.keras.models.load_model("/Users/sam/Documents/GitHub/fyp/22067588d_additional/new_finalcustomedVGG16_model.h5")
model.load_weights("/Users/sam/Documents/GitHub/fyp/22067588d_additional/new_finalVGG16_weight.weights.h5")

mapper = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happiness',
    4: 'sadness',
    5: 'surprise',
    6: 'neutral'
}

# 定義圖片預處理函數
def preprocess_image(image_path, target_size):
    """
    預處理輸入圖片：加載圖片、調整大小、歸一化
    """
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} not found!")
    
    # 調整圖片大小
    image = cv2.resize(image, target_size)
    # 轉換為RGB (如需要)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 歸一化至 [0, 1]
    image = image / 255.0
    # 增加批次維度 (形狀為 [1, height, width, channels])
    image = np.expand_dims(image, axis=0)
    
    return image

# 定義分類函數
def predict_image(image_path, model, target_size=(64, 64)):
    """
    使用模型對圖片進行分類
    """
    # 預處理圖片
    processed_image = preprocess_image(image_path, target_size)
    
    # 預測
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)  # 獲取預測類別
    confidence = np.max(predictions)         # 獲取預測概率
    
    return predicted_class, confidence

# 測試分類功能
if __name__ == "__main__":
    # 設置圖片路徑
    image_path = "/Users/sam/Documents/GitHub/fyp/Test/image.jpeg"  # 替換為你自己的圖片路徑
    
    # 設置目標輸入大小（根據模型的 input_shape）
    target_size = (64, 64)  # 替換為你的模型輸入大小
    
    # 分類圖片
    try:
        predicted_class, confidence = predict_image(image_path, model, target_size)
        predicted_emotion = mapper[predicted_class]
        print(f"Predicted Class: {predicted_emotion}, Confidence: {confidence:.2f}")
    except FileNotFoundError as e:
        print(e)
