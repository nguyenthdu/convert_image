from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
from paddleocr import PaddleOCR  # Sử dụng PaddleOCR cho nhận dạng
from transformers import pipeline  # Sử dụng pipeline dịch từ HuggingFace

app = Flask(__name__)
CORS(app)

# --- Caching cho PaddleOCR ---
ocr_cache = {}

def get_ocr_reader(lang):
    """
    Lấy (hoặc khởi tạo nếu chưa có) đối tượng PaddleOCR cho ngôn ngữ được yêu cầu.
    Giá trị lang nhận từ frontend (ví dụ: "ch", "en", "vie").
    """
    if lang not in ocr_cache:
        ocr_cache[lang] = PaddleOCR(use_gpu=False, lang=lang)
    return ocr_cache[lang]

# --- Caching cho pipeline dịch ---
translation_cache = {}

def get_translation_pipeline(src, tgt):
    """
    Lấy (hoặc khởi tạo nếu chưa có) pipeline dịch với cặp ngôn ngữ src-tgt.
    Ví dụ: src='zh', tgt='en' → model "Helsinki-NLP/opus-mt-zh-en"
    """
    key = f"{src}-{tgt}"
    if key not in translation_cache:
        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        translation_cache[key] = pipeline("translation", model=model_name)
    return translation_cache[key]

# Bản đồ ánh xạ mã ngôn ngữ từ frontend sang mã dùng cho mô hình dịch
translation_lang_map = {
    "ch": "zh",   # Tiếng Trung
    "en": "en",   # Tiếng Anh
    "vie": "vi"   # Tiếng Việt
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/ocr', methods=['POST'])
def ocr():
    # Kiểm tra file ảnh được gửi lên
    if 'image' not in request.files:
        return jsonify({'error': 'Không tìm thấy file ảnh trong yêu cầu.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Tên file ảnh không hợp lệ.'}), 400

    # Lấy tham số ngôn ngữ dùng cho OCR (nguồn)
    lang = request.form.get('language', 'ch')
    # Lấy tham số ngôn ngữ đích (để dịch), nếu không có hoặc nếu trùng với ngôn ngữ nguồn thì không dịch
    target_language = request.form.get('target_language', None)

    file_bytes = file.read()
    npimg = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'File ảnh không hợp lệ hoặc không thể giải mã.'}), 400

    try:
        # Thực hiện OCR với PaddleOCR
        ocr_reader = get_ocr_reader(lang)
        results = ocr_reader.ocr(img, cls=True)  # cls=True nếu cần xử lý hướng văn bản

        recognized_text = ""
        # Kết quả của PaddleOCR là danh sách các list (mỗi list chứa các mục: [bounding_box, (text, confidence)])
        for line in results:
            for res in line:
                if res is not None and len(res) > 1:
                    recognized_text += res[1][0] + "\n"
        recognized_text = recognized_text.strip()

        # Nếu không lấy được nội dung hoặc không yêu cầu dịch (hoặc target trùng với nguồn) → trả về text gốc
        if recognized_text == "" or not target_language or target_language == lang:
            return jsonify({'text': recognized_text, 'translation': ""}), 200

        # Ánh xạ mã ngôn ngữ cho dịch
        src = translation_lang_map.get(lang, "en")
        tgt = translation_lang_map.get(target_language, "en")
        translation_pipe = get_translation_pipeline(src, tgt)
        # Pipeline dịch nhận đầu vào là chuỗi (hoặc danh sách chuỗi)
        translation_output = translation_pipe(recognized_text, max_length=512)
        # Tổng hợp kết quả dịch từ pipeline
        translated_text = " ".join([item['translation_text'] for item in translation_output])

        return jsonify({'text': recognized_text, 'translation': translated_text}), 200

    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý OCR/Translation: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
