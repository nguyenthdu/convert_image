from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
from paddleocr import PaddleOCR  # Sử dụng PaddleOCR cho nhận dạng OCR
from transformers import pipeline  # Sử dụng pipeline dịch từ HuggingFace

app = Flask(__name__)
CORS(app)

# --- Caching đối tượng PaddleOCR ---
ocr_cache = {}

def get_ocr_reader(lang):
    """
    Lấy (hoặc khởi tạo nếu chưa có) đối tượng PaddleOCR cho ngôn ngữ được yêu cầu.
    Ví dụ: lang nhận từ frontend có thể là "ch", "en", "vie".
    """
    if lang not in ocr_cache:
        ocr_cache[lang] = PaddleOCR(use_gpu=False, lang=lang)
    return ocr_cache[lang]

# --- Caching cho pipeline dịch ---
translation_cache = {}

def get_translation_pipeline(src, tgt):
    """
    Lấy (hoặc khởi tạo nếu chưa có) pipeline dịch với cặp ngôn ngữ src-tgt.
    Ví dụ: nếu src='zh' và tgt='en' → sử dụng model "Helsinki-NLP/opus-mt-zh-en"
    """
    key = f"{src}-{tgt}"
    if key not in translation_cache:
        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        translation_cache[key] = pipeline("translation", model=model_name)
    return translation_cache[key]

# Bản đồ ánh xạ mã ngôn ngữ từ frontend (cho OCR) sang mã dùng cho mô hình dịch
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
    print("Ngôn ngữ lấy nội dung trong hình ảnh:", lang)
    # Lấy tham số ngôn ngữ đích (để dịch) nếu có; nếu không có hoặc nếu trùng với ngôn ngữ nguồn thì không dịch
    target_language = request.form.get('target_language', None)
    print("Ngôn ngữ dịch:", target_language)
    file_bytes = file.read()
    npimg = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'File ảnh không hợp lệ hoặc không thể giải mã.'}), 400

    try:
        # Thực hiện OCR với PaddleOCR
        ocr_reader = get_ocr_reader(lang)
        results = ocr_reader.ocr(img, cls=True)  # cls=True bật nhận diện hướng văn bản (nếu cần)

        recognized_text = ""
        # Kết quả của PaddleOCR là danh sách các list, mỗi list chứa các mục: [bounding_box, (text, confidence)]
        for line in results:
            for res in line:
                if res is not None and len(res) > 1:
                    recognized_text += res[1][0] + "\n"
        recognized_text = recognized_text.strip()

        # Nếu không lấy được nội dung hoặc không yêu cầu dịch (hoặc target trùng với ngôn ngữ nguồn) → trả về kết quả OCR
        if recognized_text == "" or not target_language or target_language == lang:
            print("================Không dịch:", recognized_text)
            return jsonify({'text': recognized_text, 'translation': ""}), 200


        print("Bắt đầu dịch nội dung OCR...")
        # Ánh xạ mã ngôn ngữ cho dịch (ví dụ: "ch" → "zh")
        src = translation_lang_map.get(lang, "en")
        tgt = translation_lang_map.get(target_language, "en")
        try:
            print("Dịch từ", src, "sang", tgt)
            translation_pipe = get_translation_pipeline(src, tgt)
            # Pipeline dịch nhận đầu vào là chuỗi (hoặc danh sách chuỗi)
            print("Chuỗi cần dịch:", recognized_text)
            translation_output = translation_pipe(recognized_text, max_length=512)
            print("Kết quả dịch:", translation_output)
            # Tổng hợp kết quả dịch (mỗi phần tử là một dict chứa key 'translation_text')
            translated_text = " ".join([item['translation_text'] for item in translation_output])
            print("================Translation:", translated_text)
        except Exception as te:
            # Nếu có lỗi trong quá trình dịch, in log và để nội dung dịch rỗng
            print("Translation error:", te)
            translated_text = ""
        return jsonify({'text': recognized_text, 'translation': translated_text}), 200

    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý OCR/Translation: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
