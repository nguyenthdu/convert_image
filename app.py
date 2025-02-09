from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
from paddleocr import PaddleOCR  # Sử dụng PaddleOCR cho nhận dạng OCR
from googletrans import Translator  # Import googletrans

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
            return jsonify({'text': recognized_text, 'translation': ""}), 200


        print("Bắt đầu dịch nội dung OCR...")
        # Không cần ánh xạ ngôn ngữ nữa, googletrans dùng mã tương tự
        src_lang = lang
        target_lang = target_language
        try:
            print("Dịch từ", src_lang, "sang", target_lang)
            translator = Translator() # Khởi tạo đối tượng Translator của googletrans
            print("Chuỗi cần dịch:", recognized_text)
            translation = translator.translate(recognized_text , dest=target_lang) # Dịch bằng googletrans # ĐÃ SỬA LỖI: chỉ truyền 'recognized_text' một lần
            translated_text = translation.text # Lấy nội dung đã dịch
            print("Kết quả dịch: ", translated_text)
        except Exception as te:
            # Nếu lỗi dịch, vẫn trả về kết quả OCR và thông báo lỗi
            print("Translation error:", te)
            translated_text = "" # Nội dung dịch để trống khi có lỗi
        return jsonify({'text': recognized_text, 'translation': translated_text}), 200

    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý OCR/Translation: {str(e)}'}), 500
#test web truy cập tra ve message
@app.route('/test')
def test():
    return "Chào mừng bạn đến với ứng dụng OCR!"
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)