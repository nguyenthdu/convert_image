# Sử dụng image Python nền
FROM python:3.10-slim-buster

# Cập nhật danh sách gói và cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy toàn bộ mã nguồn vào thư mục /app
COPY . /app

# Chuyển đến thư mục làm việc trong container
WORKDIR /app

# Cài đặt các gói phụ thuộc từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Khai báo cổng mà ứng dụng sẽ lắng nghe
EXPOSE 5000

# Khởi chạy ứng dụng khi container được khởi động
CMD ["python", "app.py"]
