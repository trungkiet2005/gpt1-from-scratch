---
title: Vietnamese Poetry Generator
emoji: 🎭
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
license: mit
---

# Vietnamese Poetry Generator 🎭

Một AI generator tạo thơ tiếng Việt sử dụng mô hình GPT được train từ đầu.

## Tính năng

- **Generate thơ từ text prompt**: Nhập một đoạn text và AI sẽ tạo ra bài thơ tiếp theo
- **Điều chỉnh độ sáng tạo**: Sử dụng temperature để control tính sáng tạo
- **Vocabulary size 194**: Model được tối ưu cho tiếng Việt lowercase
- **FastAPI backend**: RESTful API với documentation tự động

## API Endpoints

### `GET /`
Thông tin cơ bản về API

### `GET /health`
Kiểm tra trạng thái API

### `POST /generate`
Tạo thơ cơ bản
```json
{
  "prompt": "tình yêu đẹp như",
  "max_length": 300,
}
```

### `POST /generate_creative`
Tạo thơ với điều khiển temperature
```json
{
  "prompt": "thiên nhiên xanh tươi",
  "max_length": 300,
}
```

## Cách sử dụng

1. **Qua Web Interface**: Truy cập giao diện web để tương tác trực tiếp
2. **Qua API**: Gửi POST request đến các endpoints trên
3. **Programmatically**: Sử dụng trong code của bạn

## Ví dụ Response

```json
{
  "generated_text": "tình yêu đẹp như hoa nở trong tim...",
  "original_prompt": "tình yêu đẹp như",
}
```

## Model Architecture

- **Transformer-based GPT**: 12 layers, 12 attention heads
- **Embedding dimension**: 768
- **Context length**: 512 tokens

## Lưu ý

- Model sẽ tự động convert input text thành lowercase
- Chỉ hỗ trợ các ký tự có trong vocabulary training
- Độ dài prompt nên dưới 200 ký tự để tối ưu

## Technical Stack

- **Backend**: FastAPI + PyTorch
- **Model**: Custom GPT implementation
- **Deployment**: Docker + Hugging Face Spaces
- **Frontend**: Vanilla HTML/CSS/JavaScript

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py

# Build Docker
docker build -t poetry-generator .
docker run -p 7860:7860 poetry-generator
```

---

Made with ❤️ for Vietnamese poetry generation 
