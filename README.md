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

## Cài đặt và Thiết lập

### Requirements
```bash
# Cài đặt dependencies
pip install -r requirements.txt
```

### Dependencies chính:
- `fastapi==0.104.1` - Web framework
- `uvicorn==0.24.0` - ASGI server
- `torch==2.1.1` - Deep learning framework
- `numpy==1.24.3` - Numerical computing
- `pydantic==2.5.2` - Data validation

## Cách chạy Training Mode

### 1. Chuẩn bị Dataset
Đảm bảo có file `dataset.txt` chứa text tiếng Việt để train:


### 2. Chạy Training
```bash
# Chạy training script
python model.py
```

### 3. Model Checkpoints
Model sẽ được lưu tự động:
- `final_model.pt` - Model cuối cùng
- `model_checkpoint.pt` - Checkpoint định kỳ
- `encoded_data.pt` - Data đã encode để tăng tốc

### 4. Monitor Training
Training progress sẽ hiển thị:
- Train loss và validation loss
- Estimated time remaining
- Generated samples định kỳ

## Cách chạy API Mode

### 1. Chạy Local Development Server
```bash
# Cách 1: Chạy trực tiếp
python app.py

# Cách 2: Sử dụng uvicorn
uvicorn app:app --host 0.0.0.0 --port 7860 --reload

# Cách 3: Sử dụng start script
bash start.sh
```

### 2. Chạy với Docker
```bash
# Build Docker image
docker build -t poetry-generator .

# Chạy container
docker run -p 7860:7860 poetry-generator
```

### 3. Truy cập API
- **API Documentation**: http://localhost:7860/docs
- **Alternative docs**: http://localhost:7860/redoc
- **Health check**: http://localhost:7860/health

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
  "temperature": 1.0
}
```

### `POST /generate_creative`
Tạo thơ với điều khiển temperature
```json
{
  "prompt": "thiên nhiên xanh tươi",
  "max_length": 300,
  "temperature": 1.2
}
```

## Testing API

### Sử dụng curl:
```bash
# Test health endpoint
curl http://localhost:7860/health

# Test generation
curl -X POST "http://localhost:7860/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "mùa thu lá vàng", "max_length": 200, "temperature": 1.0}'
```

### Sử dụng test script:
```bash
python test_api.py
```

## Ví dụ Response

```json
{
  "generated_text": "tình yêu đẹp như hoa nở trong tim...",
  "original_prompt": "tình yêu đẹp như",
  "vocab_size": 194
}
```

## Model Architecture

- **Transformer-based GPT**: 12 layers, 12 attention heads
- **Embedding dimension**: 512
- **Context length**: 256 tokens
- **Vocabulary**: 194 unique characters (Vietnamese lowercase)
- **Parameters**: ~87M parameters

## Development

### Local Development
```bash
# Clone repository
git clone <repository-url>
cd GPT-1

# Install dependencies
pip install -r requirements.txt

# Chạy training (nếu cần train lại)
python model.py

# Chạy API server
python app.py
```

### Testing
```bash
# Install test dependencies
pip install -r test_requirements.txt

# Run API tests
python test_api.py
```

### Deployment

#### Docker Deployment
```bash
docker build -t poetry-generator .
docker run -p 7860:7860 poetry-generator
```

#### Hugging Face Spaces
Project đã được setup để deploy lên Hugging Face Spaces với Docker SDK.

## File Structure

```
GPT-1/
├── app.py                 # FastAPI application
├── model.py              # Training script và model architecture
├── train.py              # Training utilities
├── dataset.txt           # Training dataset
├── final_model.pt        # Trained model
├── requirements.txt      # Dependencies
├── Dockerfile           # Container configuration
├── start.sh             # Startup script
├── test_api.py          # API testing
└── static/              # Static files
```

## Lưu ý

### Training:
- Cần GPU để train hiệu quả (khuyến nghị ít nhất 8GB VRAM)
- Training có thể mất vài giờ đến vài ngày tùy dataset size
- Model sẽ tự động save checkpoint để resume training


## Technical Stack

- **Backend**: FastAPI + PyTorch
- **Model**: Custom GPT implementation
- **Deployment**: Docker + Hugging Face Spaces
- **Frontend**: Vanilla HTML/CSS/JavaScript (in static/)

---

