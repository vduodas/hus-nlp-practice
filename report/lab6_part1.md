# Báo cáo Lab6 - Part1: Giới thiệu về Transformers

## Giới thiệu

Trong **Lab 6 - Part 1: Giới thiệu về Transformers**, em thực hiện thí nghiệm nhằm làm quen với kiến trúc Transformer và cách sử dụng các mô hình Transformer tiền huấn luyện cho các tác vụ NLP cơ bản. Trọng tâm của bài lab là:

- Ôn tập kiến trúc Transformer cơ bản (Encoder, Decoder, Self-Attention).
- Sử dụng thư viện **Hugging Face `transformers`** để triển khai nhanh các pipeline NLP phổ biến.
- Thực hành hai cơ chế huấn luyện quan trọng của Transformer:
  - **Masked Language Modeling (MLM)** với mô hình Encoder-only.
  - **Next Token Prediction** với mô hình Decoder-only.
- Tính toán **vector biểu diễn câu** bằng phương pháp **Mean Pooling** từ hidden states của BERT.

Mục tiêu của bài lab là hiểu rõ sự khác biệt giữa các loại mô hình Transformer và mối liên hệ giữa kiến trúc mô hình với từng loại tác vụ NLP cụ thể.

- [File mã nguồn ở đây](../notebook/transformers_introduction.ipynb)
- Hướng dẫn chạy code:
    Sau khi kích hoạt môi trường ảo và cài đặt các thư viện cần thiết, thực hiện:

    ```bash
    Mở file notebook/transformers_introduction.ipynb → chọn kernel Python → Run All hoặc chạy từng cell.
    ```

    Toàn bộ kết quả huấn luyện và đánh giá được hiển thị trực tiếp trong notebook.

---

## 1. Kiến thức cơ bản: Ôn tập về Transformers

### 1.1. Kiến trúc Transformer

Kiến trúc Transformer bao gồm hai thành phần chính:

- **Encoder**: tiếp nhận toàn bộ câu đầu vào và tạo ra các biểu diễn ngữ cảnh hai chiều cho từng token.
- **Decoder**: sinh văn bản đầu ra dựa trên các token đã xuất hiện trước đó (tự hồi quy).
- **Cơ chế cốt lõi – Self-Attention**: cho phép mô hình xác định mức độ quan trọng của các token khác nhau trong câu khi mã hóa ngữ nghĩa.

Nhờ cơ chế self-attention, Transformer có thể mô hình hóa các quan hệ phụ thuộc dài hạn mà không gặp hạn chế như RNN/LSTM truyền thống.

---

### 1.2. Các loại mô hình Transformer

Dựa trên kiến thức từ tài liệu về **Transformers** được cung cấp, các mô hình Transformer có thể được phân loại như sau:

| Loại mô hình | Ví dụ | Đặc điểm | Tác vụ phù hợp |
|-------------|-------|----------|----------------|
| **Encoder-only** | BERT, RoBERTa | Nhìn hai chiều (bidirectional), hiểu ngữ cảnh sâu | Phân loại, NER, QA, MLM |
| **Decoder-only** | GPT, BLOOM | Nhìn một chiều (unidirectional), dự đoán token tiếp theo | Sinh văn bản |
| **Encoder–Decoder** | T5, BART | Kết hợp Encoder và Decoder | Dịch máy, tóm tắt |

---

## 2. Thực hành với Hugging Face Pipeline

### 2.1. Bài 1: Masked Language Modeling (MLM)

**Yêu cầu:** Dự đoán từ bị che `[MASK]` trong câu:  
`Hanoi is the [MASK] of Vietnam.`

**Mô hình sử dụng:** Encoder-only (BERT)

#### Code
```python
from transformers import pipeline

mask_filler = pipeline("fill-mask")
input_sentence = "Hanoi is the [MASK] of Vietnam."
predictions = mask_filler(input_sentence, top_k=5)

print(f"Câu gốc: {input_sentence}")
for pred in predictions:
    print(f"Dự đoán: '{pred['token_str']}' với độ tin cậy: {pred['score']:.4f}")
    print(f" -> Câu hoàn chỉnh: {pred['sequence']}")
 ```

#### Kết quả

```
Câu gốc: Hanoi is the [MASK] of Vietnam.
Dự đoán: 'capital' với độ tin cậy: 0.9991
 -> Câu hoàn chỉnh: hanoi is the capital of vietnam.
...
 ```

#### Nhận xét

- Mô hình dự đoán chính xác từ **capital** với độ tin cậy rất cao.
- BERT phù hợp với tác vụ này vì được huấn luyện theo cơ chế **Masked Language Modeling** và có khả năng nhìn hai chiều, cho phép tận dụng cả ngữ cảnh trước và sau token bị che.

---

### 2.2. Bài 2: Next Token Prediction (Text Generation)

**Yêu cầu:** Sinh tiếp văn bản từ câu mồi:  
`The best thing about learning NLP is`

**Mô hình sử dụng:** Decoder-only (GPT-2)

#### Code
```python
from transformers import pipeline

generator = pipeline("text-generation")
prompt = "The best thing about learning NLP is"

generated_texts = generator(
    prompt,
    max_length=50,
    num_return_sequences=1,
    truncation=True
)

for text in generated_texts:
    print(text['generated_text'])
 ```

#### Nhận xét

- Văn bản sinh ra có cấu trúc ngữ pháp hợp lý và mạch lạc.
- Nội dung có phần lan man, phản ánh đặc điểm của mô hình decoder-only là **không có hiểu biết ngữ nghĩa thực sự**, mà chỉ dựa trên xác suất xuất hiện của token tiếp theo.

---

### 2.3. Bài 3: Vector biểu diễn câu (Sentence Representation)

**Yêu cầu:** Tính vector biểu diễn cho câu  
`This is a sample sentence.`  
bằng **Mean Pooling** từ BERT (`bert-base-uncased`).

#### Code
```python
import torch
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

sentences = ["This is a sample sentence."]
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_state = outputs.last_hidden_state
attention_mask = inputs['attention_mask']
mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

sentence_embedding = (
    torch.sum(last_hidden_state * mask_expanded, 1) /
    torch.clamp(mask_expanded.sum(1), min=1e-9)
)

print(sentence_embedding.shape)
 ```

#### Nhận xét

- Vector biểu diễn câu có kích thước **(1, 768)**, tương ứng với **hidden_size** của BERT-base.
- Việc sử dụng `attention_mask` là cần thiết để loại bỏ ảnh hưởng của các token đệm, giúp vector biểu diễn phản ánh chính xác ngữ nghĩa câu.

---

## 4. Tài liệu tham khảo

- Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
