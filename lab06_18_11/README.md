Trong **Lab 6: Giới thiệu về Transformers**, em đã thực hiện các nội dung sau:
- Ôn tập kiến trúc Transformer cơ bản (Encoder, Decoder, Self-Attention).
- Sử dụng thư viện **Hugging Face `transformers`** để thực hiện các tác vụ NLP cơ bản với pretrained models.
- Triển khai **Masked Language Modeling** (Encoder-only) và **Next Token Prediction** (Decoder-only).
- Tính toán **Vector biểu diễn của câu** bằng phương pháp Mean Pooling.

---

## 1. Kiến thức cơ bản: Ôn tập về Transformers

### 1.1. Kiến trúc Transformer
Kiến trúc Transformer bao gồm hai phần chính:
- **Encoder**: Đọc và hiểu văn bản đầu vào để tạo ra các biểu diễn giàu ngữ cảnh.
- **Decoder**: Dựa vào biểu diễn của Encoder để sinh ra văn bản đầu ra.
- **Cơ chế cốt lõi**: **Self-Attention**, giúp mô hình cân nhắc tầm quan trọng của các từ khác nhau trong câu. 

### 1.2. Các loại mô hình Transformer
Dựa trên kiến thức từ tài liệu về `Transformers` thầy cung cấp, em có tóm tắt/tổng hợp lại 1 cách ngắn gọn như sau:
| Loại mô hình | Ví dụ | Đặc điểm | Tác vụ phù hợp |
| :--- | :--- | :--- | :--- |
| **Encoder-only** | BERT, ROBERTa | Nhìn **hai chiều** (bidirectional), hiểu ngữ cảnh sâu sắc. | Phân loại văn bản, NER, Trả lời câu hỏi, MLM. |
| **Decoder-only** | GPT, BLOOM | Nhìn **một chiều** (unidirectional), dự đoán từ tiếp theo. | Sinh văn bản, Next Token Prediction. |
| **Encoder-Decoder** | T5, BART | Kết hợp hai thành phần, phù hợp cho chuyển đổi chuỗi. | Dịch máy, Tóm tắt văn bản. |

---

## 2. Bài tập thực hành với Hugging Face Pipeline

### 2.1. Bài 1: Khôi phục Masked Token (Masked Language Modeling)

**Yêu cầu:** Sử dụng pipeline `fill-mask` để dự đoán từ bị thiếu trong câu:  
`Hanoi is the [MASK] of Vietnam.`  
**Mô hình sử dụng:** Encoder-only (ví dụ: BERT).


#### Code
```python
from transformers import pipeline

# 1. Tải pipeline "fill-mask"
mask_filler = pipeline("fill-mask")

# 2. Câu đầu vào
input_sentence = "Hanoi is the [MASK] of Vietnam."

# 3. Thực hiện dự đoán, lấy 5 dự đoán hàng đầu
predictions = mask_filler(input_sentence, top_k=5)

# 4. In kết quả
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
Dự đoán: 'center' với độ tin cậy: 0.0001
 -> Câu hoàn chỉnh: hanoi is the center of vietnam.
Dự đoán: 'birthplace' với độ tin cậy: 0.0001
 -> Câu hoàn chỉnh: hanoi is the birthplace of vietnam.
Dự đoán: 'headquarters' với độ tin cậy: 0.0001
 -> Câu hoàn chỉnh: hanoi is the headquarters of vietnam.
Dự đoán: 'city' với độ tin cậy: 0.0001
 -> Câu hoàn chỉnh: hanoi is the city of vietnam.
```

#### Em trả lời các câu hỏi trong bài:
    1. Mô hình đã dự đoán đúng từ capital không?
        - Trả lời: Có, độ tin cậy rất cao: 99.91%

    2. Tại sao các mô hình Encoder-only như BERT lại phù hợp cho tác vụ này?
        - Trả lời: Context \- là linh hồn của ngôn ngữ. Thông tin bối cảnh càng nhiều, dự đoán từ sẽ càng đúng hơn. Nói cách khác, tác vụ MLM này đòi hỏi khả năng nhìn hai chiều (bidirectional)\– tức là mô hình cần xem xét cả ngữ cảnh trước và sau từ bị mask để dự đoán từ gốc chính xác.


---

### 2.2. Bài 2: Dự đoán từ tiếp theo (Next Token Prediction)

**Yêu cầu:** Sử dụng pipeline `text-generation` để sinh ra phần tiếp theo cho câu:  
`The best thing about learning NLP is`  
**Mô hình sử dụng:** Thuộc họ Decoder-only (ví dụ: GPT).

#### Code
```python
from transformers import pipeline

# 1. Tải pipeline "text-generation" (tự động tải mô hình mặc định, thường là GPT-2)
generator = pipeline("text-generation")

# 2. Đoạn văn bản mồi (prompt)
prompt = "The best thing about learning NLP is"

# 3. Sinh văn bản (max_length=50, num_return_sequences=3)
# params:
#     max_length: tổng độ dài của câu mồi và phần được sinh ra
#     num_return_sequences: số lượng chuỗi kết quả muốn nhận
generated_texts = generator(prompt, max_length=50, num_return_sequences=1, truncation=True)

# 4. In kết quả
print(f"Câu mồi: '{prompt}'")
for text in generated_texts:
    print("Văn bản được sinh ra:")
    print(text['generated_text'])
```

#### Kết quả
```
Câu mồi: 'The best thing about learning NLP is'
Văn bản được sinh ra:
The best thing about learning NLP is that it's easy and fast.

NLP is a simple, easy to learn, and highly practical, language for all skill levels. There are no specific requirements, so learning NLP is easy.

This guide is for those who have already learned NLP in class. It covers the basic concepts of NLP, as well as the more advanced concepts such as declarative, imperative, and a bit more.

To learn NLP, you need to understand the underlying concepts of C++, and learn the concepts of imperative, declarative, and a bit more.

NLP is best learned by using NST and NLP-C++.

NLP-C++ is a great, high quality language for C++.

So what is NLP?

NLP is a programming language suitable for C++.

I've read a lot of NLP books and articles about NLP, and I think this is a really good starting point for learning NLP. NLP is a very simple language, and can be used for many different purposes.

This is a very good starting point for learning NLP.

I'm going to show you how to gain access to NLP
```

#### Em trả lời các câu hỏi trong bài:
    1. Kết quả sinh ra có hợp lý không?
        - Trả lời: Có, về mặt ngữ pháp và logic tổng thể thì hợp lý, dù nội dung hơi lạc chủ đề do mô hình không thật sự "hiểu" NLP.

    2. Tại sao mô hình Decoder-only như GPT phù hợp cho tác vụ này? 
        - Trả lời: Vì các mô hình Decoder-only được huấn luyện để dự đoán từ tiếp theo và chỉ có khả năng nhìn một chiều (unidirectional), tức là chỉ dựa vào các token đã xuất hiện trước đó để sinh ra chuỗi tiếp theo. Đây là cơ chế cốt lõi của việc sinh văn bản.

---
### 2.3. Bài 3: Tính toán Vector biểu diễn của câu (Sentence Representation)

**Yêu cầu:** Tính toán vector biểu diễn cho câu  
`This is a sample sentence.`  
bằng phương pháp **Mean Pooling**.  
**Mô hình sử dụng:** BERT (`bert-base-uncased`).

#### Về phương pháp Mean Pooling
- Lấy trung bình cộng của các vector đầu ra (`last_hidden_state`) của tất cả các token trong câu.  
- Loại trừ các token đệm (padding tokens) bằng cách sử dụng `attention_mask`.  

#### Code
```python
import torch
from transformers import AutoTokenizer, AutoModel

# 1. Chọn một mô hình BERT
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. Câu đầu vào
sentences = ["This is a sample sentence."]

# 3. Tokenize câu
# padding=True: đệm các câu ngắn hơn để có cùng độ dài
# truncation=True: cắt các câu dài hơn
# return_tensors='pt': trả về kết quả dưới dạng PyTorch tensors
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# 4. Đưa qua mô hình để lấy hidden states
# torch.no_grad() để không tính toán gradient, tiết kiệm bộ nhớ
with torch.no_grad():
    outputs = model(**inputs)

# outputs.last_hidden_state chứa vector đầu ra của tất cả các token
last_hidden_state = outputs.last_hidden_state
# shape: (batch_size, sequence_length, hidden_size)

# 5. Thực hiện Mean Pooling
# Để tính trung bình cộng của các vector, cần bỏ qua token đệm (padding tokens)
attention_mask = inputs['attention_mask']
mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

# Nhân các vector ẩn với mask và tính tổng (chỉ giữ lại vector của các token thực)
sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)

# Tính tổng của mask để chia trung bình (sử dụng min=1e-9 để tránh chia cho 0)
sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)

# Vector biểu diễn cuối cùng của câu
sentence_embedding = sum_embeddings / sum_mask

# 6. In kết quả
print("Vector biểu diễn của câu:")
print(sentence_embedding)
print("\nKích thước của vector:", sentence_embedding.shape)
```

#### Kết quả
```
Vector biểu diễn của câu:
tensor([[-6.3874e-02, -4.2837e-01, -6.6779e-02, -3.8430e-01, -6.5784e-02,
         -2.1826e-01,  4.7636e-01,  4.8659e-01,  4.0647e-05, -7.4273e-02,
         -7.4740e-02, -4.7635e-01, -1.9773e-01,  2.4824e-01, -1.2162e-01,
          1.6678e-01,  2.1045e-01, -1.4576e-01,  1.2636e-01,  1.8635e-02,
          2.4640e-01,  5.7090e-01, -4.7014e-01,  1.3782e-01,  7.3650e-01,
         -3.3808e-01, -5.0331e-02, -1.6452e-01, -4.3517e-01, -1.2900e-01,
          1.6516e-01,  3.4004e-01, -1.4930e-01,  2.2422e-02, -1.0488e-01,
         -5.1916e-01,  3.2964e-01, -2.2162e-01, -3.4206e-01,  1.1993e-01,
         -7.0148e-01, -2.3126e-01,  1.1224e-01,  1.2550e-01, -2.5191e-01,
         -4.6374e-01, -2.7261e-02, -2.8415e-01, -9.9249e-02, -3.7017e-02,
         -8.9192e-01,  2.5005e-01,  1.5816e-01,  2.2701e-01, -2.8497e-01,
          4.5300e-01,  5.0945e-03, -7.9441e-01, -3.1008e-01, -1.7403e-01,
          4.3029e-01,  1.6816e-01,  1.0590e-01, -4.8987e-01,  3.1856e-01,
          3.2861e-01, -1.3403e-02,  1.8807e-01, -1.0905e+00,  2.1009e-01,
         -6.7579e-01, -5.7076e-01,  8.5947e-02,  1.9121e-01, -3.3818e-01,
          2.7744e-01, -4.0539e-01,  3.1305e-01, -4.1197e-01, -5.6820e-01,
         -3.9074e-01,  4.0747e-01,  9.9898e-02,  2.3719e-01,  1.0154e-01,
         -2.5670e-01, -2.0583e-01,  1.1762e-01, -5.1439e-01,  4.0979e-01,
          1.2149e-01,  1.9333e-02, -5.9029e-02, -2.0141e-01,  7.0860e-01,
         -6.4609e-02,  2.4779e-02, -9.0578e-03,  1.9666e-02,  3.0815e-01,
         -4.9832e-02, -1.0691e+00,  6.1072e-01, -4.9722e-02, -1.5156e-01,
         -6.7778e-02,  4.7812e-02,  5.2103e-01,  1.6951e-01,  1.0146e-02,
          5.3093e-01, -7.8189e-02,  6.5843e-02, -2.9382e-01, -4.6045e-01,
          4.2071e-01,  1.1822e-01,  2.3631e-01, -4.5379e-02, -1.3740e-01,
...
         -3.9555e-02, -5.4193e-01, -4.4191e-01,  2.4927e-01,  6.6517e-01,
         -1.7534e-01, -1.2388e-01,  3.1970e-01]])

Kích thước của vector: torch.Size([1, 768])
```

#### Em trả lời các câu hỏi trong bài:

    1. Kích thước (chiều) của vector biểu diễn là bao nhiêu? Con số này tương ứng với tham số nào của mô hình BERT?
        - Trả lời: Kích thước của vector là (1, 768). Con số 768 tương ứng với tham số hidden_size (hay chiều ẩn) của mô hình BERT (bert-base-uncased).

    2. Tại sao chúng ta cần sử dụng attention_mask khi thực hiện Mean Pooling? 
        - Trả lời: Cần sử dụng attention_mask để loại bỏ các token đệm (padding tokens) khỏi phép tính trung bình. Điều này đảm bảo rằng chỉ các vector biểu diễn của các từ ngữ thực sự có ý nghĩa trong câu mới được tính vào trung bình, tránh làm sai lệch ngữ nghĩa của vector cuối cùng