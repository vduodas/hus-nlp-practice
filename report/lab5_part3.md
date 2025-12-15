# Báo cáo Lab 5: Xây dựng mô hình RNN cho bài toán Part-of-Speech Tagging

## Giới thiệu

Trong bài lab này, em triển khai và đánh giá một hệ thống **gán nhãn từ loại (Part-of-Speech Tagging)** cho văn bản tiếng Anh dựa trên **Mạng Nơ-ron Hồi quy (Recurrent Neural Network – RNN)**. Bài toán *POS Tagging* là một nhiệm vụ cơ bản nhưng quan trọng trong xử lý ngôn ngữ tự nhiên, đóng vai trò tiền đề cho nhiều bài toán nâng cao như *parsing*, *named entity recognition* và *machine translation*.

Dữ liệu được sử dụng trong bài lab là **Universal Dependencies – English Web Treebank (UD English-EWT)**, ở định dạng **CoNLL-U** (xem mô tả tại [đây](../data/conllu_UD_English_EWT.md)). Nội dung bài lab bao gồm:

- Đọc và tiền xử lý dữ liệu CoNLL-U, trích xuất cặp `(word, UPOS)`.
- Xây dựng từ điển (*vocabulary*) cho từ và nhãn.
- Triển khai lớp `Dataset` và `DataLoader` trong **PyTorch**, có xử lý *padding* cho chuỗi độ dài biến đổi.
- Xây dựng mô hình **RNN** đơn giản cho bài toán phân loại nhãn theo token.
- Huấn luyện, đánh giá và phân tích kết quả mô hình trên tập *train* và *dev*.
- *(Nâng cao)* Xây dựng hàm dự đoán nhãn từ loại cho câu mới.

Mục tiêu của bài lab là nắm được toàn bộ *pipeline* xây dựng một mô hình *sequence labeling* bằng RNN trong PyTorch, từ tiền xử lý dữ liệu đến đánh giá mô hình.


---

## 1. Các bước thực hiện

## 1.0. Hướng dẫn chạy code

Sau khi clone source code và kích hoạt môi trường ảo, thực hiện các bước sau:

```bash
Mở file notebook/rnn_pos_tagging.ipynb → chọn kernel Python → bấm Run All hoặc chạy từng cell.
```

Kết quả được hiển thị dưới dạng các ô kết quả ngay bên dưới các ô chạy mã nguồn.

File mã nguồn có tại [đây](../notebook/rnn_pos_tagging.ipynb)

### 1.1. Tải và tiền xử lý dữ liệu (Task 1)

Dữ liệu được đọc từ các file:

- `en_ewt-ud-train.conllu`
- `en_ewt-ud-dev.conllu`

Mỗi file bao gồm nhiều câu, mỗi câu được phân tách bởi một dòng trống. Mỗi dòng trong câu chứa thông tin của một token, được phân cách bằng ký tự **tab**.

Trong bài lab này, em chỉ sử dụng hai cột quan trọng:

- **FORM (cột 2)**: từ gốc.
- **UPOS (cột 4)**: nhãn từ loại theo chuẩn *Universal POS*.

Hàm `load_conllu(file_path)` được triển khai để:

- Bỏ qua các dòng comment bắt đầu bằng `#`.
- Bỏ qua *multi-word token*.
- Trả về danh sách các câu, mỗi câu là danh sách các cặp `(word, upos)`.

Ví dụ định dạng dữ liệu sau khi đọc:
```text
[('From', 'ADP'), ('the', 'DET'), ('AP', 'PROPN'), ...]
```


---

### 1.2. Xây dựng Vocabulary (Task 1)

Từ tập dữ liệu huấn luyện, em xây dựng hai từ điển:

- `word_to_ix`: ánh xạ mỗi từ sang một chỉ số nguyên.  
  Thêm hai token đặc biệt:
  - `<PAD>`: dùng cho *padding*.
  - `<UNK>`: dùng cho từ chưa xuất hiện trong tập *train*.

- `tag_to_ix`: ánh xạ mỗi nhãn UPOS sang một chỉ số nguyên.  
  Thêm `<PAD>` để phục vụ *padding* nhãn.

Kết quả thu được:

- Vocabulary từ có kích thước tương đối lớn do đặc trưng dữ liệu *Web Treebank*.
- Số lượng nhãn UPOS phù hợp với chuẩn *Universal Dependencies*.

---

## 2. Tạo Dataset và DataLoader (Task 2)

### 2.1. Lớp POSDataset

Lớp `POSDataset` được xây dựng kế thừa từ `torch.utils.data.Dataset`, với các phương thức:

- `__init__`: nhận danh sách câu và hai từ điển.
- `__len__`: trả về số lượng câu.
- `__getitem__`: chuyển mỗi câu thành hai tensor:
  - Tensor chỉ số từ.
  - Tensor chỉ số nhãn.

Mỗi câu được giữ nguyên độ dài ban đầu (chưa *padding*).

---

### 2.2. Collate function và DataLoader

Do các câu trong một batch có độ dài khác nhau, em triển khai hàm `collate_fn` để:

- *Padding* các câu trong batch về cùng độ dài (theo câu dài nhất).
- Sử dụng `pad_sequence` với `batch_first=True`.

`DataLoader` được khởi tạo cho:

- Tập *train*: `shuffle=True`.
- Tập *dev*: `shuffle=False`.

---

## 3. Xây dựng mô hình RNN (Task 3)

Mô hình `SimpleRNNForTokenClassification` gồm ba thành phần chính:

1. **Embedding layer (`nn.Embedding`)**  
   Chuyển chỉ số từ thành vector *dense*. *Padding token* được chỉ định bằng `padding_idx`.

2. **Recurrent layer (`nn.RNN`)**  
   Xử lý chuỗi embedding theo thứ tự thời gian, sinh ra *hidden state* cho mỗi token.

3. **Fully-connected layer (`nn.Linear`)**  
   Ánh xạ *hidden state* của mỗi token sang không gian nhãn POS.

Luồng dữ liệu:
```text
Input indices → Embedding → RNN → Linear → POS logits
```

Output của mô hình có dạng ```(batch_size, seq_len, num_tags).```



---

## 4. Huấn luyện mô hình (Task 4)

### 4.1. Cấu hình huấn luyện

- **Optimizer**: Adam  
- **Learning rate**: 0.001  
- **Loss function**: `nn.CrossEntropyLoss`  

Thiết lập `ignore_index` cho nhãn `<PAD>` để không tính loss trên các token *padding*.

---

### 4.2. Kết quả huấn luyện

Mô hình được huấn luyện trong **5 epoch**. Kết quả thu được:

| Epoch | Train Loss | Train Accuracy | Dev Accuracy |
|-------|------------|----------------|--------------|
| 1     | 1.0603     | 0.7859         | 0.7679       |
| 2     | 0.5758     | 0.8498         | 0.8227       |
| 3     | 0.4321     | 0.8817         | 0.8508       |
| 4     | 0.3448     | 0.9076         | 0.8633       |
| 5     | 0.2843     | 0.9223         | 0.8744       |



---

## 5. Đánh giá mô hình (Task 5)

### 5.1. Phương pháp đánh giá

- Mô hình được chuyển sang chế độ `eval()`.
- Tắt gradient bằng `torch.no_grad()`.
- Dự đoán nhãn bằng `argmax` trên chiều nhãn.
- Accuracy chỉ được tính trên các token **không phải padding**.

---

### 5.2. Phân tích kết quả

- *Dev Accuracy* cuối cùng đạt khoảng **~87.44%**, cho thấy mô hình học được các phụ thuộc ngữ cảnh cơ bản.
- *Train accuracy* cao hơn *dev accuracy* khoảng **4–5%**, biểu hiện *overfitting* nhẹ nhưng vẫn trong mức chấp nhận được.
- Với một mô hình **RNN một chiều**, không *BiRNN*, không *CRF*, kết quả này được xem là tốt.

Các nhãn phổ biến như `DET`, `NOUN`, `VERB`, `ADP`, `PUNCT` thường được dự đoán chính xác.  
Một số nhãn dễ nhầm lẫn gồm:

- `NOUN` ↔ `PROPN`
- `AUX` ↔ `VERB`
- `ADJ` ↔ `NOUN`

---

## 6. Kết luận

Trong bài lab này, em đã xây dựng thành công một hệ thống **POS Tagging** hoàn chỉnh dựa trên **RNN**, bao gồm toàn bộ các bước từ tiền xử lý dữ liệu CoNLL-U, xây dựng Dataset, mô hình hóa bằng PyTorch cho đến huấn luyện và đánh giá.

Kết quả thực nghiệm cho thấy mô hình RNN cơ bản có khả năng học tốt cấu trúc tuần tự của ngôn ngữ và đạt độ chính xác tương đối cao trên tập dữ liệu thực tế. Tuy nhiên, vẫn còn nhiều hướng cải tiến tiềm năng như sử dụng **BiRNN**, **LSTM** hoặc thêm tầng **CRF** để mô hình hóa sự phụ thuộc giữa các nhãn.

---

## 7. Tài liệu tham khảo

- **PyTorch – Sequence Models and RNNs**: [https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- **Universal Dependencies**: [https://universaldependencies.org/](https://universaldependencies.org/)
- **CoNLL-U Format Specification**: [https://universaldependencies.org/format.html](https://universaldependencies.org/format.html)

