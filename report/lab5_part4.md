# Báo cáo Lab5 - Part4: Xây dựng mô hình RNN cho bài toán Nhận dạng Thực thể Tên (Named Entity Recognition)

## Giới thiệu

Trong bài lab này, em triển khai và đánh giá một hệ thống **Nhận dạng Thực thể Tên (Named Entity Recognition – NER)** cho văn bản tiếng Anh dựa trên **Mạng Nơ-ron Hồi quy (Recurrent Neural Network – RNN)**.  
NER là một bài toán quan trọng trong xử lý ngôn ngữ tự nhiên, nhằm xác định và phân loại các thực thể có ý nghĩa trong văn bản như **tên người (PER)**, **địa điểm (LOC)**, **tổ chức (ORG)**, v.v.

Dữ liệu được sử dụng trong bài lab là **CoNLL 2003**, một bộ dữ liệu chuẩn benchmark cho bài toán NER. Các nhãn được gán theo định dạng **IOB (Inside–Outside–Beginning)**.

Nội dung chính của bài lab bao gồm:

- Tải và tiền xử lý dữ liệu NER từ thư viện **Hugging Face datasets**.
- Chuyển đổi nhãn NER từ dạng số sang dạng chuỗi.
- Xây dựng từ điển (*vocabulary*) cho từ và nhãn NER.
- Triển khai lớp `Dataset` và `DataLoader` trong **PyTorch**, có xử lý *padding*.
- Xây dựng mô hình **RNN** cho bài toán *token classification*.
- Huấn luyện, đánh giá mô hình và phân tích kết quả.
- Xây dựng hàm dự đoán nhãn NER cho câu mới.

Mục tiêu của bài lab là hiểu rõ toàn bộ *pipeline* xây dựng một mô hình **sequence labeling** cho NER bằng PyTorch.

---

## 1. Các bước thực hiện

## 1.0. Hướng dẫn chạy code

Sau khi kích hoạt môi trường ảo và cài đặt các thư viện cần thiết, thực hiện:

```bash
Mở file notebook/rnn_ner.ipynb → chọn kernel Python → Run All hoặc chạy từng cell.
```

Toàn bộ kết quả huấn luyện và đánh giá được hiển thị trực tiếp trong notebook.

---

## 1.1. Tải và tiền xử lý dữ liệu (Task 1)

Bộ dữ liệu **CoNLL 2003** được tải thông qua thư viện `datasets` của Hugging Face.  
Do phiên bản mới của thư viện không còn hỗ trợ dataset dạng *loading script*, dữ liệu được tải từ nhánh **Parquet conversion**:

```python
load_dataset("conll2003", revision="refs/convert/parquet")
```

Dataset bao gồm ba tập:

- `train`
- `validation`
- `test`

Mỗi câu trong dữ liệu bao gồm:
- `tokens`: danh sách các token
- `ner_tags`: nhãn NER dạng số nguyên

Ánh xạ từ **ID → nhãn string** được trích xuất thông qua:

```python
dataset["train"].features["ner_tags"].feature.names
```

Ví dụ một câu sau khi tiền xử lý:

```text
Tokens: ['U.N.', 'official', 'Ekeus', 'heads', 'for', 'Baghdad', '.']
Tags:   ['B-ORG', 'O', 'B-PER', 'O', 'O', 'B-LOC', 'O']
```

---

## 1.2. Xây dựng Vocabulary (Task 1)

Từ tập huấn luyện, em xây dựng hai từ điển:

- **`word_to_ix`**: ánh xạ mỗi từ sang một chỉ số nguyên.  
  Thêm hai token đặc biệt:
  - `<PAD>`: dùng cho *padding*
  - `<UNK>`: dùng cho từ chưa xuất hiện trong tập train

- **`tag_to_ix`**: ánh xạ mỗi nhãn NER (dạng string) sang một chỉ số nguyên.

Kết quả:
- Vocabulary từ có kích thước lớn do đặc trưng dữ liệu báo chí.
- Số lượng nhãn NER phù hợp với chuẩn CoNLL 2003.

---

## 2. Tạo Dataset và DataLoader (Task 2)

### 2.1. Lớp NERDataset

Lớp `NERDataset` được xây dựng kế thừa từ `torch.utils.data.Dataset`, gồm các phương thức:

- `__init__`: nhận danh sách câu, nhãn và các từ điển.
- `__len__`: trả về số lượng câu.
- `__getitem__`: chuyển mỗi câu thành:
  - Tensor chỉ số từ
  - Tensor chỉ số nhãn NER

Độ dài câu được giữ nguyên ở bước này (chưa padding).

---

### 2.2. Collate function và DataLoader

Do các câu trong một batch có độ dài khác nhau, em triển khai hàm `collate_fn` để:

- *Padding* các câu trong batch về cùng độ dài (theo câu dài nhất).
- Sử dụng `pad_sequence` với `batch_first=True`.
- Giá trị padding cho nhãn được đặt là `-1`.

`DataLoader` được khởi tạo cho:
- Tập train: `shuffle=True`
- Tập validation: `shuffle=False`

---

## 3. Xây dựng mô hình RNN (Task 3)

Mô hình `SimpleRNNForNER` gồm ba thành phần chính:

1. **Embedding layer (`nn.Embedding`)**  
   Chuyển chỉ số từ thành vector dense. Token `<PAD>` được chỉ định bằng `padding_idx`.

2. **Recurrent layer (`nn.RNN`)**  
   Xử lý chuỗi embedding theo thứ tự thời gian, tạo hidden state cho mỗi token.

3. **Fully-connected layer (`nn.Linear`)**  
   Ánh xạ hidden state của mỗi token sang không gian nhãn NER.

Luồng xử lý:

```text
Input indices → Embedding → RNN → Linear → NER logits
```

Output của mô hình có dạng `(batch_size, seq_len, num_tags)`.

---

## 4. Huấn luyện mô hình (Task 4)

### 4.1. Cấu hình huấn luyện

- **Optimizer**: Adam  
- **Learning rate**: 0.001  
- **Loss function**: `nn.CrossEntropyLoss`  

Tham số `ignore_index` được thiết lập bằng giá trị padding của nhãn (`-1`) để bỏ qua các token padding khi tính loss.

---

### 4.2. Kết quả huấn luyện

Mô hình được huấn luyện trong **5 epoch**. Kết quả trên tập validation như sau:

| Epoch | Train Loss | Validation Accuracy |
|-------|------------|---------------------|
| 1     | 0.6412     | 0.8699              |
| 2     | 0.3720     | 0.8992              |
| 3     | 0.2572     | 0.9119              |
| 4     | 0.1859     | 0.9278              |
| 5     | 0.1381     | 0.9305              |

---

## 5. Đánh giá mô hình (Task 5)

### 5.1. Phương pháp đánh giá

- Mô hình được chuyển sang chế độ `eval()`.
- Tắt gradient bằng `torch.no_grad()`.
- Dự đoán nhãn bằng `argmax` trên chiều nhãn.
- Accuracy chỉ được tính trên các token **không phải padding**.

---

### 5.2. Dự đoán trên câu mới

Mô hình được kiểm tra nhanh trên câu:

```text
U.N. official Ekeus heads for Baghdad .
```

Kết quả dự đoán:

```text
U.N.            -> B-ORG
official        -> O
Ekeus           -> B-ORG
heads           -> O
for             -> O
Baghdad         -> B-LOC
.               -> O
```

Mô hình nhận diện đúng các thực thể **ORG** và **LOC**.  
Tên người *Ekeus* bị nhầm sang **ORG**, cho thấy hạn chế của RNN một chiều khi chưa khai thác đầy đủ ngữ cảnh hai phía.

---

## 6. Khó khăn và giải pháp

### 6.1. Khó khăn

- Phiên bản mới của thư viện **Hugging Face datasets** không còn hỗ trợ dataset dạng *loading script*, gây lỗi khi tải bộ dữ liệu CoNLL 2003 theo cách truyền thống.
- Các câu trong dữ liệu có độ dài khác nhau, gây khó khăn trong việc đưa vào mô hình RNN theo batch.
- Nhãn `O` chiếm đa số trong dữ liệu, khiến accuracy dễ bị cao nhưng chưa phản ánh đầy đủ chất lượng NER.
- Mô hình RNN một chiều còn hạn chế trong việc khai thác ngữ cảnh hai phía của câu.

### 6.2. Giải pháp

- Sử dụng phiên bản **Parquet conversion** của CoNLL 2003 (`refs/convert/parquet`) để đảm bảo tương thích với thư viện mới.
- Triển khai hàm `collate_fn` kết hợp với `pad_sequence` và `ignore_index` để xử lý padding hiệu quả.
- Đánh giá kết quả không chỉ dựa vào accuracy mà còn quan sát trực tiếp các dự đoán trên câu mới.
- Nhận diện rõ các hạn chế của mô hình để làm cơ sở cho các hướng cải tiến như **BiLSTM**, **CRF** hoặc **Transformer-based models**.

---

## 7. Kết luận

Trong bài lab này, em đã xây dựng thành công một hệ thống **NER hoàn chỉnh** dựa trên **RNN**, bao gồm các bước từ tải dữ liệu CoNLL 2003, tiền xử lý, xây dựng Dataset, mô hình hóa bằng PyTorch cho đến huấn luyện và đánh giá.

Kết quả thực nghiệm cho thấy mô hình đạt **~93% accuracy** trên tập validation, một kết quả tốt đối với mô hình RNN cơ bản. Tuy nhiên, mô hình vẫn còn hạn chế trong việc phân biệt một số loại thực thể phức tạp.

Trong tương lai, có thể cải thiện mô hình bằng cách:
- Sử dụng **BiLSTM / GRU**
- Kết hợp tầng **CRF**
- Áp dụng **embedding tiền huấn luyện** như GloVe hoặc FastText

---

## 8. Tài liệu tham khảo

- **Hugging Face Datasets**: https://huggingface.co/docs/datasets  
- **CoNLL 2003 Shared Task**: https://www.clips.uantwerpen.be/conll2003/ner/  
- **PyTorch – Sequence Models and RNNs**: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
