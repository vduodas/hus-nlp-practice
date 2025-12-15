# Báo cáo Lab6 - Part2: Phân tích cú pháp phụ thuộc (Dependency Parsing)

## Giới thiệu

Trong **Lab 6 - Part 2: Phân tích cú pháp phụ thuộc (Dependency Parsing)**, em thực hiện thí nghiệm nhằm làm quen với kỹ thuật phân tích cú pháp phụ thuộc – một phương pháp quan trọng trong Xử lý Ngôn ngữ Tự nhiên (NLP) dùng để biểu diễn cấu trúc ngữ pháp của câu thông qua các mối quan hệ phụ thuộc giữa các từ.

Trọng tâm của bài lab bao gồm:

- Làm quen với khái niệm **head – dependent** trong cây cú pháp phụ thuộc.
- Sử dụng thư viện **spaCy** để phân tích cú pháp phụ thuộc cho câu tiếng Anh.
- Trực quan hóa cây phụ thuộc nhằm hiểu rõ cấu trúc ngữ pháp của câu.
- Duyệt cây phụ thuộc theo chương trình để trích xuất thông tin ngữ pháp có ý nghĩa.
- Áp dụng dependency parsing cho các bài toán thực tế như tìm động từ chính, cụm danh từ và đường đi trong cây cú pháp.

Mục tiêu của bài lab là giúp hiểu rõ mối quan hệ giữa cấu trúc cú pháp và ý nghĩa câu, từ đó tạo nền tảng cho các tác vụ NLP nâng cao như Information Extraction, Question Answering và Semantic Parsing.

- [File mã nguồn ở đây](../notebook/dependency_parsing.ipynb)
- Hướng dẫn chạy code:
    Sau khi kích hoạt môi trường ảo và cài đặt các thư viện cần thiết, thực hiện:

    ```bash
    Mở file notebook/dependency_parsing.ipynb → chọn kernel Python → Run All hoặc chạy từng cell.
    ```

    Toàn bộ kết quả huấn luyện và đánh giá được hiển thị trực tiếp trong notebook.

---

## 1. Kiến thức cơ bản: Phân tích cú pháp phụ thuộc

### 1.1. Khái niệm Dependency Parsing

Phân tích cú pháp phụ thuộc (Dependency Parsing) là phương pháp biểu diễn cấu trúc câu dưới dạng một cây, trong đó:

- Mỗi từ (token) là một nút trong cây.
- Mỗi từ (trừ ROOT) phụ thuộc vào đúng một từ khác gọi là **head**.
- Mối quan hệ giữa head và dependent được gán một **nhãn phụ thuộc** (dependency label) như `nsubj`, `dobj`, `amod`, `prep`, …

Token có vai trò trung tâm của câu được gán nhãn **ROOT**, thường là động từ chính.

---

### 1.2. Các thành phần chính trong spaCy

Khi phân tích cú pháp bằng spaCy, mỗi token trong đối tượng `Doc` cung cấp nhiều thuộc tính quan trọng:

- `token.text`: văn bản của token.
- `token.dep_`: nhãn quan hệ phụ thuộc.
- `token.head`: token head mà token hiện tại phụ thuộc vào.
- `token.children`: các token phụ thuộc trực tiếp vào token hiện tại.
- `token.pos_`: loại từ (Part-of-Speech).

Những thuộc tính này cho phép duyệt và khai thác cây cú pháp một cách linh hoạt bằng chương trình.

---

## 2. Phân tích cú pháp và trực quan hóa

### 2.1. Phân tích câu ví dụ

Câu ví dụ được sử dụng để phân tích cú pháp:

> *“The quick brown fox jumps over the lazy dog.”*

Sau khi đưa câu qua pipeline của spaCy, mô hình xác định:

- **jumps** là động từ chính và đóng vai trò **ROOT** của câu.
- **fox** là chủ ngữ (`nsubj`) của động từ *jumps*.
- Cụm giới từ *over the lazy dog* bổ nghĩa cho động từ.

#### Code

//```python
import spacy

nlp = spacy.load("en_core_web_md")
text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)
//```

---

### 2.2. Trực quan hóa cây phụ thuộc

Cây phụ thuộc được trực quan hóa bằng công cụ **displaCy** của spaCy, cho phép quan sát trực tiếp mối quan hệ giữa các token trong câu.

Việc trực quan hóa giúp:
- Nhận diện nhanh động từ chính.
- Hiểu rõ vai trò ngữ pháp của từng thành phần.
- Dễ dàng kiểm tra tính đúng đắn của kết quả parsing.

---

## 3. Truy cập và duyệt cây phụ thuộc theo chương trình

Để khai thác thông tin từ cây cú pháp, mỗi token được duyệt và in ra các thuộc tính quan trọng như nhãn phụ thuộc, head và các children.

#### Code

//```python
for token in doc:
    children = [child.text for child in token.children]
    print(token.text, token.dep_, token.head.text, children)
//```

Kết quả cho thấy cấu trúc cây phụ thuộc được spaCy xây dựng rõ ràng và nhất quán, phản ánh chính xác quan hệ ngữ pháp trong câu.

---

## 4. Trích xuất thông tin từ cây phụ thuộc

### 4.1. Tìm chủ ngữ và tân ngữ của động từ

Dựa vào các nhãn phụ thuộc `nsubj` (chủ ngữ) và `dobj` (tân ngữ), có thể trích xuất các bộ ba (chủ ngữ, động từ, tân ngữ) trong câu.

#### Code

//```python
if token.pos_ == "VERB":
    for child in token.children:
        if child.dep_ == "nsubj":
            subject = child.text
        if child.dep_ == "dobj":
            obj = child.text
//```

Kết quả cho thấy spaCy có khả năng xác định chính xác các thành phần ngữ pháp ngay cả trong câu có cấu trúc phức tạp.

---

## 5. Bài tập tự luyện

### 5.1. Bài 1: Tìm động từ chính của câu

Động từ chính của câu thường là token có nhãn `ROOT` và loại từ `VERB`.

**Kết quả:**
```text
Main verb: studying
```

Kết quả cho thấy spaCy xác định chính xác động từ trung tâm điều khiển toàn bộ cấu trúc câu.


### 5.2. Bài 2: Trích xuất các cụm danh từ (Noun Chunks)

Một cụm danh từ đơn giản bao gồm danh từ trung tâm và các thành phần bổ nghĩa như `det`, `amod`, `compound`.

**Kết quả:**
```text
Noun Chunks: ['The big fluffy cat', 'the wooden table']
```

Các cụm danh từ được trích xuất đầy đủ cả về hình thức và ngữ nghĩa.


### 5.3. Bài 3: Tìm đường đi từ token lên ROOT

Đường đi từ một token bất kỳ lên ROOT phản ánh vị trí của token đó trong cấu trúc cú pháp của câu.

**Kết quả:**

```text
Path to ROOT: ['mouse', 'chased']
```

Điều này cho thấy danh từ mouse phụ thuộc trực tiếp vào động từ chased, là động từ chính của câu.



## 6. Khó khăn và giải pháp

Trong quá trình thực hiện bài lab, một số khó khăn mang tính học thuật và phương pháp được ghi nhận:

- Việc phân biệt giữa các nhãn phụ thuộc có ý nghĩa gần nhau như `dobj`, `pobj`, `pcomp` ban đầu gây nhầm lẫn.
- Khi trích xuất thông tin theo chương trình, cần hiểu rõ hướng duyệt cây (từ head xuống children hoặc từ token lên head) để tránh bỏ sót thông tin quan trọng.
- Đối với các câu có cấu trúc phức hợp, việc chỉ dựa vào một cấp children đôi khi chưa đủ để trích xuất đầy đủ ngữ nghĩa.

Giải pháp được áp dụng là:

- Đối chiếu kết quả parsing với trực quan hóa để hiểu rõ vai trò của từng nhãn.
- Kết hợp nhiều thuộc tính của token (dep_, pos_, children, head) khi xây dựng thuật toán trích xuất.
- Ưu tiên các cấu trúc cú pháp đơn giản trước khi mở rộng sang các trường hợp phức tạp hơn.

## 7. Tài liệu tham khảo

- spaCy Documentation: [https://spacy.io/usage/linguistic-features#dependency-parse](https://spacy.io/usage/linguistic-features#dependency-parse)
- Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing.*
- Universal Dependencies: [https://universaldependencies.org](https://universaldependencies.org)