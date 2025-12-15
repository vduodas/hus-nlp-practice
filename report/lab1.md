# Báo cáo Lab1: Tách từ văn bản bằng Tokenizer

## Giới thiệu

Trong bài lab này, em triển khai và thí nghiệm với các bộ **tokenizer** – công cụ cơ bản nhất trong xử lý ngôn ngữ tự nhiên (NLP). Nội dung bài lab bao gồm:

* Triển khai interface **`BaseTokenizer`** dưới dạng abstract class. ([file source](../src/core/interfaces.py))
* Xây dựng hai bộ tokenizer khác nhau:
  * **`SimpleTokenizer`**: dựa trên khoảng trắng và xử lý dấu câu đơn giản. ([file source](../src/lab01_16_09/preprocessing/simple_tokenizer.py))
  * **`RegexTokenizer`**: sử dụng biểu thức chính quy (regex) để tách từ chính xác hơn. ([file source](../src/lab01_16_09/preprocessing/regex_tokenizer.py))
* Thử nghiệm trên một số câu ví dụ và trên **bộ dữ liệu UD English-EWT** (xem mô tả tại [đây](../data/UD_English_EWT.md)).

Mục tiêu của bài lab là hiểu rõ vai trò của tokenizer – bước đầu tiên và quan trọng nhất trong pipeline xử lý ngôn ngữ tự nhiên.

---

## 1. Các bước thực hiện

### 1.1. Thiết kế Interface `BaseTokenizer`

Interface `BaseTokenizer` được định nghĩa trong file `src/core/interfaces.py` dưới dạng abstract class, bắt buộc tất cả các tokenizer phải cài đặt phương thức cơ bản:

```python
@abstractmethod
def tokenize(self, text: str) -> list[str]:
    pass
```

Cách tiếp cận này đảm bảo tính nhất quán – mọi tokenizer đều có giao diện `tokenize(text)` giống nhau, dù cơ chế nội bộ khác nhau.

---

### 1.2. Triển khai `SimpleTokenizer`

**Nguyên tắc hoạt động**:
* Chuyển văn bản về chữ thường.
* Tách token dựa trên khoảng trắng.
* Xử lý dấu câu riêng biệt – tách dấu câu khỏi từ và giữ lại dấu câu như token riêng.

**Ví dụ**:
```
Input: "Hello, world!"
Output: ["hello", ",", "world", "!"]
```

Cách tiếp cận này đơn giản nhưng hiệu quả cho các tập dữ liệu được chuẩn bị tốt.

---

### 1.3. Triển khai `RegexTokenizer`

**Nguyên tắc hoạt động**:
* Sử dụng **biểu thức chính quy** để định nghĩa chính xác các pattern của từ và dấu câu.
* Regex mẫu: `re.findall(r"\w+|[^\w\s]", text.lower())`

**Giải thích regex**:
* `\w+` → khớp một hoặc nhiều ký tự từ (chữ + số + underscore).
* `[^\w\s]` → khớp các ký tự không phải từ và không phải khoảng trắng (dấu câu, ký tự đặc biệt).
* `|` → toán tử "hoặc" – tìm khớp với một trong hai pattern.

**Ví dụ**:
```
Input: "NLP is fascinating... isn't it?"
Output: ["nlp", "is", "fascinating", "...", "isn", "'", "t", "it", "?"]
```

RegexTokenizer xử lý tốt hơn với các chuỗi phức tạp, nhất là khi gặp các từ viết tắt như "isn't".

---

### 1.4. Hàm `load_raw_text_data_from(path)`

Hàm này được triển khai trong `src/core/dataset_loaders.py` để:
* Đọc toàn bộ nội dung từ file văn bản (ví dụ: UD English-EWT).
* Trả về văn bản dưới dạng string.
* Hỗ trợ xử lý các file lớn bằng streaming (nếu cần).

---

## 2. Hướng dẫn chạy code

Sau khi clone source code và kích hoạt môi trường ảo, chạy lệnh sau tại thư mục gốc của project:

```bash
python -m test.lab1_test
```

Kết quả sẽ được in trực tiếp ra console, bao gồm:
* Test trên các câu ví dụ.
* Test trên 500 ký tự đầu của UD English-EWT.

---

## 3. Phân tích kết quả

### 3.1. Kết quả test trên câu ví dụ

Với các câu test:
```
"Hello, world! This is a test."
"NLP is fascinating... isn't it?"
"Let's see how it handles 123 numbers and punctuation!"
```

**SimpleTokenizer** kết quả:
```
["hello", ",", "world", "!", "this", "is", "a", "test", "."]
["nlp", "is", "fascinating", ".", ".", ".", "isn", "'", "t", "it", "?"]
["let", "'", "s", "see", "how", "it", "handles", "123", "numbers", "and", "punctuation", "!"]
```

**RegexTokenizer** kết quả:
```
["hello", ",", "world", "!", "this", "is", "a", "test", "."]
["nlp", "is", "fascinating", ".", ".", ".", "isn", "'", "t", "it", "?"]
["let", "'", "s", "see", "how", "it", "handles", "123", "numbers", "and", "punctuation", "!"]
```

Trong trường hợp test này, cả hai tokenizer cho kết quả giống nhau vì các câu test không chứa các trường hợp cạnh (edge case).

### 3.2. Kết quả test trên UD English-EWT

Khi test trên 500 ký tự đầu của UD English-EWT:
```
Sample text: Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the mosque in the town of ...

SimpleTokenizer (first 20 tokens):
['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']

RegexTokenizer (first 20 tokens):
['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']
```

### 3.3. So sánh hai tokenizer

| Tiêu chí | SimpleTokenizer | RegexTokenizer |
| :--- | :--- | :--- |
| Độ phức tạp | Thấp | Cao hơn |
| Tốc độ | Nhanh | Hơi chậm hơn |
| Xử lý trường hợp cạnh | Kém | Tốt hơn |
| Phù hợp với dữ liệu sạch | Tốt | Tốt |
| Phù hợp với dữ liệu bẩn | Kém | Tốt hơn |

### 3.4. Những khó khăn gặp phải

* **Regex phức tạp**: Để xử lý tốt các trường hợp edgecases, regex cần được thiết kế cẩn thận. Tham khảo [RegEx - Python docs](https://docs.python.org/3/howto/regex.html) để học sâu hơn.
* **Encoding**: Khi làm việc với dữ liệu đa ngôn ngữ, cần chú ý đến encoding (UTF-8, etc.).
* **Dấu câu lồng nhau**: Các chuỗi như "..." có thể tạo ra nhiều token, cần quyết định cách xử lý phù hợp.

---

## 4. Kết luận

Qua bài lab này, em đã:
* Hiểu được vai trò của tokenizer trong NLP.
* Triển khai hai cách tiếp cận khác nhau (đơn giản vs regex).
* Nhận ra rằng không có giải pháp tokenizer hoàn hảo – lựa chọn phụ thuộc vào bối cảnh dữ liệu và bài toán cụ thể.

Tokenizer là bước tiền xử lý rất quan trọng. Chất lượng tokenization trực tiếp ảnh hưởng đến hiệu năng của các bước xử lý tiếp theo (như vector hóa, phân loại). 

## 5. Tài liệu tham khảo
- [RegEx - Python docs](https://docs.python.org/3/howto/regex.html)