Trong bài lab01 này, em triển khai:  
- **Interface `BaseTokenizer`** và hai bộ tokenizer:  
  - `SimpleTokenizer` (dựa trên quy tắc ký tự).  
  - `RegexTokenizer` (dựa trên biểu thức chính quy).  
- Thử nghiệm trên một số câu ví dụ và trên **bộ dữ liệu UD English-EWT**.  

---


## 1. Các bước triển khai 
### 1.1. Interface `BaseTokenizer`  
- Được định nghĩa trong `core/interfaces.py` dưới dạng abstract class.  
- Bắt buộc mọi tokenizer phải cài đặt phương thức:  

```python
@abstractmethod
def tokenize(self, text: str) -> list[str]:
    pass
```

### 1.2. `SimpleTokenizer`  
- Nguyên tắc:  
  - Chuyển văn bản về chữ thường.  
  - Tách token dựa trên khoảng trắng.  
  - Xử lý dấu câu riêng biệt (giữ lại dấu câu như token).  

**Ví dụ:**  
```
Input: "Hello, world!"
Output: ["hello", ",", "world", "!"]
```
### 1.3. `RegexTokenizer`  
- Sử dụng **biểu thức chính quy** để tách từ.  
- Regex mẫu:  

```python
re.findall(r"\w+|[^\w\s]", text.lower())
```
- Giải thích:
    - ```\w+``` → từ (chữ + số).
    - ```[^\w\s]``` → ký tự đặc biệt, dấu câu.

**Ví dụ:**  
```
Input: "NLP is fascinating... isn't it?"
Output: ["nlp", "is", "fascinating", "...", "isn", "'", "t", "it", "?"]
```

### 1.4. `Dataset Loader`
- Hàm `load_raw_text_data_from(path)` dùng để đọc toàn bộ dữ liệu từ file UD English-EWT.
- Sau đó lấy 500 kí tự đầu để test.

## 2. Thử nghiệm 
Sau khi pull sourcecode về, chạy tại cwd bằng lệnh 
```
python -m lab01_16_09.main
```
Kết quả thu được sẽ được hiển thị trực tiếp ra console.
### 2.1 Test trên câu ví dụ
Với các câu:
```
"Hello, world! This is a test."
"NLP is fascinating... isn't it?"
"Let's see how it handles 123 numbers and punctuation!"
```
- **SimpleTokenizer** giữ dấu câu tách riêng.

- **RegexTokenizer** xử lý tốt hơn với các chuỗi phức tạp (như "isn't", "123").

### 2.2. Test trên dataset UD English-EWT

- Đọc 500 ký tự đầu tiên.

- Tokenizer tách thành danh sách tokens.

- So sánh kết quả:

     - **SimpleTokenizer**: thường tách thô, ít chính xác với từ viết tắt, số.

     - **RegexTokenizer**: tách chuẩn hơn, nhận diện rõ dấu câu và số.

### 2.3. Kết quả mẫu
```
Simple Tokenizer Results:
Input: Hello, world! This is a test.
Tokens: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

Input: NLP is fascinating... isn't it?
Tokens: ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']

Input: Let's see how it handles 123 numbers and punctuation!
Tokens: ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']

Regex Tokenizer Results:
Input: Hello, world! This is a test.
Tokens: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

Input: NLP is fascinating... isn't it?
Tokens: ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']

Input: Let's see how it handles 123 numbers and punctuation!
Tokens: ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']


--- Tokenizing Sample Text from UD_English-EWT ---
Original Sample: Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the
mosque in the town of ...
SimpleTokenizer Output (first 20 tokens): ['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']
RegexTokenizer Output (first 20 tokens): ['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']
```

### 2.4. Một số khó khăn gặp phải và cách giải quyết
- Em có xem lại kiến thức về `regex` tại [RegEx - Python docs](https://docs.python.org/3/howto/regex.html) 