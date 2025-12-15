# Báo cáo Lab2: Mã hóa văn bản bằng vector tần suất từ

## Giới thiệu

Trong bài lab này, em triển khai và thí nghiệm với **CountVectorizer** – công cụ chuyển đổi văn bản thành vector tần suất từ (bag-of-words representation). Nội dung bài lab bao gồm:

* Triển khai interface **`Vectorizer`** dưới dạng abstract class. ([file source](../src/core/interfaces.py))
* Xây dựng lớp **`CountVectorizer`** để chuyển đổi tập hợp tài liệu thành Document-Term Matrix. ([file source](../src/representations/count_vectorizer.py))
* Thử nghiệm trên một số câu ví dụ và trên **bộ dữ liệu UD English-EWT** (xem mô tả tại [đây](../data/UD_English_EWT.md)).

Mục tiêu của bài lab là hiểu rõ cách chuyển đổi dữ liệu văn bản dạng string thành biểu diễn số học dạng vector, từ đó có thể sử dụng trong các mô hình học máy.

---

## 1. Các bước thực hiện

### 1.1. Thiết kế Interface `Vectorizer`

Interface `Vectorizer` được định nghĩa trong file `src/core/interfaces.py` dưới dạng abstract class, đặc định ba phương thức cốt lõi mà mọi vectorizer phải cài đặt:

```python
@abstractmethod
def fit(self, corpus: list[str]):
    pass    

@abstractmethod
def transform(self, documents: list[str]) -> list[list[int]]:
    pass

@abstractmethod
def fit_transform(self, corpus: list[str]) -> list[list[int]]:
    pass    
```

Cách tiếp cận này cho phép:
* **`fit()`**: Học vocabulary từ corpus.
* **`transform()`**: Chuyển đổi tài liệu thành vector.
* **`fit_transform()`**: Kết hợp hai bước trên.

---

### 1.2. Triển khai `CountVectorizer`

CountVectorizer là bước chuyển đổi dữ liệu văn bản thành biểu diễn số học. Quy trình gồm ba phần chính:

#### 1.2.1. Khởi tạo (`__init__`)
* Nhận vào một **tokenizer** (đã được implement ở lab01, xem tại [báo cáo lab1](./lab1.md)).
* Mặc định dùng `SimpleTokenizer` nếu không được cung cấp.
* Khởi tạo cấu trúc từ vựng `_vocabulary` để lưu ánh xạ **token → index**.

#### 1.2.2. Học từ vựng (`fit`)
* **Tokenize toàn bộ corpus**: Mỗi tài liệu được tách thành danh sách token.
* **Gộp tất cả tokens**: Tạo tập hợp duy nhất từ tất cả token trong corpus (loại bỏ trùng lặp).
* **Sắp xếp token**: Sắp xếp theo thứ tự alphabet để đảm bảo tính nhất quán.
* **Gán index**: Gán một index duy nhất cho từng token trong vocabulary.

#### 1.2.3. Biến đổi văn bản (`transform`)
* Với mỗi document:
  * Khởi tạo vector có kích thước bằng số lượng token trong vocabulary.
  * Tokenize document và đếm số lần xuất hiện của từng token.
  * Ghi số đếm vào vị trí tương ứng trong vector.
* Trả về Document-Term Matrix (DTM): ma trận shape `(n_documents, n_vocabulary)`.

#### 1.2.4. Kết hợp (`fit_transform`)
* Gọi `fit()` để học vocabulary từ corpus.
* Gọi `transform()` để sinh ma trận document-term từ cùng corpus đó.
* Trả về DTM.

---

## 2. Hướng dẫn chạy code

Sau khi clone source code và kích hoạt môi trường ảo, chạy lệnh sau tại thư mục gốc của project:

```bash
python -m test.lab2_test
```

Chương trình sẽ tự động:
* Test trên các câu ví dụ.
* Test trên ba đoạn văn bản từ UD English-EWT (mỗi đoạn 500 ký tự).
* In ra vocabulary learned và Document-Term Matrix.

---

## 3. Phân tích kết quả

### 3.1. Kết quả test trên câu ví dụ

**Corpus test**:
```
corpus = [
    "I love NLP.",
    "I love programming.",
    "NLP is a subfield of AI."
]
```

**Vocabulary học được** (sau khi regex tokenizer xử lý):
```
{'.': 0, 'a': 1, 'ai': 2, 'i': 3, 'is': 4, 'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9}
```

**Document-Term Matrix** (sau fit_transform):
```
[
 [1, 0, 0, 1, 0, 1, 1, 0, 0, 0],   # "I love NLP."
 [1, 0, 0, 1, 0, 1, 0, 0, 1, 0],   # "I love programming."
 [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]    # "NLP is a subfield of AI."
]
```

**Giải thích**:
* Mỗi hàng biểu diễn một tài liệu (document).
* Mỗi cột tương ứng với một token trong vocabulary (theo thứ tự alphabet).
* Giá trị là số lần token xuất hiện trong document đó.
* Ví dụ: Document 1 ("I love NLP.") chứa token '.' 1 lần, 'i' 1 lần, 'love' 1 lần, 'nlp' 1 lần, và các token khác là 0 lần.

### 3.2. Kết quả test trên UD English-EWT

Khi test trên ba đoạn văn bản 500 ký tự từ UD English-EWT:

**Sample documents** (mỗi document 500 ký tự):
```
[Document 1] Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the mosque in the town of Qaim...

[Document 2] igh level members of the Weathermen bombers back in the 1960s. The third was being run by the head of an investment firm...

[Document 3] h of the ARVN officers who were secretly working for the other side in Vietnam. Al-Zaman : Guerrillas killed a member...
```

**Kích thước Vocabulary learned**: 180 token duy nhất (bao gồm dấu câu, từ thường, từ viết hoa, số, v.v...)

**Document-Term Matrix shape**: `(3, 180)` – ma trận 3 tài liệu × 180 token

**Quan sát**:
* Một số token xuất hiện trong nhiều document (ví dụ: "the", "of", "and", "a", "in", ".").
* Một số token chỉ xuất hiện trong một document (ví dụ: "weathermen", "arvn", "guerrillas").
* Tần suất từ cao nhất là token "the" với 8 lần xuất hiện trong Document 1, 3 lần trong Document 2, và 4 lần trong Document 3.

### 3.3. Ưu điểm và hạn chế của CountVectorizer

| Khía cạnh | Chi tiết |
| :--- | :--- |
| **Ưu điểm** | |
| Đơn giản, dễ hiểu | Dễ triển khai và giải thích kết quả. |
| Nhanh chóng | Thích hợp cho dữ liệu lớn. |
| Không mất thông tin tần suất từ | Giữ nguyên thông tin về số lần xuất hiện. |
| **Hạn chế** | |
| Bỏ qua thứ tự từ | Câu "tôi yêu NLP" và "NLP yêu tôi" có vector giống nhau. |
| Độ thưa cao | Hầu hết các phần tử trong ma trận là 0. |
| Từ dừng ảnh hưởng lớn | Từ thường như "the", "a" chiếm ưu thế trong vector. |
| Không xét cấu trúc ngữ pháp | Chỉ xét tần suất, không xét mối quan hệ ngữ pháp. |

### 3.4. Cải tiến có thể

* **TF-IDF (Term Frequency-Inverse Document Frequency)**: Áp dụng trọng số để giảm tầm quan trọng của từ thường.
* **N-gram**: Thay vì chỉ dùng từ (unigram), sử dụng chuỗi từ (bigram, trigram) để giữ một phần thông tin về thứ tự.
* **Lọc từ dừng (Stopword removal)**: Loại bỏ các từ thường như "the", "a", "is" trước khi vector hóa.

---

## 4. Kết luận

Qua bài lab này, em đã:
* Hiểu được cách chuyển đổi văn bản thành biểu diễn vector (bag-of-words).
* Triển khai interface `Vectorizer` và lớp `CountVectorizer` đầy đủ.
* Nhận ra rằng CountVectorizer là bước cơ bản nhưng quan trọng trong xử lý ngôn ngữ tự nhiên.

Với CountVectorizer, dữ liệu văn bản bây giờ có thể được sử dụng trực tiếp trong các mô hình học máy (như Logistic Regression, Naive Bayes, SVM) để giải quyết các bài toán như phân loại văn bản, phân tích cảm xúc, v.v...

## Tài liệu tham khảo
- Scikit-learn – Text Feature Extraction: [https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

- Jurafsky & Martin – Speech and Language Processing (Bag-of-Words & Vector Space Models): [https://web.stanford.edu/~jurafsky/slp3/6.pdf](https://web.stanford.edu/~jurafsky/slp3/6.pdf)
