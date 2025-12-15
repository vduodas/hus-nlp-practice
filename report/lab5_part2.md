# Báo cáo Lab5 - Part2: Phân loại Văn bản với Mạng Nơ-ron Hồi quy (RNN/LSTM)

## Giới thiệu

Trong bài lab này, em thực hiện thí nghiệm với bài toán **`phân loại ý định (intent classification)`**, sử dụng nhiều cách tiếp cận khác nhau từ mô hình truyền thống đến mô hình học sâu. Trọng tâm của bài lab là khảo sát vai trò của biểu diễn văn bản (text representation) và khả năng mô hình hóa chuỗi đối với hiệu năng phân loại.
Bốn pipeline mô hình được xây dựng và đánh giá bao gồm:

- **`TF-IDF + Logistic Regression`** (baseline truyền thống).
- **`Word2Vec (trung bình vector câu) + Dense Layer.`**
- **`Embedding Pre-trained + LSTM.`**
- **`Embedding học từ đầu (Scratch) + LSTM.`**

Mục tiêu của bài lab là:

- So sánh hiệu năng định lượng giữa các pipeline.
- Phân tích định tính khả năng hiểu câu phức tạp, đặc biệt là các câu có phủ định hoặc cấu trúc dài, của các mô hình dựa trên chuỗi (LSTM).
- Làm rõ ưu và nhược điểm của từng cách tiếp cận trong bối cảnh dữ liệu thực tế có nhiều lớp và phân bố không cân bằng.

---

## 1. Các bước thực hiện

### 1.1. Chuẩn bị dữ liệu

Tập dữ liệu được sử dụng cho bài toán phân loại ý định được mô tả tại [đây](../data/hwu.md)

Dữ liệu được tiền xử lý thống nhất cho các pipeline, bao gồm:

- Chuẩn hóa văn bản (lowercase).
- Tokenization theo từ.
- Với các mô hình chuỗi (LSTM), dữ liệu được chuyển sang chuỗi chỉ số và padding về cùng độ dài.

### 1.2. Pipeline 1: TF-IDF + Logistic Regression

Đây là mô hình baseline nhằm cung cấp một mốc so sánh ban đầu.

- Văn bản được biểu diễn bằng **`TF-IDF`**, với số lượng đặc trưng giới hạn **`(max_features = 5000)`**.  
- Bộ phân loại sử dụng Logistic Regression đa lớp.  
- Mô hình không xét đến thứ tự từ trong câu, mỗi câu được xem như một vector đặc trưng rời rạc.  

**Ưu điểm**: đơn giản, dễ triển khai, huấn luyện nhanh, thường cho kết quả khá tốt với các câu ngắn và cấu trúc đơn giản.

### 1.3. Pipeline 2: Word2Vec (Trung bình) + Dense Layer

Ở pipeline này, embedding được đưa vào nhưng chưa có khả năng mô hình hóa chuỗi.

- Word2Vec được huấn luyện từ tập dữ liệu train.  
- Mỗi câu được biểu diễn bằng **`vector trung bình`** của các từ trong câu.  
- Vector câu được đưa vào một mạng nơ-ron đơn giản gồm:
    - Dense layer với hàm kích hoạt ReLU.
    - Dropout để giảm overfitting.
    - Softmax output cho 66 lớp ý định.

**Ưu điểm**: giảm sparsity, bắt được một phần ngữ nghĩa.  
**Nhược điểm**: mất hoàn toàn thứ tự từ trong câu.

### 1.4. Pipeline 3: Embedding Pre-trained + LSTM

Đây là mô hình học sâu đầu tiên có khả năng xử lý chuỗi.

- Word2Vec huấn luyện ở pipeline 2 được dùng để khởi tạo trọng số cho Embedding layer.  
- Embedding layer được đóng băng (trainable = False) để giữ nguyên tri thức đã học.
- LSTM được sử dụng để học quan hệ phụ thuộc theo thời gian giữa các từ.
- Đầu ra là một Dense layer với Softmax cho bài toán phân loại.

Mục tiêu của pipeline này là kiểm tra:
- Việc kết hợp embedding có sẵn với mô hình chuỗi có giúp cải thiện khả năng hiểu câu phức tạp hay không.

### 1.5. Pipeline 4: Embedding học từ đầu + LSTM

- Kiến trúc gần giống pipeline 3 nhưng embedding được học từ đầu.  
- Trọng số embedding cập nhật trực tiếp theo loss của bài toán phân loại ý định.

**Ưu điểm**: embedding tối ưu trực tiếp cho intent classification.  
**Nhược điểm**: nguy cơ overfitting cao nếu dữ liệu không đủ lớn.

---

## 2. Hướng dẫn chạy code

Sau khi clone source code và kích hoạt môi trường ảo, thực hiện các bước sau:

```bash
Mở file notebook/rnn_text_classification.ipynb → chọn kernel Python → bấm Run All hoặc chạy từng cell.
```

Kết quả được hiển thị dưới dạng các ô kết quả ngay bên dưới các ô chạy mã nguồn.

## 3. Phân tích kết quả

### 3.1. So sánh định lượng

Sau khi huấn luyện bốn pipeline khác nhau, em tiến hành đánh giá hiệu năng của các mô hình trên tập kiểm tra (test set). Hai chỉ số được sử dụng là **macro F1-score** và **test loss**.

Việc sử dụng macro F1-score là đặc biệt quan trọng trong bài toán phân loại ý định với nhiều lớp (66 intent), bởi vì chỉ số này tính trung bình F1-score của từng lớp, không bị chi phối bởi các lớp có số lượng mẫu lớn, từ đó phản ánh tốt hơn hiệu năng của mô hình trên các lớp thiểu số.

**Bảng 1. So sánh kết quả định lượng trên tập test**

| Pipeline                        | F1-score (Macro) | Test Loss |
|---------------------------------|------------------|-----------|
| TF-IDF + Logistic Regression    | 0.84             | N/A       |
| Word2Vec (Avg) + Dense          | ~0.18            | ~3.20     |
| Embedding (Pre-trained) + LSTM  | ~0.10            | ~3.38     |
| Embedding (Scratch) + LSTM      | ~0.02            | ~4.13     |

**Nhận xét:**

- TF-IDF + Logistic Regression đạt kết quả vượt trội nhất với macro F1-score khoảng 0.84, cho thấy khả năng phân biệt tốt trên hầu hết các intent, kể cả các lớp có ít mẫu.  
- Các mô hình dựa trên embedding và mạng nơ-ron sâu cho kết quả thấp hơn đáng kể. Nguyên nhân chủ yếu đến từ quy mô dữ liệu huấn luyện còn hạn chế, chưa đủ để học các biểu diễn ngữ nghĩa phức tạp.  
- Đặc biệt, mô hình Embedding học từ đầu + LSTM cho kết quả gần với đoán ngẫu nhiên (loss xấp xỉ log(66)), thể hiện hiện tượng underfitting nghiêm trọng.  

---

### 3.2. Phân tích định tính

Để đánh giá khả năng hiểu ngữ nghĩa và xử lý cấu trúc câu phức tạp của các mô hình, đặc biệt là LSTM, em tiến hành kiểm tra trên một số câu “khó” trong tập test. Các câu này chứa yếu tố phủ định, liên từ, hoặc cấu trúc dài, vốn là thách thức đối với các mô hình chỉ dựa trên từ khóa.

#### Câu 1  
“can you remind me to not call my mom”  
Nhãn thật: reminder_create  

| Mô hình                    | Dự đoán          | Nhận xét                                   |
|-----------------------------|------------------|--------------------------------------------|
| TF-IDF + LR                 | reminder_create  | Dựa vào các từ khóa mạnh như remind, call  |
| Word2Vec (Avg) + Dense      | reminder_create / sai nhẹ | Không biểu diễn rõ phủ định       |
| Embedding + LSTM            | reminder_create  | Có khả năng nắm bắt quan hệ chuỗi          |
| Embedding scratch + LSTM    | Sai              | Mô hình không hội tụ                       |

**Nhận xét:**  
Yếu tố phủ định “not call” yêu cầu mô hình hiểu quan hệ giữa các từ trong chuỗi. LSTM có lợi thế lý thuyết, tuy nhiên TF-IDF vẫn cho kết quả chính xác nhờ các từ khóa đặc trưng.

---

#### Câu 2  
“is it going to be sunny or rainy tomorrow”  
Nhãn thật: weather_query  

| Mô hình                    | Dự đoán          | Nhận xét                                   |
|-----------------------------|------------------|--------------------------------------------|
| TF-IDF + LR                 | weather_query    | Từ khóa sunny, rainy, tomorrow             |
| Word2Vec (Avg) + Dense      | weather_query / không ổn định | Trung bình vector làm mờ ngữ nghĩa |
| Embedding + LSTM            | weather_query    | Học được cấu trúc câu hỏi                  |
| Embedding scratch + LSTM    | Sai              | Không học được biểu diễn                   |

**Nhận xét:**  
Trong câu hỏi dạng lựa chọn (sunny or rainy), LSTM có khả năng mô hình hóa cấu trúc tốt hơn, tuy nhiên lợi thế này chưa được thể hiện rõ do hạn chế về dữ liệu huấn luyện.

---

#### Câu 3  
“find a flight from new york to london but not through paris”  
Nhãn thật: flight_search  

| Mô hình                    | Dự đoán          | Nhận xét                                   |
|-----------------------------|------------------|--------------------------------------------|
| TF-IDF + LR                 | flight_search    | Nhận diện tốt từ khóa flight, from, to     |
| Word2Vec (Avg) + Dense      | Sai / không ổn định | Mất thông tin quan hệ tuyến              |
| Embedding + LSTM            | flight_search    | Có khả năng xử lý phủ định not through     |
| Embedding scratch + LSTM    | Sai              | Underfitting                               |

**Nhận xét:**  
Câu này chứa cấu trúc phức tạp với ràng buộc phủ định (not through paris). Về mặt lý thuyết, LSTM phù hợp hơn để xử lý dạng chuỗi này, tuy nhiên trong thực nghiệm, TF-IDF vẫn cho kết quả tốt nhờ các từ khóa mạnh liên quan đến tìm chuyến bay.

---

### 3.3. Kết luận chung

Mặc dù các mô hình LSTM có ưu thế về mặt lý thuyết trong việc xử lý chuỗi và cấu trúc câu phức tạp, nhưng trong bối cảnh bài toán phân loại ý định với dữ liệu có quy mô vừa và câu ngắn, TF-IDF kết hợp Logistic Regression vẫn là mô hình hiệu quả và ổn định nhất. Các mô hình deep learning chỉ phát huy được sức mạnh khi có embedding chất lượng cao (pre-trained thật sự) và tập dữ liệu đủ lớn.

## 4. Khó khăn và giải pháp

### **Khó khăn 1: Mất thông tin thứ tự từ trong các mô hình baseline**
* **Nguyên nhân:** các pipeline **TF-IDF + Logistic Regression** và **Word2Vec (trung bình) + Dense** *không mô hình hóa được thứ tự từ trong câu*.  
* **Ảnh hưởng:**  
  - Gặp khó khăn với các câu có **phủ định** (ví dụ: *“not call”*, *“not through”*).  
  - Khó xử lý câu có nhiều **mệnh đề** hoặc **liên từ** (*“but”*, *“or”*).  
* **Giải pháp:**  
  - Áp dụng mô hình **LSTM**, cho phép học quan hệ phụ thuộc theo chuỗi thời gian giữa các từ.  
  - Thực hiện **phân tích định tính** trên các câu “khó” để đánh giá rõ lợi thế của mô hình chuỗi.  

---

### **Khó khăn 2: Overfitting trong các mô hình học sâu**
* **Nguyên nhân:** các mô hình **LSTM** có số lượng tham số lớn, trong khi kích thước tập dữ liệu huấn luyện *không quá lớn*.  
* **Ảnh hưởng:**  
  - **Loss** trên tập *train* giảm nhanh nhưng *không cải thiện* hoặc thậm chí *tăng* trên tập *validation*.  
  - Mô hình học quá sát dữ liệu huấn luyện (**overfitting**).  
* **Giải pháp:**  
  - Sử dụng **Dropout** trong các lớp Dense và LSTM.  
  - Áp dụng **EarlyStopping** dựa trên loss của tập validation để dừng huấn luyện khi mô hình không còn cải thiện.  
  - So sánh **embedding pre-trained (đóng băng)** với **embedding học từ đầu** để kiểm soát mức độ học của mô hình.  

---

### **Khó khăn 3: Đảm bảo tính nhất quán trong tiền xử lý dữ liệu**
* **Nguyên nhân:** các mô hình **LSTM** yêu cầu quá trình tiền xử lý phức tạp hơn (*tokenization*, *padding*, *vocab size*).  
* **Ảnh hưởng:**  
  - Nếu *không nhất quán*, kết quả so sánh giữa các mô hình sẽ *không công bằng*.  
  - Sai lệch **vocab** hoặc **độ dài chuỗi** có thể dẫn đến *lỗi huấn luyện*.  
* **Giải pháp:**  
  - Sử dụng chung **Tokenizer**, **vocab_size** và **max_len** cho hai mô hình LSTM.  
  - Cố định **quy trình tiền xử lý** trước khi huấn luyện để đảm bảo tính **tái lập** của thí nghiệm.  

---

## 5. Tài liệu tham khảo

* **Gensim Documentation – Word2Vec**: [https://radimrehurek.com/gensim/](https://radimrehurek.com/gensim/)

* **scikit-learn Documentation – TF-IDF Vectorizer, Logistic Regression**: [https://scikit-learn.org/stable/modules/feature_extraction.html](https://scikit-learn.org/stable/modules/feature_extraction.html)

* **TensorFlow & Keras Documentation – Embedding Layer, LSTM, EarlyStopping**: [https://www.tensorflow.org/](https://www.tensorflow.org/)
