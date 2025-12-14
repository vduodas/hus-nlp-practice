# Báo cáo Lab4: Mã hóa văn bản bằng vector dày đặc

## Giới thiệu

Trong bài lab này, em thực hiện thí nghiệm với **word embedding** dựa trên mô hình **Word2Vec**, bao gồm ba hướng tiếp cận chính:

* Sử dụng **Word2Vec pre-trained** để khảo sát độ tương đồng ngữ nghĩa và phép suy luận tương tự (analogy).
* Huấn luyện **Word2Vec từ đầu (from scratch)** bằng **Gensim** trên tập dữ liệu **UD English Web Treebank (UD English-EWT)** (xem mô tả tại [đây](../data/UD_English_EWT.md)).
* Sử dụng **Spark MLlib Word2Vec** để huấn luyện embedding trên tập dữ liệu lớn hơn (**C4 – c4-train.00000-of-01024-30K.json**), từ đó so sánh ảnh hưởng của **quy mô dữ liệu** và **framework xử lý** đến chất lượng embedding. (Xem mô tả dữ liệu tại [đây](../data/c4-train.00000-of-01024-30K.md))

Mục tiêu của bài lab là làm rõ vai trò của dữ liệu, mô hình và công cụ trong việc xây dựng biểu diễn ngữ nghĩa dạng vector dày đặc.

---

## 1. Các bước thực hiện

### 1.1. Chuẩn bị dữ liệu

* Sử dụng hai nguồn dữ liệu chính:

  * **UD English-EWT**: dữ liệu dạng text, quy mô nhỏ, dùng cho huấn luyện Word2Vec bằng Gensim.
  * **C4 (Colossal Clean Crawled Corpus)** – file `c4-train.00000-of-01024-30K.json`: dữ liệu quy mô lớn hơn, dùng cho huấn luyện Word2Vec bằng Spark.
* Dữ liệu được xử lý theo dạng **stream** hoặc **distributed processing** nhằm tiết kiệm bộ nhớ.
* Mỗi câu được tách thành danh sách token và làm đầu vào cho mô hình Word2Vec.

---

### 1.2. Sử dụng Word2Vec pre-trained

* Tải và sử dụng model Word2Vec đã được huấn luyện sẵn (ví dụ: Google News).
* Thực hiện các thao tác:

  * Lấy vector biểu diễn của từ.
  * Tính độ tương đồng cosine giữa các cặp từ.
  * Tìm các từ gần nhất (*most similar*).
  * Thực hiện phép toán vector cho bài toán analogy.
  * Tạo embedding cho câu bằng cách lấy trung bình vector các từ.

Mục đích của bước này là sử dụng **pre-trained model làm baseline** để đánh giá chất lượng các mô hình tự huấn luyện.

---

### 1.3. Huấn luyện Word2Vec từ đầu bằng Gensim (UD English-EWT)

* Huấn luyện Word2Vec trực tiếp trên tập **UD English-EWT** với Gensim.
* Quy trình gồm:

  * Streaming dữ liệu từ file văn bản.
  * Huấn luyện mô hình Word2Vec.
  * Lưu model ra file để tái sử dụng.
* Sau huấn luyện, tiến hành:

  * Tìm các từ tương tự với một từ cho trước.
  * Kiểm tra bài toán analogy.

Cách tiếp cận này giúp quan sát rõ hạn chế của việc huấn luyện embedding trên **corpus nhỏ**.

---

### 1.4. Huấn luyện Word2Vec bằng Spark (C4 dataset)

* Sử dụng **Apache Spark MLlib – Word2Vec** để huấn luyện embedding.
* Dữ liệu đầu vào là file **`c4-train.00000-of-01024-30K.json`**, có quy mô lớn hơn đáng kể so với UD English-EWT.
* Quy trình thực hiện:

  * Đọc dữ liệu JSON bằng Spark DataFrame.
  * Trích xuất trường văn bản và tokenize thành danh sách từ.
  * Huấn luyện Word2Vec phân tán trên Spark.
  * Truy vấn các từ tương đồng (*findSynonyms*).

Việc sử dụng Spark cho phép:

* Xử lý dữ liệu lớn hiệu quả hơn.
* Khai thác lợi thế của **distributed training** cho embedding.

---

### 1.5. Trực quan hóa embedding

* Sử dụng phương pháp giảm chiều như **PCA** (và/hoặc t-SNE) để chiếu embedding về không gian 2 chiều.
* Áp dụng chủ yếu cho embedding **tự huấn luyện bằng Gensim** nhằm phân tích cấu trúc không gian vector.
* Quan sát vị trí tương đối và sự hình thành cụm của các từ.

---

## 2. Hướng dẫn chạy code

Sau khi clone source code và kích hoạt môi trường ảo, chạy các lệnh sau tại thư mục gốc của project:

```bash
Mở file test/lab4_word2vec_embedding.ipynb → chọn kernel Python → bấm Run All hoặc chạy từng cell.

python -m test.lab4_test

python -m test.lab4_embedding_training_demo
```

Trong đó:

* `lab4_spark_word2vec_demo`: huấn luyện và demo Word2Vec bằng Spark trên C4 dataset. [Link tới file mã nguồn](../test/lab4_spark_word2vec_demo.py)
* `lab4_test`: demo Word2Vec pre-trained. [Link tới file mã nguồn](../test/lab4_test.py)
* `lab4_word2vec_embedding.ipynb`: huấn luyện Word2Vec từ đầu bằng Gensim trên UD English-EWT và trực quan hóa embeddings. [Link tới file mã nguồn](../test/lab4_word2vec_embedding.ipynb)

Kết quả được in ra console và model huấn luyện được lưu vào thư mục `results/`.

---

## 3. Phân tích kết quả

### 3.1. Nhận xét về model Word2Vec pre-trained

Kết quả thu được cho phần này như sau:
```
Vector for 'king':
[ 0.50451   0.68607  -0.59517  -0.022801  0.60046  -0.13498  -0.08813
  0.47377  -0.61798  -0.31012  -0.076666  1.493    -0.034189 -0.98173
  0.68229   0.81722  -0.51874  -0.31503  -0.55809   0.66421   0.1961
 -0.13495  -0.11476  -0.30344   0.41177  -2.223    -1.0756   -1.0783
 -0.34354   0.33505   1.9927   -0.04234  -0.64319   0.71125   0.49159
  0.16754   0.34344  -0.25663  -0.8523    0.1661    0.40102   1.1685
 -1.0137   -0.21585  -0.15155   0.78321  -0.91241  -1.6106   -0.64426
 -0.51042 ]
--------------------------------------------------
Similarity king - queen: 0.7839043140411377
Similarity king - man: 0.5309377312660217
--------------------------------------------------
10 most similar words to 'computer':
computers: 0.9165045022964478
software: 0.8814994096755981
technology: 0.8525559306144714
electronic: 0.812586784362793
internet: 0.8060454726219177
computing: 0.802603542804718
devices: 0.8016185760498047
digital: 0.7991792559623718
applications: 0.7912740707397461
pc: 0.7883161306381226
--------------------------------------------------
Document embedding for:
"The queen rules the country."
[ 0.04564168  0.36530998 -0.55974334  0.04014383  0.09655549  0.15623933
 -0.33622834 -0.12495166 -0.01031508 -0.5006717   0.18690467  0.17482166
 -0.268985   -0.03096624  0.36686516  0.29983264  0.01397333 -0.06872118
 -0.3260683  -0.210115    0.16835399 -0.03151734 -0.06204716  0.04301083
 -0.06958768 -1.7792168  -0.54365396 -0.06104483 -0.17618     0.009181
  3.3916333   0.08742473 -0.4675417  -0.213435    0.02391887 -0.04470453
  0.20636833 -0.12902866 -0.28527132 -0.2431805  -0.3114423  -0.03833717
  0.11977985 -0.01418401 -0.37086335  0.22069354 -0.28848937 -0.36188802
 -0.00549529 -0.46997246]

```

**a) Độ tương đồng và từ đồng nghĩa**
Với từ khóa **“computer”**, các từ tương tự gồm:

> computers, software, technology, electronic, internet, pc

* Các từ tìm được đều thuộc cùng trường nghĩa công nghệ.
* Độ tương đồng cao phản ánh mối liên hệ ngữ nghĩa rõ ràng.
* Điều này cho thấy model pre-trained đã học được cấu trúc ngữ nghĩa tốt nhờ được huấn luyện trên corpus rất lớn.

**b) Quan hệ ngữ nghĩa (analogy)**

* Similarity `king - queen` ≈ 0.78
* Similarity `king - man` ≈ 0.53
* Phép toán `king - man + woman` cho kết quả gần với `queen`.

⟹ Model pre-trained thể hiện đúng đặc trưng đại số ngữ nghĩa của Word2Vec.

---

### 3.2. Kết quả Word2Vec huấn luyện bằng Spark (C4 dataset)

Kết quả thu được cho phần này như sau:
```
Top 5 words similar to 'computer':
+----------+------------------+
|word      |similarity        |
+----------+------------------+
|computers |0.7049033641815186|
|desktop   |0.7020003199577332|
|laptop    |0.6919171214103699|
|device    |0.6692103147506714|
|wirelessly|0.6493337154388428|
+----------+------------------+
```

* Với từ khóa **“computer”**, các từ tương đồng tìm được như:

> computers, desktop, laptop, device, wirelessly

* Các từ có liên quan chặt chẽ về mặt ngữ nghĩa và ngữ cảnh sử dụng.
* So với mô hình Gensim huấn luyện trên UD English-EWT, kết quả từ Spark Word2Vec hợp lý và ổn định hơn.

⟹ Điều này cho thấy **quy mô dữ liệu lớn (C4)** đóng vai trò quyết định trong việc học embedding, ngay cả khi sử dụng mô hình Word2Vec tương đối đơn giản.

---

### 3.3. Kết quả huấn luyện Word2Vec từ đầu bằng Gensim (UD English-EWT)

Kết quả thu được cho phần này như sau:
```
--- DEMO ---

Most similar words to 'computer':
sons: 0.9992411732673645
restrictions: 0.9991543292999268
barely: 0.9991511702537537
ring: 0.9991382956504822
adult: 0.9991285800933838

Analogy: king - man + woman ≈ ?
uvb: 0.9947797656059265
sold: 0.994763970375061
scheduled: 0.9947484731674194
approach: 0.9947425723075867
counsel: 0.994651198387146
manage: 0.9946424961090088
colors: 0.9946327209472656
google: 0.9945788979530334
feathers: 0.9945738315582275
standard: 0.9945706129074097
```

Với từ **“computer”**, kết quả thu được:

> sons, restrictions, barely, ring, adult
> (similarity ≈ 0.999)

* Các từ không liên quan về mặt ngữ nghĩa.
* Độ tương đồng gần 1 cho thấy hiện tượng **vector collapse**.
* Bài toán analogy (`king - man + woman`) cho kết quả không mang ý nghĩa ngữ nghĩa rõ ràng.

⟹ Model chưa học được biểu diễn ngữ nghĩa hiệu quả do corpus nhỏ và thiếu đa dạng ngữ cảnh.

---

### 3.4. Trực quan hóa embedding (PCA)

* Embedding từ mô hình Gensim (UD English-EWT) được chiếu xuống 2 chiều bằng **PCA**.
* Quan sát cho thấy:

  * Một số từ có xu hướng nằm gần nhau theo chủ đề ngữ nghĩa (ví dụ: **“man”** với **“king”**).
  * Tuy nhiên các cụm không tách biệt rõ ràng. Điều này cho thấy embedding học được từ corpus nhỏ chỉ phản ánh một phần mối quan hệ ngữ cảnh, chưa đủ để hình thành các cụm ngữ nghĩa ổn định.
  * Ngoài ra, mô hình cũng được huấn luyện với nhiều giá trị tham số khác nhau, ví dụ `window` từ 5 tới 10, hay `vector_size` từ 50 đến 150.
  Có một điểm thú vị là **“king”** đều luôn khá gần **“man”**, **“woman”** và **“computer”** (hoặc **“laptop”**). Có thể là: vị vua (king) là đàn ông (man), thích phụ nữ (woman) và computer/laptop. 

Phân tích PCA giúp minh họa trực quan hạn chế của embedding khi dữ liệu huấn luyện không đủ lớn.

---

### 3.5. So sánh các mô hình

| Tiêu chí             | Pre-trained | Spark (C4) | Gensim (UD EWT)    |
| :------------------- | :---------- | :--------- | :----------------- |
| Quy mô corpus        | Rất lớn     | Lớn        | Nhỏ (~254k tokens) |
| Chất lượng ngữ nghĩa | Rất tốt     | Tốt        | Kém                |
| Similar words        | Hợp lý      | Hợp lý     | Không hợp lý       |
| Analogy              | Chính xác   | Tương đối  | Không chính xác    |

**Kết luận:** Word2Vec phụ thuộc rất mạnh vào **quy mô và chất lượng dữ liệu**. Việc sử dụng Spark trên tập dữ liệu lớn giúp cải thiện đáng kể chất lượng embedding so với huấn luyện từ đầu trên corpus nhỏ.

---

## 4. Khó khăn và giải pháp

**Khó khăn 1: Kết quả embedding từ model Gensim tự huấn luyện không hợp lý**

* *Nguyên nhân:* corpus nhỏ, tham số mặc định chưa phù hợp.
* *Giải pháp:* điều chỉnh tham số, so sánh với pre-trained và Spark Word2Vec làm baseline.

**Khó khăn 2: Xử lý dữ liệu lớn với Spark**

* *Nguyên nhân:* dữ liệu JSON lớn, yêu cầu tiền xử lý phân tán.
* *Giải pháp:* sử dụng Spark DataFrame và pipeline MLlib.

---

## 5. Tài liệu tham khảo

1. Mikolov et al., *Efficient Estimation of Word Representations in Vector Space*, 2013.
2. Universal Dependencies English Web Treebank.
3. Gensim Word2Vec documentation.
4. Apache Spark MLlib Word2Vec documentation.
