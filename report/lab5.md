# Báo cáo Lab5: Giải quyết bài toán phân loại văn bản bằng pipeline sử dụng các kỹ thuật tiền xử lý, mã hóa văn bản và model đã học

## Giới thiệu
Trong bài lab này, em triển khai và đánh giá một hệ thống phân loại văn bản (text classification) cho bài toán phân tích cảm xúc (sentiment analysis; xem mô tả dữ liệu tại [đây](../data/sentiments.md)). Nội dung bài lab bao gồm:

* Triển khai lớp TextClassifier sử dụng thư viện scikit-learn. ([file source](../src/models/text_classifier.py))
* Xây dựng test case cơ bản để kiểm tra pipeline phân loại văn bản. ([file source](../test/lab5_test.py))
* Chạy ví dụ Spark ML sentiment analysis để làm baseline trên tập dữ liệu lớn hơn. ([file source](../test/lab5_spark_sentiment_analysis.ipynb))
* Thực hiện thí nghiệm cải tiến mô hình bằng cách thay thế Logistic Regression bằng Naive Bayes. (cùng trong notebook Spark ML sentiment analysis phía trên)
* So sánh và phân tích kết quả giữa các mô hình.

Mục tiêu của bài lab là làm quen với quy trình xây dựng, đánh giá và cải tiến mô hình phân loại văn bản trong các bối cảnh dữ liệu khác nhau.

**Lưu ý:** lab5 này `không` phải phần báo cáo tổng hợp của các **`lab5_part{i}.md`**

## 1. Các bước thực hiện

### 1.1. Triển khai lớp TextClassifier (Task 1)
Lớp `TextClassifier` được triển khai trong file `src/models/text_classifier.py`, đóng vai trò kết hợp giữa bước biểu diễn văn bản và mô hình học máy.
Quy trình triển khai gồm các bước:

* Tiền xử lý và vector hóa: Văn bản đầu vào được chuyển đổi sang vector đặc trưng bằng CountVectorizer kết hợp với RegexTokenizer.
* Huấn luyện mô hình: Sử dụng LogisticRegression với solver liblinear.
* Dự đoán: Sinh nhãn dự đoán cho các văn bản mới.
* Đánh giá: Tính các chỉ số Accuracy, Precision, Recall và F1-score.

Các phương thức `fit`, `predict` và `evaluate` đều được triển khai đúng theo yêu cầu và hoạt động ổn định.

### 1.2. Test case cơ bản (lab5_test.py – Task 2)
File `test/lab5_test.py` được xây dựng nhằm kiểm tra tính đúng đắn của pipeline phân loại văn bản trên một tập dữ liệu sentiment nhỏ (6 mẫu).
Quy trình test bao gồm:

* Chia dữ liệu thành tập huấn luyện và tập kiểm tra theo tỉ lệ 80:20.
* Huấn luyện mô hình Logistic Regression.
* Thực hiện dự đoán và đánh giá kết quả.

Kết quả thu được:
```text
Evaluation Metrics: accuracy: 0.5000 precision: 0.5000 recall: 1.0000 f1: 0.6667
```

Do tập dữ liệu rất nhỏ, sau khi chia train–test, tập kiểm tra chỉ còn một mẫu. Vì vậy, các chỉ số đánh giá trong test này không mang ý nghĩa thống kê và chỉ được sử dụng để kiểm tra pipeline triển khai, không dùng để đánh giá hiệu năng mô hình.

### 1.3. Spark ML sentiment analysis (Baseline – Task 3)
Script `test/lab5_spark_sentiment_analysis.py` được chạy thành công để xây dựng baseline phân loại cảm xúc bằng Spark ML trên tập dữ liệu lớn hơn.
Thông qua ví dụ này, em nắm được các thành phần chính của Spark ML pipeline, bao gồm:

* Tokenizer
* Feature extraction
* Classifier trong môi trường xử lý phân tán

Kết quả đánh giá của mô hình Spark baseline:
```text
=== Evaluation Results === Accuracy : 0.7295 F1-score : 0.7266
```

Kết quả này phản ánh hiệu năng thực tế và đáng tin cậy hơn so với test case nhỏ trong `lab5_test.py`.

### 1.4. Thí nghiệm cải tiến mô hình (Task 4)
Để cải thiện mô hình, em thực hiện một thí nghiệm bằng cách thay thế Logistic Regression bằng Naive Bayes classifier, vốn thường phù hợp với dữ liệu văn bản dạng bag-of-words.
Thay vì tạo file kiểm thử mới `test/lab5_improvement_test.py`, để tiện hơn cho việc tái sử dụng dữ liệu đã chia và các module tiền xử lý trước đó, em thực hiện huấn luyện và đánh giá mô hình cải tiến ở ngay các cells bên dưới, cùng trong file `test/lab5_spark_sentiment_analysis.ipynb`.

Kết quả thu được:
```text
=== Improved Model (Naive Bayes) === Accuracy : 0.6844 F1-score : 0.6842
```

## 2. Hướng dẫn chạy code
Sau khi clone source code và kích hoạt môi trường ảo, chạy các lệnh sau tại thư mục gốc của project:

```bash
# Test pipeline cơ bản
python -m test.lab5_test

# Chạy Spark ML baseline (notebook này chứa cả phần thí nghiệm mô hình cải tiến)
Mở file test/lab5_spark_sentiment_analysis.ipynb → chọn kernel Python → bấm Run All hoặc chạy từng cell.

```

Kết quả sẽ được in trực tiếp ra console (đối với `.py`) và in ra dưới dạng các cells kết quả (đối với `.ipynb`).

## 3. Phân tích kết quả

### 3.1. Nhận xét về lab5_test (Logistic Regression – dữ liệu nhỏ)
Kết quả trong `lab5_test.py` cho thấy các chỉ số như recall và F1-score có thể đạt giá trị tương đối cao mặc dù accuracy thấp. Điều này xuất phát từ việc tập kiểm tra chỉ gồm một mẫu, khiến các metric trở nên không ổn định và dễ gây hiểu nhầm. Do đó, test này chỉ có ý nghĩa kiểm tra tính đúng đắn của pipeline, không phản ánh hiệu năng phân loại thực tế.

### 3.2. Nhận xét về Spark baseline
Mô hình Spark baseline đạt Accuracy ≈ 0.73 và F1-score ≈ 0.73. Đây là kết quả đáng tin cậy hơn do được đánh giá trên tập dữ liệu lớn và quy trình huấn luyện – đánh giá đầy đủ hơn. Kết quả này được sử dụng làm baseline chính để so sánh.

### 3.3. Nhận xét về mô hình cải tiến (Naive Bayes)
So với Spark baseline, mô hình Naive Bayes cho kết quả thấp hơn. Tuy nhiên, Naive Bayes vẫn cho hiệu năng tương đối ổn định và phù hợp với bối cảnh dữ liệu nhỏ và biểu diễn đặc trưng dạng bag-of-words. Thí nghiệm này minh họa ảnh hưởng của việc lựa chọn mô hình đối với kết quả phân loại văn bản.

### 3.4. So sánh các mô hình

| Mô hình | Ngữ cảnh sử dụng | Accuracy | F1-score |
| :--- | :--- | :--- | :--- |
| Logistic Regression (lab5_test) | Test pipeline, dữ liệu rất nhỏ | 0.5000 | 0.6667 |
| Spark ML baseline | Dữ liệu lớn | 0.7295 | 0.7266 |
| Naive Bayes (improved) | Dữ liệu vừa/nhỏ | 0.6844 | 0.6842 |

## 4. Khó khăn và giải pháp
**Khó khăn 1: Kết quả đánh giá dễ gây hiểu nhầm trên tập dữ liệu nhỏ**
* **Nguyên nhân:** tập kiểm tra quá nhỏ sau khi chia train–test.
* **Giải pháp:** chỉ sử dụng test này để kiểm tra pipeline và dựa vào Spark baseline để đánh giá hiệu năng.

**Khó khăn 2: Lựa chọn mô hình phù hợp cho từng bối cảnh dữ liệu**
* **Nguyên nhân:** mỗi mô hình có giả định và yêu cầu dữ liệu khác nhau.
* **Giải pháp:** so sánh Logistic Regression và Naive Bayes để quan sát sự khác biệt.

## 5. Tài liệu tham khảo
* Scikit-learn – Text Feature Extraction: [https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
* Scikit-learn – Working with Text Data (Tutorial): [https://scikit-learn.org/1.3/tutorial/text_analytics/working_with_text_data.html](https://scikit-learn.org/1.3/tutorial/text_analytics/working_with_text_data.html)
* Apache Spark ML – Pipelines: [https://spark.apache.org/docs/latest/ml-pipeline.html](https://spark.apache.org/docs/latest/ml-pipeline.html)