# HUS-NLP Practice

Repository này chứa các bài thực hành NLP (Natural Language Processing) trong khuôn khổ học tập và luyện tập tại HUS, bao gồm các chủ đề như tokenization, vectorization, word embedding, các bài toán xử lý dữ liệu văn bản (NER, POS tagging, ...), thử nghiệm mô hình, ...

Mục tiêu của repo:
- Thực hành các kỹ thuật NLP theo từng chủ đề
- Dễ dàng tái chạy thí nghiệm và kiểm tra kết quả
- Tách bạch rõ ràng giữa code, notebook, dữ liệu và báo cáo

---

## Cấu trúc thư mục

```text
./
├── src/        # Source code chính (.py)
├── notebook/   # Notebook phục vụ thử nghiệm nhanh theo từng chủ đề
├── test/       # Script test và demo
├── data/       # Dữ liệu đầu vào (chỉ chứa mô tả)
├── report/     # Báo cáo và mô tả thí nghiệm
├── README.md   # Mô tả tổng quan repository
└── .gitignore  # Loại bỏ các tệp không cần thiết hoặc quá lớn
```

---

## Yêu cầu môi trường

- Python >= 3.9
- Khuyến nghị sử dụng virtual environment

---

## Cài đặt

Tại thư mục gốc của project:

```bash
pip install -r requirements.txt
```

Cài đặt project ở chế độ editable để có thể import các module nội bộ:

```bash
pip install -e .
```

---

## Cách chạy ví dụ

Chạy theo module path:

```bash
python -m test.lab4_test
```

Hoặc chạy trực tiếp file:

```bash
python test/lab4_spark_word2vec_demo.py
```

---

## Thông tin

Repository phục vụ mục đích học tập và thực hành NLP.
