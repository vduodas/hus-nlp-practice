# Báo cáo Lab5 Part 1: Giới thiệu PyTorch và các khái niệm cơ bản

## Giới thiệu

Trong phần lab5 part 1 này, em thực hiện học tập và thí nghiệm với **PyTorch** – một thư viện học sâu phổ biến. Nội dung bài lab bao gồm:

* Khám phá cấu trúc dữ liệu **Tensor** – đơn vị cơ bản trong PyTorch.
* Thực hiện các phép toán trên tensor (cộng, nhân, nhân ma trận).
* Sử dụng **indexing và slicing** để trích xuất dữ liệu từ tensor.
* Thay đổi hình dạng (reshape) tensor.
* Tìm hiểu về **Automatic Differentiation (autograd)** – cơ chế tự động tính đạo hàm trong PyTorch.
* Xây dựng các mô hình đầu tiên sử dụng `torch.nn` module, bao gồm các lớp `Linear`, `Embedding` và tổ hợp chúng thành một mô hình hoàn chỉnh.

Mục tiêu của bài lab là nắm vững các khái niệm cơ bản của PyTorch, từ đó làm nền tảng cho việc xây dựng các mô hình học sâu phức tạp hơn.

---

## 1. Các bước thực hiện

### 1.1. Khám phá Tensor

Tensor là đơn vị cơ bản trong PyTorch – một mảng dữ liệu đa chiều có khả năng tính toán trên GPU. Bước đầu tiên bao gồm:

* **Tạo tensor từ các nguồn dữ liệu khác nhau**:
  * Từ Python list: `torch.tensor(data)`
  * Từ NumPy array: `torch.from_numpy(np_array)`
  * Từ các hàm tạo tensor: `torch.ones_like()`, `torch.rand_like()`

* **Kiểm tra thuộc tính tensor**:
  * Shape (kích thước): `tensor.shape`
  * Datatype (kiểu dữ liệu): `tensor.dtype`
  * Device (thiết bị lưu trữ): `tensor.device`

Việc hiểu rõ các cách tạo tensor và kiểm tra thuộc tính là cơ sở để thao tác hiệu quả với dữ liệu trong PyTorch.

---

### 1.2. Phép toán trên Tensor

PyTorch hỗ trợ các phép toán cơ bản trên tensor:

* **Phép cộng**: `x + y` hoặc `torch.add(x, y)`
* **Phép nhân vô hướng**: `x * 5`
* **Nhân ma trận**: `x @ y` (phép nhân ma trận) hoặc `torch.matmul(x, y)`

Các phép toán này hoạt động theo cơ chế **broadcasting**, cho phép thực hiện phép toán giữa các tensor có hình dạng tương thích. Ví dụ:
* Cộng hai tensor có shape `(2, 2)` đơn giản là cộng từng phần tử tương ứng.
* Nhân tensor `(2, 2)` với tensor transpose của nó `(2, 2)` sinh ra tensor `(2, 2)` là tích ma trận.

---

### 1.3. Indexing và Slicing

Để trích xuất dữ liệu từ tensor, em sử dụng indexing và slicing:

* **Indexing**: `tensor[0]` lấy hàng đầu tiên, `tensor[1, 1]` lấy phần tử tại hàng 2 cột 2.
* **Slicing**: `tensor[:, 1]` lấy toàn bộ cột thứ 2, `tensor[0:2]` lấy hai hàng đầu tiên.

Cơ chế indexing trong PyTorch tương tự NumPy, cho phép thao tác linh hoạt với dữ liệu.

---

### 1.4. Thay đổi hình dạng Tensor

Thường xuyên cần phải thay đổi hình dạng tensor để phù hợp với đầu vào của các lớp mạng:

* **`reshape()`**: Thay đổi hình dạng tensor, trả về tensor mới có cùng dữ liệu.
* **`view()`**: Thay đổi hình dạng tensor tương tự `reshape()`, nhưng yêu cầu tensor liên tục trong bộ nhớ.

Ví dụ: tensor shape `(4, 4)` có 16 phần tử, có thể reshape thành `(16, 1)`, `(2, 8)`, v.v...

---

### 1.5. Automatic Differentiation (Autograd)

Một trong những tính năng mạnh nhất của PyTorch là **autograd** – cơ chế tự động tính đạo hàm. Quy trình:

1. **Kích hoạt tính đạo hàm**: Khi tạo tensor, đặt `requires_grad=True` để PyTorch theo dõi các phép toán trên tensor này.
2. **Xây dựng computational graph**: Các phép toán được lưu lại dưới dạng một đồ thị tính toán (computational graph).
3. **Tính đạo hàm**: Gọi `.backward()` để tính đạo hàm theo chuỗi quy tắc (chain rule).
4. **Lấy kết quả đạo hàm**: Đạo hàm được lưu trong thuộc tính `.grad` của tensor.

**Ví dụ thực tế**: Với $z = 3(x+2)^2$, đạo hàm theo $x$ là $\frac{dz}{dx} = 6(x+2)$. Khi $x=1$, ta có $\frac{dz}{dx} = 18$.

**Lưu ý quan trọng**: Sau khi gọi `.backward()`, PyTorch mặc định giải phóng computational graph để tiết kiệm bộ nhớ. Nếu cần gọi `.backward()` nhiều lần, phải sử dụng `retain_graph=True`.

---

### 1.6. Xây dựng mô hình với `torch.nn`

PyTorch cung cấp module `torch.nn` chứa các lớp tiêu chuẩn để xây dựng mạng nơ-ron. Ba lớp cơ bản được khám phá:

#### 1.6.1. `nn.Linear`
* Thực hiện phép biến đổi tuyến tính: $y = xW^T + b$
* Chuyển đổi tensor từ `in_features` chiều sang `out_features` chiều.
* Ví dụ: `nn.Linear(5, 2)` chuyển đổi từ 5 chiều thành 2 chiều.

#### 1.6.2. `nn.Embedding`
* Là một bảng tra cứu ánh xạ chỉ số từ sang vector embedding.
* Hữu ích khi xử lý dữ liệu văn bản – mỗi từ được biểu diễn bằng một chỉ số, và embedding layer chuyển đổi chỉ số thành vector dense.
* Ví dụ: `nn.Embedding(10, 3)` tạo 10 embedding vector, mỗi vector có 3 chiều.

#### 1.6.3. Kết hợp thành `nn.Module`
* `nn.Module` là lớp cơ sở để định nghĩa các mô hình tùy chỉnh.
* Định nghĩa các lớp trong `__init__()` và sắp xếp logic trong `forward()`.
* Ví dụ thực tế:
  ```python
  class MyFirstModel(nn.Module):
      def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
          super(MyFirstModel, self).__init__()
          self.embedding = nn.Embedding(vocab_size, embedding_dim)
          self.linear = nn.Linear(embedding_dim, hidden_dim)
          self.activation = nn.ReLU()
          self.output_layer = nn.Linear(hidden_dim, output_dim)
      
      def forward(self, indices):
          embeds = self.embedding(indices)
          hidden = self.activation(self.linear(embeds))
          output = self.output_layer(hidden)
          return output
  ```

Mô hình này kết hợp embedding → linear layer → ReLU activation → output layer, tạo thành một mạng nơ-ron đơn giản nhưng đầy đủ.

---

## 2. Hướng dẫn chạy code

Sau khi clone source code và kích hoạt môi trường ảo, chạy notebook sau:

```bash
Mở file notebook/pytorch_introduction.ipynb → chọn kernel Python → bấm Run All hoặc chạy từng cell.
```

Notebook được tổ chức thành ba phần chính:
* **Phần 1**: Khám phá Tensor
* **Phần 2**: Autograd (tính đạo hàm tự động)
* **Phần 3**: Xây dựng mô hình với `torch.nn`

Mỗi phần chứa các task cụ thể và output minh họa.

---

## 3. Phân tích kết quả

### 3.1. Nhận xét về Tensor

Tensor là trừu tượng hóa của mảng dữ liệu đa chiều, cho phép thực hiện các phép toán nhanh chóng trên GPU. Việc hiểu rõ shape, dtype và device là rất quan trọng để tránh lỗi runtime (ví dụ: lỗi không khớp shape khi nhân ma trận).

### 3.2. Nhận xét về Autograd

Autograd là tính năng đặc biệt của PyTorch, cho phép tự động tính đạo hàm mà không cần phải viết công thức đạo hàm bằng tay. Điều này rất hữu ích khi xây dựng các mô hình phức tạp với hàng trăm lớp. Tuy nhiên, cần chú ý đến việc quản lý bộ nhớ – computational graph có thể tiêu tốn nhiều bộ nhớ đối với các mô hình lớn.

### 3.3. Nhận xét về `torch.nn`

Các lớp trong `torch.nn` (Linear, Embedding, ReLU, v.v...) cung cấp các khối xây dựng tiêu chuẩn cho mạng nơ-ron. Kế thừa từ `nn.Module` cho phép định nghĩa các mô hình tùy chỉnh với:
* Khả năng lưu/tải (saving/loading) mô hình dễ dàng.
* Hỗ trợ tính toán đạo hàm tự động thông qua autograd.
* Khả năng chuyển mô hình lên GPU chỉ bằng một lệnh `.to(device)`.

### 3.4. Sơ đồ so sánh các khối xây dựng

| Lớp | Chức năng | Đầu vào | Đầu ra |
| :--- | :--- | :--- | :--- |
| `nn.Linear` | Biến đổi tuyến tính | Tensor shape `(*, in_features)` | Tensor shape `(*, out_features)` |
| `nn.Embedding` | Tra cứu embedding | Tensor chỉ số shape `(*, seq_len)` | Tensor shape `(*, seq_len, embedding_dim)` |
| `nn.ReLU` | Hàm kích hoạt | Tensor bất kỳ | Tensor cùng shape |
| Custom Module | Kết hợp các lớp | Tùy vào định nghĩa | Tùy vào định nghĩa |

---

## 4. Kết luận

Qua bài lab này, em đã nắm được các khái niệm cơ bản của PyTorch:
* Tensor là đơn vị dữ liệu cơ bản, với các phép toán hiệu quả trên CPU/GPU.
* Autograd cho phép tính đạo hàm tự động, nền tảng của việc huấn luyện mạng nơ-ron.
* `torch.nn` cung cấp các lớp tiêu chuẩn để xây dựng mô hình, từ đơn giản đến phức tạp.

Những kiến thức này sẽ là nền tảng để tiếp tục học các kỹ thuật nâng cao như RNN, CNN, Transformer trong các phần tiếp theo.

## Tài liệu tham khảo
- PyTorch – Official Documentation: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)