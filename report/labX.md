# BÁO CÁO NGHIÊN CỨU: TỔNG QUAN & CÁC PHƯƠNG PHÁP TRIỂN KHAI TEXT-TO-SPEECH (TTS)

---

## Giới thiệu

Trong bối cảnh sự phát triển nhanh chóng của Trí tuệ Nhân tạo (AI) và các hệ thống tương tác người – máy, **Text-to-Speech (TTS)** đã trở thành một thành phần nền tảng, đóng vai trò cầu nối giữa ngôn ngữ viết và giao tiếp bằng giọng nói tự nhiên. Bài toán TTS không chỉ dừng lại ở việc “đọc to văn bản”, mà hướng tới việc **tái tạo tiếng nói giống con người cả về nội dung, ngữ điệu, cảm xúc và ngữ cảnh sử dụng**.

Báo cáo này được thực hiện với mục tiêu:
- Cung cấp **bức tranh tổng quan** về tình hình nghiên cứu và sự tiến hóa của các phương pháp TTS.
- Phân tích **các hướng tiếp cận chính** (từ Rule-based đến Generative AI), tương ứng với các mức độ phát triển công nghệ.
- Đánh giá **ưu điểm – nhược điểm** của từng hướng triển khai và chỉ ra các **trường hợp ứng dụng phù hợp**.
- Trình bày cách các nghiên cứu và hệ thống hiện đại xây dựng **pipeline tối ưu** nhằm dung hòa giữa chất lượng, tốc độ, tài nguyên và tính an toàn.

Qua đó, báo cáo hướng tới việc giúp người học hình thành cái nhìn tổng thể và có hệ thống về bài toán **Text-to-Speech**. Đây là bước khởi đầu nhằm hỗ trợ việc định hướng học tập và nghiên cứu tiếp theo, trước khi đi sâu vào các mô hình, thuật toán và triển khai kỹ thuật cụ thể trong lĩnh vực Speech Processing, Multimodal AI và Human–Computer Interaction.


---

## 1. Tổng quan tình hình nghiên cứu (State of the Art)

Bài toán Text-to-Speech (TTS), hay Tổng hợp tiếng nói, hiện đang nằm tại giao điểm của ba lĩnh vực lớn: **Xử lý tín hiệu số (DSP)**, **Xử lý ngôn ngữ tự nhiên (NLP)** và **Generative AI (AI tạo sinh)**.

Trong thập kỷ qua, lĩnh vực này đã chứng kiến sự chuyển dịch mô hình nghiên cứu (paradigm shift) mạnh mẽ:

- **Mục tiêu nghiên cứu**: Chuyển từ việc chỉ đảm bảo *độ dễ hiểu* (Intelligibility) sang chinh phục *độ tự nhiên* (Naturalness) và hiện tại là *khả năng biểu đạt & sao chép* (Expressiveness & Zero-shot Cloning).
- **Xu hướng công nghệ**: Sự thống trị của các phương pháp ghép nối truyền thống đã hoàn toàn bị thay thế bởi **Neural TTS** (TTS dựa trên mạng nơ-ron). Gần đây nhất, sự bùng nổ của **Large Language Models (LLMs)** đã mở ra hướng đi mới: coi âm thanh như một “ngôn ngữ”, cho phép mô hình không chỉ *đọc* mà còn *hiểu* ngữ cảnh để diễn xuất.
- **Thách thức cốt lõi**: Thách thức lớn nhất hiện nay của cộng đồng nghiên cứu không còn là làm cho máy nói giống người, mà là giải quyết bài toán **“Tam giác cân bằng”**:
  - Chất lượng âm thanh cao  
  - Tốc độ suy luận nhanh (Real-time)  
  - Tài nguyên tính toán thấp  

---

## 2. Các hướng tiếp cận và sự tiến hóa của công nghệ TTS

### Level 1: Kỷ nguyên luật & Ghép nối (Rule-based & Concatenative)

Đây là nền móng đầu tiên của ngành tổng hợp tiếng nói, nơi con người can thiệp sâu vào các quy tắc.

**Cơ chế hoạt động**:  
Hệ thống hoạt động dựa trên việc cắt nhỏ giọng người thật thành các đơn vị âm thanh nhỏ (từ hoặc âm vị) và lưu trong cơ sở dữ liệu cực lớn. Khi có văn bản, máy tính sẽ dùng các thuật toán tìm kiếm và ghép nối các mảnh này lại dựa trên luật ngôn ngữ (Grapheme-to-Phoneme).

**Ưu điểm**:
- **Tốc độ phản hồi cực nhanh:** Do không cần tính toán phức tạp, chỉ là tra cứu và ghép.
- **Kiểm soát chính xác:** Rất hiếm khi đọc sai từ nếu từ điển đã được định nghĩa đúng.

**Nhược điểm**:
- **Thiếu tự nhiên:** Âm thanh thường bị “giật cục” ở các điểm nối, ngữ điệu đều đều như robot.
- **Khó mở rộng:** Muốn thay đổi giọng nói, phải thu âm lại toàn bộ từ điển dữ liệu mới.

---

### Level 2: Deep Learning & Cá nhân hóa (Personalized Fine-tuning)

Sự ra đời của Deep Learning đã khắc phục được sự cứng nhắc của Level 1, tạo ra sự mượt mà vượt trội.

**Cơ chế hoạt động**:  
Thay vì ghép nối thủ công, các mô hình mạng nơ-ron (như Tacotron, FastSpeech) học cách ánh xạ trực tiếp từ văn bản sang biểu đồ đặc trưng âm thanh (Mel-spectrogram). Để tạo ra giọng nói riêng biệt, hệ thống sử dụng phương pháp **Fine-tuning**: tinh chỉnh mô hình gốc trên một tập dữ liệu nhỏ của người dùng cụ thể.

**Ưu điểm**:
- **Độ tự nhiên cao:** Các nối âm mượt mà, ngữ điệu lên xuống giống người thật.
- **Tiết kiệm tài nguyên vận hành:** Khi suy luận (inference), mô hình nhẹ hơn so với Level 3.

**Nhược điểm**:
- **Yêu cầu dữ liệu sạch:** Cần dữ liệu thu âm chất lượng cao, ít nhiễu.
- **Độ trễ trong triển khai:** Không thể tạo giọng ngay lập tức, cần thời gian huấn luyện cho mỗi giọng mới.

---

### Level 3: Zero-Shot & Few-Shot Generative Models

Đây là bước tiến hiện đại nhất, hướng tới khả năng khái quát hóa và sao chép giọng nói tức thì.

**Cơ chế hoạt động**:  
Sử dụng kiến trúc Transformer hoặc Diffusion Models. Mô hình được huấn luyện trên tập dữ liệu khổng lồ, đa ngôn ngữ và đa giọng nói. Khi cần tạo giọng mới, hệ thống chỉ cần vài giây mẫu âm thanh để trích xuất **"vector đặc trưng"** (Speaker Embedding) và sinh ra âm thanh mới mà không cần huấn luyện lại.

**Ưu điểm**:
- **Instant Cloning (Sao chép tức thì):** Có thể nhại giọng bất kỳ ai chỉ với mẫu âm thanh cực ngắn, kể cả mẫu có nhiễu.
- **Đa ngữ & Đa cảm xúc:** Có khả năng chuyển giọng nói sang ngôn ngữ khác hoặc thay đổi thái độ (vui, buồn, thì thầm) linh hoạt.

**Nhược điểm**:
- **Tài nguyên tính toán lớn:** Đòi hỏi GPU mạnh để vận hành mượt mà.
- **Vấn đề ổn định:** Đôi khi gặp hiện tượng "ảo giác" (nói lặp từ, bỏ từ, hoặc tự cười/khóc không kiểm soát).

---

## 3. Chiến lược tối ưu Pipeline (Giải quyết nhược điểm từng hướng)

Để đưa các nghiên cứu vào ứng dụng thực tế, các nhà phát triển thường xây dựng các Pipeline kết hợp nhiều kỹ thuật nhằm tối đa hóa ưu điểm và giảm thiểu hạn chế:

### 3.1. Tăng tốc độ & Giảm tài nguyên (Optimizing Inference)
- **Non-Autoregressive Models:** Áp dụng cho Level 2 & 3. Thay vì sinh âm thanh tuần tự từng chút một (rất chậm), kỹ thuật này cho phép sinh toàn bộ câu nói song song cùng lúc, tăng tốc độ lên hàng trăm lần.
- **Knowledge Distillation (Chưng cất tri thức):** "Nén" một mô hình Level 3 khổng lồ thành một phiên bản nhỏ gọn (Student model) để có thể chạy trên thiết bị di động mà vẫn giữ được 90% chất lượng.
- **Streaming Vocoder:** Kỹ thuật phát luồng (streaming) cho phép phát âm thanh ngay khi văn bản vẫn đang được xử lý, giúp người dùng không cảm thấy độ trễ.

### 3.2. Cải thiện tính tự nhiên & Kiểm soát (Expressiveness Control)
- **Prosody Modeling (Mô hình hóa ngữ điệu):** Tách biệt các yếu tố như cao độ (pitch), năng lượng (energy) và thời lượng (duration). Điều này giúp khắc phục nhược điểm "đều đều" của Level 1 và tăng tính biểu cảm cho Level 2.
- **Latent Space Manipulation:** Với Level 3, các nghiên cứu tập trung vào việc điều hướng không gian ẩn để kiểm soát cảm xúc cụ thể (ví dụ: bắt buộc giọng nói phải nghe "hào hứng" thay vì để mô hình tự quyết định).

### 3.3. Đảm bảo an toàn & đạo đức (Safety & Ethics)
Do khả năng sao chép giọng nói quá giống thật của Level 3, các pipeline hiện đại bắt buộc phải tích hợp module **Watermarking** (đóng dấu ngầm vào sóng âm) để phân biệt giọng AI và giọng người thật, ngăn chặn Deepfake.

---

## 4. Tổng kết so sánh

Bảng dưới đây tổng hợp các đặc tính kỹ thuật để so sánh tính phù hợp của từng phương pháp:

| Tiêu chí | Level 1: Rule-based / Concatenative | Level 2: Deep Learning (Fine-tuning) | Level 3: Generative AI (Zero-shot) |
|---|---|---|---|
| Công nghệ lõi | Unit Selection, HMM, DSP thuần túy | CNN, RNN, Transformer (cần train lại) | Large Audio Models, Diffusion (không cần train lại) |
| Dữ liệu đầu vào (User) | Không hỗ trợ người dùng tự tạo giọng | Cần 10 phút - 1 giờ (Thu âm sạch, chất lượng cao) | Cần 3 - 10 giây (Bất kỳ, chấp nhận nhiễu nhẹ) |
| Tính linh hoạt (Scalability) | Thấp. Cần chuyên gia can thiệp nếu đổi giọng. | Trung bình. Cần thời gian & tài nguyên để train mỗi giọng mới. | Rất cao. Tạo giọng mới ngay lập tức (Real-time). |
| Độ tự nhiên & Cảm xúc | Thấp. Ngữ điệu máy móc. | Cao. Tự nhiên, ổn định, nhưng khó thay đổi cảm xúc linh hoạt. | Rất cao. Giàu cảm xúc, có thể diễn xuất (khóc, cười, thở dài). |
| Độ ổn định | Tuyệt đối. Luôn đọc đúng những gì đã lập trình. | Cao. Ít khi gặp lỗi phát âm lạ. | Trung bình. Có thể gặp "ảo giác" (thêm/bớt từ) hoặc lỗi lạ. |
| Tài nguyên phần cứng | Rất thấp. Chạy tốt trên CPU, chip nhúng rẻ tiền. | Trung bình. Cần GPU để train, chạy infer có thể dùng CPU mạnh. | Rất cao. Bắt buộc GPU mạnh để vận hành mượt mà. |
| Ứng dụng phù hợp | Hệ thống thông báo công cộng, thiết bị IoT cấu hình thấp, ngôn ngữ hiếm. | Trợ lý ảo cá nhân (Siri, Alexa), Sách nói (Audiobook), CSKH tự động. | Sáng tạo nội dung (Youtube/Tiktok), Game NPC, Dubbing phim tự động. |


---

## 5. Khó khăn và giải pháp trong quá trình nghiên cứu

### Khó khăn
- **Khoảng cách kiến thức liên ngành**: TTS yêu cầu hiểu đồng thời DSP, NLP và Deep Learning, gây khó khăn cho một người mới tiếp cận như em.
- **Thuật ngữ mới và mô hình phức tạp**: Nhiều khái niệm như *mel-spectrogram, vocoder, diffusion model, speaker embedding, ...* chưa quen thuộc.

### Giải pháp
- **Tra cứu và đối chiếu bằng LLM**: Sử dụng Large Language Models như một công cụ hỗ trợ học thuật để giải thích nhanh thuật ngữ và so sánh các phương pháp.
- **Tiếp cận theo tầng (Level-based learning)**: Bắt đầu từ Level 1 để nắm nền tảng, sau đó tiến dần lên Level 2 và Level 3.

**Ví dụ prompt sử dụng LLM trong quá trình tìm hiểu**:

```
- Prompt:
Giải thích sự khác nhau giữa vocoder-based TTS và mô hình end-to-end neural TTS 1 cách đơn giản, dễ hình dung.

- Response (tóm tắt):
Vocoder-based TTS tách quá trình tổng hợp thành hai bước: tạo đặc trưng âm thanh và chuyển đặc trưng đó thành sóng âm,
trong khi end-to-end neural TTS học trực tiếp ánh xạ từ văn bản sang sóng âm trong một mô hình thống nhất.
```

---

## 6. Tài liệu tham khảo

1. Daniel Jurafsky and James H. Martin. **Speech and Language Processing**, 3rd edition, 2025.  
   Online manuscript: https://web.stanford.edu/~jurafsky/slp3/

2. Renqian Luo et al. **FastSpeech: Fast, Robust and Controllable Text to Speech**. *Microsoft Research*.  
   Project page: https://speechresearch.github.io/fastspeech/


---
