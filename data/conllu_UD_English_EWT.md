# Mô tả dữ liệu định dạng `.conllu`

Dataset được lưu trữ theo định dạng **CoNLL-U**, là định dạng chuẩn thường được sử dụng trong các bài toán **Xử lý Ngôn ngữ Tự nhiên (NLP)**, đặc biệt là **phân tích cú pháp phụ thuộc (Dependency Parsing)** và **gán nhãn từ loại (Part-of-Speech Tagging)**.

Mỗi file `.conllu` bao gồm nhiều câu, trong đó **mỗi câu được ngăn cách bởi một dòng trống** và có thể kèm theo các **dòng chú thích (comment lines)** bắt đầu bằng ký tự `#`.

---

## Cấu trúc câu (Sentence Structure)

Mỗi câu trong file `.conllu` bao gồm hai phần chính: **metadata** và **các token**.

---

## 1. Dòng chú thích (Metadata)

Các dòng bắt đầu bằng ký tự `#` cung cấp thông tin bổ sung cho câu, bao gồm:

- `# sent_id` : định danh duy nhất của câu  
- `# newdoc id` : định danh tài liệu  
- `# newpar id` : định danh đoạn văn  
- `# text` : câu gốc ở dạng văn bản thuần  

Ví dụ:

```text
# sent_id = weblog-blogspot.com_nominations_20041117172713_ENG_20041117_172713-0001
# text = From the AP comes this story :
```

---

## 2. Các dòng token

Mỗi dòng token tương ứng với **một từ hoặc dấu câu** trong câu.  
Các cột được phân tách bằng ký tự **tab (`\t`)** và tuân theo chuẩn **10 cột của CoNLL-U**.

| Cột | Tên cột | Mô tả |
|----|--------|------|
| 1 | ID | Chỉ số của token trong câu (bắt đầu từ 1) |
| 2 | FORM | Hình thức bề mặt của token |
| 3 | LEMMA | Dạng gốc (lemma) của token |
| 4 | UPOS | Nhãn từ loại phổ quát (Universal POS) |
| 5 | XPOS | Nhãn từ loại chi tiết theo ngôn ngữ |
| 6 | FEATS | Đặc trưng hình thái học |
| 7 | HEAD | Chỉ số token cha trong cây phụ thuộc |
| 8 | DEPREL | Quan hệ phụ thuộc cú pháp |
| 9 | DEPS | Quan hệ phụ thuộc nâng cao |
| 10 | MISC | Thông tin bổ sung |

Ví dụ một dòng token:

```text
4	comes	come	VERB	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	root	0:root	_
```

Trong ví dụ trên:
- Token **"comes"** là một **động từ (VERB)**
- Là **gốc của câu (`root`)**
- Có các đặc trưng hình thái học như thì hiện tại, ngôi thứ ba, số ít

---

## Quan hệ phụ thuộc (Dependency Relations)

Hai cột `HEAD` và `DEPREL` mô tả **cấu trúc cú pháp phụ thuộc** của câu:

- `HEAD = 0` biểu thị token là **gốc của câu**
- Các token khác trỏ về token cha thông qua chỉ số `HEAD`
- `DEPREL` mô tả loại quan hệ cú pháp, ví dụ:
  - `nsubj` (chủ ngữ)
  - `obj` (tân ngữ)
  - `obl` (bổ ngữ trạng ngữ)
  - `case`, `det`, `amod`
  - `punct` (dấu câu)

Ví dụ:

```text
6	story	story	NOUN	NN	Number=Sing	4	nsubj	4:nsubj	_
```

→ Token **"story"** là **chủ ngữ (`nsubj`)** của động từ **"comes"** (token có ID = 4).

---

## Đặc điểm dữ liệu

- Dữ liệu được **gán nhãn đầy đủ** cho:
  - Từ loại (POS)
  - Đặc trưng hình thái học
  - Quan hệ cú pháp phụ thuộc
- Phù hợp cho các bài toán:
  - **Part-of-Speech Tagging**
  - **Dependency Parsing**
  - **Joint Learning (POS + Parsing)**
- Dữ liệu mẫu thuộc **ngữ liệu tiếng Anh**, chủ yếu từ **blog và văn bản báo chí**, phản ánh ngôn ngữ tự nhiên trong thực tế.

---

## Dữ liệu mẫu

```conllu
# newdoc id = weblog-blogspot.com_nominations_20041117172713_ENG_20041117_172713
# sent_id = weblog-blogspot.com_nominations_20041117172713_ENG_20041117_172713-0001
# newpar id = weblog-blogspot.com_nominations_20041117172713_ENG_20041117_172713-p0001
# text = From the AP comes this story :
1	From	from	ADP	IN	_	3	case	3:case	_
2	the	the	DET	DT	Definite=Def|PronType=Art	3	det	3:det	_
3	AP	AP	PROPN	NNP	Number=Sing	4	obl	4:obl:from	_
4	comes	come	VERB	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	root	0:root	_
5	this	this	DET	DT	Number=Sing|PronType=Dem	6	det	6:det	_
6	story	story	NOUN	NN	Number=Sing	4	nsubj	4:nsubj	_
7	:	:	PUNCT	:	_	4	punct	4:punct	_

# sent_id = weblog-blogspot.com_nominations_20041117172713_ENG_20041117_172713-0002
# newpar id = weblog-blogspot.com_nominations_20041117172713_ENG_20041117_172713-p0002
# text = President Bush on Tuesday nominated two individuals to replace retiring jurists on federal courts in the Washington area.
1	President	President	PROPN	NNP	Number=Sing	5	nsubj	5:nsubj	_
2	Bush	Bush	PROPN	NNP	Number=Sing	1	flat	1:flat	_
3	on	on	ADP	IN	_	4	case	4:case	_
4	Tuesday	Tuesday	PROPN	NNP	Number=Sing	5	obl	5:obl:on	_
5	nominated	nominate	VERB	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	0	root	0:root	_
6	two	two	NUM	CD	NumForm=Word|NumType=Card	7	nummod	7:nummod	_
7	individuals	individual	NOUN	NNS	Number=Plur	5	obj	5:obj	_
8	to	to	PART	TO	_	9	mark	9:mark	_
9	replace	replace	VERB	VB	VerbForm=Inf	5	advcl	5:advcl:to	_
10	retiring	retire	VERB	VBG	VerbForm=Ger	11	amod	11:amod	_
11	jurists	jurist	NOUN	NNS	Number=Plur	9	obj	9:obj	_
12	on	on	ADP	IN	_	14	case	14:case	_
13	federal	federal	ADJ	JJ	Degree=Pos	14	amod	14:amod	_
14	courts	court	NOUN	NNS	Number=Plur	11	nmod	11:nmod:on	_
15	in	in	ADP	IN	_	18	case	18:case	_
16	the	the	DET	DT	Definite=Def|PronType=Art	18	det	18:det	_
17	Washington	Washington	PROPN	NNP	Number=Sing	18	compound	18:compound	_
18	area	area	NOUN	NN	Number=Sing	14	nmod	14:nmod:in	SpaceAfter=No
19	.	.	PUNCT	.	_	5	punct	5:punct	_
```
