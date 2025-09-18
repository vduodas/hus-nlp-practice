Trong bài lab02 này, em triển khai:  
- **Interface `Vectorizer`**: `CountVectorizer` (chuyển văn bản thành vector tần suất từ).  
- Thử nghiệm trên một số câu ví dụ và trên **bộ dữ liệu UD English-EWT**.  
---

## 1. Các bước triển khai
### 1.1. Interface `Vectorizer`  
- Được định nghĩa trong `core/interfaces.py` dưới dạng abstract class.  
- Bắt buộc mọi vectorizer phải cài đặt 03 phương thức:  
```
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

### 1.2. `CountVectorizer`
**Nguyên tắc:**  

- **Khởi tạo (`__init__`)**  
  - Nhận vào một tokenizer (đã được implement ở lab01, xem tại [README-lab-01](lab01_16_09/README.md); mặc định dùng `SimpleTokenizer`).  
  - Tạo cấu trúc từ vựng `_vocabulary` để lưu ánh xạ **token → index**.  

- **Học từ vựng (`fit`)**  
  - Tokenize toàn bộ corpus (mỗi document thành danh sách token).  
  - Gộp tất cả tokens thành một tập hợp duy nhất (loại bỏ trùng lặp).  
  - Sắp xếp token theo thứ tự alphabet.  
  - Gán index cho từng token trong vocabulary.  

- **Biến đổi văn bản (`transform`)**  
  - Với mỗi document:  
    - Khởi tạo vector có kích thước bằng số lượng từ vựng.  
    - Tokenize document, đếm số lần xuất hiện của từng token.  
    - Ghi kết quả vào vector tương ứng.  

- **Kết hợp (`fit_transform`)**  
  - Gọi `fit` để học vocabulary từ corpus.  
  - Gọi `transform` để sinh ma trận document-term matrix.  

## 2. Thử nghiệm 
Sau khi pull sourcecode về, chạy tại cwd bằng lệnh 
```
python -m lab02_16_09.main
```
Kết quả thu được sẽ được hiển thị trực tiếp ra console.
### 2.1. Test trên câu ví dụ
Với corpus:
```
corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]
```
- **Bước 1: Fit và Transform**

    - `CountVectorizer` sử dụng `RegexTokenizer` để tách token.

    - Từ toàn bộ corpus, vectorizer học được vocabulary (tập từ duy nhất).

    - Sau đó, corpus được chuyển thành Document-Term Matrix.

- **Kết quả Vocabulary học được (chưa qua lọc bỏ dấu câu):**
```
{'.': 0, 'a': 1, 'ai': 2, 'i': 3, 'is': 4, 'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9}
```
- **Document-Term Matrix thu được:**
```
[
 [0, 1, 0, 1, 1, 0, 0, 0],   # "I love NLP."
 [0, 1, 0, 1, 0, 0, 1, 0],   # "I love programming."
 [1, 0, 1, 0, 1, 1, 0, 1]    # "NLP is a subfield of AI."
]
```

- Giải thích:

    - Mỗi hàng biểu diễn một câu (document).

    - Mỗi cột tương ứng với một token trong vocabulary.

    - Giá trị là số lần token xuất hiện trong document đó.

### 2.2. Test trên dataset UD English EWT**
- **Bước 1: Chuẩn bị dữ liệu**  
    - Đọc file: `en_ewt-ud-train.txt`.  
    - Trích 3 đoạn văn bản liên tiếp, mỗi đoạn dài 500 ký tự, coi như 3 document mẫu.  

- **Bước 2: Vector hóa văn bản**
    - Gọi fit_transform(sample_texts) để:
    - Tokenize mỗi đoạn.
    - Học vocabulary.
    - Sinh Document-Term Matrix.

### 2.3. Kết quả mẫu
```
Learned Vocabulary: {'.': 0, 'a': 1, 'ai': 2, 'i': 3, 'is': 4, 'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9}
Document-Term Matrix: [[1, 0, 0, 1, 0, 1, 1, 0, 0, 0], [1, 0, 0, 1, 0, 1, 0, 0, 1, 0], [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]]

--- Vectorizing Sample Text from UD_English-EWT ---
[Document 1] First 500 characters: Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the
mosque in the town of Qaim, near the Syrian border. [This killing of a respected
cleric will be causing us trouble for years to come.] DPA: Iraqi authorities
announced that they had busted up 3 terrorist cells operating in Baghdad. Two of
them were being run by 2 officials of the Ministry of the Interior! The MoI in
Iraq is equivalent to the US FBI, so this would be like having J. Edgar Hoover
unwittingly employ at a h

[Document 2] First 500 characters: igh level members of the Weathermen bombers back in the
1960s. The third was being run by the head of an investment firm. You wonder if
he was manipulating the market with his bombing targets. The cells were
operating in the Ghazaliyah and al-Jihad districts of the capital. Although the
announcement was probably made to show progress in identifying and breaking up
terror cells, I don't find the news that the Baathists continue to penetrate the
Iraqi government very hopeful. It reminds me too muc

[Document 3] First 500 characters: h of the ARVN officers who
were secretly working for the other side in Vietnam. Al-Zaman : Guerrillas
killed a member of the Kurdistan Democratic Party after kidnapping him in Mosul.
The police commander of Ninevah Province announced that bombings had declined 80
percent in Mosul, whereas there had been a big jump in the number of
kidnappings. On Wednesday guerrillas had kidnapped a cosmetic surgeon and his
wife while they were on their way home. In Suwayrah, Kut Province, two car bombs
were dis

Learned Vocabulary: {'!': 0, "'": 1, ',': 2, '-': 3, '.': 4, '1960s': 5, '2': 6, '3': 7, '80': 8, ':': 9, '[': 10, ']': 11, 'a': 12, 'abdullah': 13, 'after': 14, 'al': 15, 'although': 16, 'american': 17, 'an': 18, 'and': 19, 'ani': 20, 'announced': 21, 'announcement': 22, 'arvn': 23, 'at': 24, 'authorities': 25, 'baathists': 26, 'back': 27, 'baghdad': 28, 'be': 29, 'been': 30, 'being': 31, 'big': 32, 'bombers': 33, 'bombing': 34, 'bombings': 35, 'bombs': 36, 'border': 37, 'breaking': 38, 'busted': 39, 'by': 40, 'capital': 41, 'car': 42, 'causing': 43, 'cells': 44, 'cleric': 45, 'come': 46, 'commander': 47, 'continue': 48, 'cosmetic': 49, 'declined': 50, 'democratic': 51, 'dis': 52, 'districts': 53, 'don': 54, 'dpa': 55, 'edgar': 56, 'employ': 57, 'equivalent': 58, 'fbi': 59, 'find': 60, 'firm': 61, 'for': 62, 'forces': 63, 'ghazaliyah': 64, 'government': 65, 'guerrillas': 66, 'h': 67, 'had': 68, 'having': 69, 'he': 70, 'head': 71, 'him': 72, 'his': 73, 'home': 74, 'hoover': 75, 'hopeful': 76, 'i': 77, 'identifying': 78, 'if': 79, 'igh': 80, 'in': 81, 'interior': 82, 'investment': 83, 'iraq': 84, 'iraqi': 85, 'is': 86, 'it': 87, 'j': 88, 'jihad': 89, 'jump': 90, 'kidnapped': 91, 'kidnapping': 92, 'kidnappings': 93, 'killed': 94, 'killing': 95, 'kurdistan': 96, 'kut': 97, 'level': 98, 'like': 99, 'made': 100, 'manipulating': 101, 'market': 102, 'me': 103, 'member': 104, 'members': 105, 'ministry': 106, 'moi': 107, 'mosque': 108, 'mosul': 109, 'muc': 110, 'near': 111, 'news': 112, 'ninevah': 113, 'number': 114, 'of': 115, 'officers': 116, 'officials': 117, 'on': 118, 'operating': 119, 'other': 120, 'party': 121, 'penetrate': 122, 'percent': 123, 'police': 124, 'preacher': 125, 'probably': 126, 'progress': 127, 'province': 128, 'qaim': 129, 'reminds': 130, 'respected': 131, 'run': 132, 'secretly': 133, 'shaikh': 134, 'show': 135, 'side': 136, 'so': 137, 'surgeon': 138, 'suwayrah': 139, 'syrian': 140, 't': 141, 'targets': 142, 'terror': 143, 'terrorist': 144, 'that': 145, 'the': 146, 'their': 147, 'them': 148, 'there': 149, 'they': 150, 'third': 151, 'this': 152, 'to': 153, 'too': 154, 'town': 155, 'trouble': 156, 'two': 157, 'unwittingly': 158, 'up': 159, 'us': 160, 'very': 161, 'vietnam': 162, 'was': 163, 'way': 164, 'weathermen': 165, 'wednesday': 166, 'were': 167, 'whereas': 168, 'while': 169, 'who': 170, 'wife': 171, 'will': 172, 'with': 173, 'wonder': 174, 'working': 175, 'would': 176, 'years': 177, 'you': 178, 'zaman': 179}
Document-term matrix: [[1, 0, 3, 2, 4, 0, 1, 1, 0, 2, 1, 1, 2, 1, 0, 2, 0, 1, 0, 0, 1, 1, 0, 0, 2, 1, 0, 0, 1, 2, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 5, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 8, 0, 1, 0, 1, 0, 2, 2, 0, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1], [0, 1, 1, 1, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 3, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 12, 0, 0, 0, 0, 1, 0, 2, 1, 0, 0, 0, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0], [0, 0, 3, 1, 4, 0, 0, 0, 1, 1, 0, 0, 3, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 3, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 4, 1, 0, 2, 0, 1, 1, 0, 1, 1, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 5, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 3, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1]]
```