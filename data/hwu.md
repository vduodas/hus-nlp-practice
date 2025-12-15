# Mô tả bộ dữ liệu

## Tổng quan
Dataset này bao gồm các câu truy vấn của người dùng (**user utterances**) được gán với nhãn ý định (**intent categories**) tương ứng.  
Dataset được thiết kế để huấn luyện và đánh giá các mô hình **hiểu ngôn ngữ tự nhiên (NLU)** trong bối cảnh trợ lý giọng nói hoặc hệ thống hội thoại.

- **Tổng số nhãn ý định:** 66  
- **Chia tập dữ liệu:**
  - **Train:** 8,954 mẫu  
  - **Validation:** 1,076 mẫu  
  - **Test:** 1,076 mẫu  

Mỗi mẫu dữ liệu bao gồm:
- **Utterance** – câu văn bản đầu vào từ người dùng (ví dụ: `"wake me at daybreak"`)  
- **Category** – nhãn ý định tương ứng (ví dụ: `alarm_set`)  

---

## Các nhãn ý định (Categories)

Dataset bao phủ nhiều chức năng khác nhau của trợ lý ảo. Dưới đây là danh sách đầy đủ các nhãn ý định:

- **Báo thức (Alarms):**  
  `alarm_query`, `alarm_remove`, `alarm_set`

- **Điều khiển âm thanh (Audio Control):**  
  `audio_volume_down`, `audio_volume_mute`, `audio_volume_up`

- **Lịch (Calendar):**  
  `calendar_query`, `calendar_remove`, `calendar_set`

- **Nấu ăn (Cooking):**  
  `cooking_recipe`

- **Ngày & Giờ (Date & Time):**  
  `datetime_convert`, `datetime_query`

- **Email:**  
  `email_addcontact`, `email_query`, `email_querycontact`, `email_sendemail`

- **Chung (General):**  
  `general_affirm`, `general_commandstop`, `general_confirm`, `general_dontcare`,  
  `general_explain`, `general_joke`, `general_negate`, `general_praise`,  
  `general_quirky`, `general_repeat`

- **IoT (Thiết bị thông minh):**  
  - Dọn dẹp: `iot_cleaning`  
  - Cà phê: `iot_coffee`  
  - Đèn Hue: `iot_hue_lightchange`, `iot_hue_lightdim`, `iot_hue_lightoff`, `iot_hue_lighton`, `iot_hue_lightup`  
  - Wemo: `iot_wemo_off`, `iot_wemo_on`

- **Danh sách (Lists):**  
  `lists_createoradd`, `lists_query`, `lists_remove`

- **Âm nhạc (Music):**  
  `music_likeness`, `music_query`, `music_settings`

- **Tin tức (News):**  
  `news_query`

- **Phát nội dung (Playback):**  
  `play_audiobook`, `play_game`, `play_music`, `play_podcasts`, `play_radio`

- **Hỏi đáp (Q&A):**  
  `qa_currency`, `qa_definition`, `qa_factoid`, `qa_maths`, `qa_stock`

- **Gợi ý / Đề xuất (Recommendations):**  
  `recommendation_events`, `recommendation_locations`, `recommendation_movies`

- **Mạng xã hội (Social):**  
  `social_post`, `social_query`

- **Đặt đồ ăn (Takeaway):**  
  `takeaway_order`, `takeaway_query`

- **Giao thông (Transport):**  
  `transport_query`, `transport_taxi`, `transport_ticket`, `transport_traffic`

- **Thời tiết (Weather):**  
  `weather_query`

---

## Một vài bản ghi dữ liệu

Dưới đây là một số ví dụ câu truy vấn của người dùng và nhãn ý định tương ứng:

| Utterance                                                   | Category           |
|-------------------------------------------------------------|--------------------|
| "please let me know the alarm kept for tuesday's meeting"   | `alarm_query`      |
| "kickball is over i do not need the alarm for wednesday"    | `alarm_remove`     |
| "wake me at daybreak"                                       | `alarm_set`        |
| "lower volume to half"                                      | `audio_volume_down`|

---

## Chia tập dữ liệu (Data Splits)

Dataset đã được chia sẵn thành ba phần nhằm phục vụ cho quá trình huấn luyện và đánh giá mô hình:

| Split       | Samples | Mục đích                                      |
|-------------|---------|-----------------------------------------------|
| Train       | 8,954   | Dùng để huấn luyện mô hình                    |
| Validation  | 1,076   | Dùng để điều chỉnh siêu tham số và tránh overfitting |
| Test        | 1,076   | Dùng để đánh giá hiệu năng cuối cùng của mô hình |
