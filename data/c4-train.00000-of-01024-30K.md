**C4 – `c4-train.00000-of-01024-30K.json`** là một tập con của *Colossal Clean Crawled Corpus (C4)*, bao gồm khoảng **30.000 bản ghi văn bản tiếng Anh** được thu thập từ web và đã qua bước làm sạch.  

Mỗi bản ghi được lưu dưới dạng **JSON**, gồm ba trường chính:  
- **`text`**: nội dung văn bản liên tục, gồm nhiều câu;  
- **`timestamp`**: thời điểm thu thập hoặc xuất bản (ISO 8601);  
- **`url`**: địa chỉ web nguồn của văn bản.  

Dữ liệu phản ánh ngôn ngữ tự nhiên trong các bối cảnh thực tế như tin tức, blog và thông báo, phù hợp cho các thí nghiệm huấn luyện **word embedding**.


Một vài bản ghi mẫu:

```json
{"text":"Beginners BBQ Class Taking Place in Missoula!\nDo you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers. He will be teaching a beginner level class for everyone who wants to get better with their culinary skills.\nHe will teach you everything you need to know to compete in a KCBS BBQ competition, including techniques, recipes, timelines, meat selection and trimming, plus smoker and fire information.\nThe cost to be in the class is $35 per person, and for spectators it is free. Included in the cost will be either a t-shirt or apron and you will be tasting samples of each meat that is prepared.","timestamp":"2019-04-25T12:57:54Z","url":"https://klyq.com/beginners-bbq-class-taking-place-in-missoula/"}

{"text":"Discussion in 'Mac OS X Lion (10.7)' started by axboi87, Jan 20, 2012.\nI've got a 500gb internal drive and a 240gb SSD.\nWhen trying to restore using disk utility i'm given the error \"Not enough space on disk ____ to restore\"\nBut I shouldn't have to do that!!!\nAny ideas or workarounds before resorting to the above?\nUse Carbon Copy Cloner to copy one drive to the other. I've done this several times going from larger HDD to smaller SSD and I wound up with a bootable SSD drive. One step you have to remember not to skip is to use Disk Utility to partition the SSD as GUID partition scheme HFS+ before doing the clone. If it came Apple Partition Scheme, even if you let CCC do the clone, the resulting drive won't be bootable. CCC usually works in \"file mode\" and it can easily copy a larger drive (that's mostly empty) onto a smaller drive. If you tell CCC to clone a drive you did NOT boot from, it can work in block copy mode where the destination drive must be the same size or larger than the drive you are cloning from (if I recall).\nI've actually done this somehow on Disk Utility several times (booting from a different drive (or even the dvd) so not running disk utility from the drive your cloning) and had it work just fine from larger to smaller bootable clone. Definitely format the drive cloning to first, as bootable Apple etc..\nThanks for pointing this out. My only experience using DU to go larger to smaller was when I was trying to make a Lion install stick and I was unable to restore InstallESD.dmg to a 4 GB USB stick but of course the reason that wouldn't fit is there was slightly more than 4 GB of data.","timestamp":"2019-04-21T10:07:13Z","url":"https://forums.macrumors.com/threads/restore-from-larger-disk-to-smaller-disk.1311329/"}

{"text":"Foil plaid lycra and spandex shortall with metallic slinky insets. Attached metallic elastic belt with O-ring. Headband included. Great hip hop or jazz dance costume. Made in the USA.","timestamp":"2019-04-25T10:40:23Z","url":"https://awishcometrue.com/Catalogs/Clearance/Tweens/V1960-Find-A-Way"}
```