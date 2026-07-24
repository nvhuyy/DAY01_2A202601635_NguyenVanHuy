# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng placeholder bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Qua bốn lần chạy với các giá trị temperature khác nhau, có thể thấy khi temperature = 0.0, câu trả lời ổn định, gần như giống nhau ở mỗi lần gọi và chỉ tập trung vào những thông tin phổ biến. Khi tăng lên 0.5 và 1.0, câu trả lời trở nên đa dạng hơn, cách diễn đạt phong phú hơn và có thể đưa thêm các ví dụ hoặc góc nhìn khác. Đến 1.5, mô hình sáng tạo hơn nhưng cũng có xu hướng dài dòng, ít nhất quán hơn và đôi khi đưa vào những chi tiết kém chính xác hoặc ít liên quan. Điều này cho thấy temperature càng cao thì tính ngẫu nhiên và sáng tạo của mô hình càng tăng, trong khi temperature thấp ưu tiên sự ổn định và khả năng lặp lại kết quả.


### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Đối với chatbot hỗ trợ khách hàng, tôi sẽ chọn temperature khoảng 0.2–0.3. Mức này giúp mô hình tạo ra các câu trả lời ổn định, nhất quán và chính xác, hạn chế việc thay đổi cách diễn đạt hoặc tự bổ sung những thông tin không cần thiết. Trong lĩnh vực chăm sóc khách hàng, ưu tiên hàng đầu là cung cấp thông tin đúng và đáng tin cậy hơn là sự sáng tạo, vì vậy một giá trị temperature thấp sẽ phù hợp hơn so với các tác vụ như viết nội dung hay kể chuyện.


### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Với kịch bản trên, hệ thống sẽ tạo ra khoảng 10,5 triệu output token mỗi ngày. Theo bảng giá API, GPT-4o đắt hơn khoảng 16,7 lần. GPT-4o xứng đáng với chi phí trong các tác vụ đòi hỏi khả năng suy luận và chất lượng câu trả lời cao, chẳng hạn như trợ lý pháp lý, phân tích tài liệu chuyên sâu hoặc hỗ trợ lập trình phức tạp. Ngược lại, GPT-4o-mini phù hợp với các ứng dụng có lưu lượng lớn như chatbot chăm sóc khách hàng, hỏi đáp thông thường hoặc tóm tắt văn bản, nơi yêu cầu phản hồi nhanh và tối ưu chi phí quan trọng hơn việc đạt chất lượng cao nhất.


---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Khi sử dụng hai system prompt khác nhau với cùng một câu hỏi, mô hình tạo ra hai câu trả lời có phong cách rất khác biệt. Với vai trò giáo viên tiểu học, câu trả lời ngắn gọn, dùng từ ngữ đơn giản, gần gũi và thường kèm các ví dụ dễ hình dung để trẻ em có thể hiểu. Ngược lại, với vai trò chuyên gia tài chính, câu trả lời dài hơn, sử dụng nhiều thuật ngữ kỹ thuật như sổ cái phân tán, cơ chế đồng thuận và phân tích sâu về nguyên lý hoạt động. Điều này cho thấy system prompt đóng vai trò định hướng hành vi của mô hình, quyết định cách diễn đạt, mức độ chi tiết, đối tượng hướng đến và phong cách trả lời, dù nội dung cốt lõi của câu hỏi vẫn không thay đổi.


### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Kết quả cho thấy hai giá trị chênh lệch khoảng 10–20%, trong đó số token thực tế thường cao hơn giá trị ước lượng. Nguyên nhân là tiếng Việt sử dụng nhiều từ đa âm tiết, dấu thanh và ký tự Unicode, khiến bộ mã hóa của mô hình phải tách thành nhiều token hơn. Trong khi đó, công thức số từ / 0.75 chỉ là một quy tắc kinh nghiệm, được xây dựng chủ yếu dựa trên tiếng Anh nên không phản ánh chính xác cách tokenizer xử lý văn bản tiếng Việt.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi thời gian sinh kết quả dài nhưng người dùng muốn thấy phản hồi ngay, non-streaming lại phù hợp hơn khi kết quả trả về rất nhỏ, tính toán nhanh, cần toàn bộ kết quả mới xử lý tiếp.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> So với delay cố định, exponential backoff giúp giãn các request, server có thêm thời gian phục hồi. Nếu hàng nghìn client cùng retry với delay cố định giống nhau, chúng sẽ liên tục gửi request đồng loạt, làm server khó phục hồi và có thể kéo dài tình trạng quá tải.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System prompt như sau: "Bạn là một trợ lý AI chuyên về lập trình và trí tuệ nhân tạo. - Luôn trả lời bằng tiếng Việt, trừ khi người dùng yêu cầu sử dụng ngôn ngữ khác. - Trả lời chính xác, ngắn gọn và có cấu trúc rõ ràng. - Khi phù hợp, hãy đưa ra ví dụ hoặc đoạn mã minh họa để người dùng dễ hiểu.". Yêu cầu "Luôn trả lời bằng tiếng Việt" vì assistant này hướng đến người dùng Việt Nam nên việc mặc định sử dụng tiếng Việt giúp giao tiếp tự nhiên và dễ hiểu hơn, đồng thời, quy định "trừ khi người dùng yêu cầu" vẫn đảm bảo tính linh hoạt khi cần trao đổi bằng tiếng Anh hoặc ngôn ngữ khác; Yêu cầu "Trả lời chính xác, ngắn gọn" giúp tránh lan man, người đọc dễ theo dõi và tìm được thông tin mình cần.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất của trợ lý hiện tại là chỉ lưu lịch sử hội thoại tối đa 3 lượt gần nhất (6 message), nên dễ mất ngữ cảnh khi cuộc trò chuyện kéo dài. Nếu người dùng nhắc lại thông tin đã đề cập từ nhiều lượt trước, trợ lý sẽ không còn nhớ và có thể đưa ra câu trả lời thiếu chính xác hoặc yêu cầu người dùng cung cấp lại thông tin. Một cải thiện phù hợp là bổ sung bộ nhớ dài hạn (long-term memory). Cách triển khai là lưu toàn bộ lịch sử hội thoại vào cơ sở dữ liệu hoặc tệp lưu trữ, đồng thời sử dụng mô hình embedding kết hợp vector database để tìm kiếm những đoạn hội thoại liên quan mỗi khi người dùng đặt câu hỏi mới. Các đoạn được truy xuất sẽ được thêm vào prompt cùng với lịch sử gần nhất trước khi gửi đến mô hình, giúp trợ lý duy trì ngữ cảnh tốt hơn mà vẫn không vượt quá giới hạn số lượng token của API.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
