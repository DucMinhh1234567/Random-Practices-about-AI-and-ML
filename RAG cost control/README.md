# RAG Cost Control

Đại khái về vấn đề và ý tưởng là:
- RAG được thiết kế để giải quyết vấn đề về truy xuất và chưa bao giờ được thiết kế để giải quyết vấn đề về chi phí.
- Không phải chỉ trích rằng RAG không tốt mà đây là 2 layer khác nhau.
- Trong môi trường development thì thường chưa quan tâm nhiều đến chi phí mà đến hiệu suất
- Tuy nhiên trong môi trường production thì 2 vấn đề này bắt đầu xung đột.

## Các vấn đề chính:

### 1. Context window over fetching

Trong hầu hết các pipeline RAG, thường sẽ truy xuất top 10 chunks để cho an toàn, tuy nhiên từ đây bắt đầu nảy sinh vấn đề
- Thông thường chỉ có khoảng 2 - 3 chunks là chứa thông tin cần thiết, không phải tất cả đều vậy nhưng phần lớn là thế
- Vì vậy các chunks khác coi như bỏ đi và không để làm gì

**Lưu ý:** Đây vẫn chỉ là lý thuyết.


### 2. No Caching Layer:

Trong trường hợp 2 user hỏi cùng 1 câu hỏi hoặc 2 câu hỏi có ý giống nhau chỉ cách vài phút, hệ thống vẫn tạo ra các embedding giống nhau, truy xuất các chunks giống nhau và trả về kết quả giống nhau. -> Từ đó phải trả tiền 2 lần.

- Không có semantic memory giữa các request
- Mọi query đều được xử lý như chưa từng được thấy trước đây

### 3. No Model Routing:

Một số pipeline thường mặc định duy nhất một model cho mọi query, mặc cho độ phức tạp của query có cao hay không
Một câu hỏi đơn giản như: "RAG là gì?" không cần thiết phải đưa vào một model cao hay cần reasoning nhiều bước để trả lời. Chỉ cần một model nhanh, rẻ để trả lời trong tgian ngắn.






link tham khảo: https://towardsdatascience.com/rag-is-burning-money-i-built-a-cost-control-layer-to-fix-it/