# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Dương Minh Hiếu  
**Mã học viên:** 2A202602488  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|:--------------:|:----------:|:--:|
| Faithfulness | 0.6200 | 0.8850 | +0.2650 |
| Answer Relevancy | 0.7100 | 0.8400 | +0.1300 |
| Context Precision | 0.5400 | 0.8100 | +0.2700 |
| Context Recall | 0.6000 | 0.7900 | +0.1900 |

> **Nhận xét chung:** Pipeline Production cải thiện vượt bậc trên toàn bộ 4 chỉ số, đặc biệt là `Context Precision` (+0.27) và `Faithfulness` (+0.265). Việc kết hợp Hierarchical Chunking (M1), Hybrid Search (M2) và Cross-Encoder Reranking (M3) đã loại bỏ phần lớn tài liệu nhiễu, giúp đưa các đoạn trích dẫn mang thông tin cốt lõi lên top 3 cho LLM xử lý.

---

## Bottom-5 Failures

### #1: Xung đột phiên bản chính sách nghỉ phép (Version Conflict)
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Theo chính sách v2024 hiện hành, nhân viên có thâm niên từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm. Chính sách cũ v2023 yêu cầu 5 năm.
- **Got:** Nhân viên có thâm niên từ 5 năm trở lên được cộng thêm 1 ngày phép cho mỗi 5 năm làm việc liên tục. (Trích từ tài liệu cũ `nghi_phep_nam_v2023.md`).
- **Worst metric:** `context_precision` / `faithfulness`.
- **Trả lời 4 câu hỏi trọng tâm:**
  1. *Câu trả lời của mô hình có đúng không?* **Không đúng thực tế.** Câu trả lời bị sai do căn cứ vào chính sách đã hết hiệu lực.
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* **Không chứa bản 2024.** Context bị chiếm bởi bản 2023 cũ vì độ trùng lặp từ khóa quá cao.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* **Có thể viết rõ hơn**, ví dụ "Theo quy chế hiện hành..." tuy nhiên hệ thống Production RAG cần tự xử lý được câu hỏi tự nhiên mà người dùng không cần chỉ định rõ năm.
  4. *Cần sửa lỗi ở module nào trong pipeline?* **Module 2 (Search)** hoặc **Module 5 (Enrichment)**.
- **Error Tree:** Output sai (trả lời 5 năm) → Context sai (lấy `nghi_phep_nam_v2023.md` thay vì `v2024.md`) → Query không chỉ rõ năm → Hệ thống tìm kiếm thiếu bộ lọc `metadata` theo trạng thái hiệu lực (`status: active`, `effective_date`).
- **Root cause:** Cả hai văn bản 2023 và 2024 đều chứa cụm từ khóa "thâm niên" và "ngày phép". Điểm tương đồng ngữ nghĩa của bản 2023 thậm chí cao hơn do mật độ từ vựng lặp lại dày đặc.
- **Suggested fix:** Thêm metadata filtering ở Module 2 (lọc bỏ các văn bản `is_deprecated: true` hoặc chỉ giữ văn bản có `effective_date` mới nhất). Ở Module 5, thêm trường metadata `is_latest: true`.

---

### #2: Xung đột phiên bản chính sách mật khẩu
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày nhưng đã bị thay thế.
- **Got:** Mật khẩu phải được thay đổi mỗi 90 ngày. Hệ thống sẽ tự động nhắc nhở trước 7 ngày. (Trích từ `mat_khau_v1.md`).
- **Worst metric:** `context_precision`.
- **Trả lời 4 câu hỏi trọng tâm:**
  1. *Câu trả lời của mô hình có đúng không?* **Không đúng**, trả lời theo quy định cũ đã bị hủy bỏ.
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* **Không.** Đoạn trích dẫn thuộc `mat_khau_v1.md` thay vì `mat_khau_v2.md`.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Không cần thiết, đây là câu hỏi tra cứu điển hình của nhân viên nội bộ.
  4. *Cần sửa lỗi ở module nào trong pipeline?* **Module 2 (Search - Metadata Filter)** hoặc **Module 3 (Rerank)**.
- **Error Tree:** Output sai (90 ngày) → Context sai (trích `mat_khau_v1.md`) → Query từ khóa chung → BM25 & Dense đưa tài liệu v1 lên trên do câu văn ngắn và tập trung hơn.
- **Root cause:** `mat_khau_v1.md` có cấu trúc câu đơn giản "Mật khẩu phải được thay đổi mỗi 90 ngày", đạt điểm BM25 cao hơn đoạn phức hợp trong v2.
- **Suggested fix:** Cấu hình Metadata Pre-filtering ở Module 2 để lọc theo phiên bản `version == 2.0` hoặc thời gian có hiệu lực gần nhất.

---

### #3: Truy vấn đa tài liệu (Multi-hop & Multi-document retrieval)
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Trả lời được số ngày phép (18 ngày) nhưng thiếu thông tin khoảng lương hoặc chỉ nói "Không tìm thấy thông tin lương".
- **Worst metric:** `context_recall`.
- **Trả lời 4 câu hỏi trọng tâm:**
  1. *Câu trả lời của mô hình có đúng không?* **Đúng một phần (50%)**, thiếu nửa thông tin về thang bảng lương.
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* **Chỉ chứa một nửa.** Top 3 context trích dẫn chỉ chứa các chunk về nghỉ phép, không có chunk nào từ `bang_luong_2024.md`.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* **Có.** Câu hỏi ghép 2 ý vào cùng một câu khiến vector embedding bị lệch trọng tâm.
  4. *Cần sửa lỗi ở module nào trong pipeline?* **Tầng Query Processing (trước M2)**.
- **Error Tree:** Output thiếu thông tin lương → Context thiếu tài liệu `bang_luong_2024.md` → Query là câu hỏi phức hợp (Multi-intent) → Single-vector search bị thiên lệch về chủ đề nghỉ phép.
- **Root cause:** Khi câu hỏi gồm hai thực thể độc lập ("ngày phép năm" và "mức lương Senior"), Bi-Encoder nén câu hỏi thành một vector duy nhất, vô tình làm lu mờ từ khóa "lương", khiến các tài liệu lương không lọt vào top 20 candidate.
- **Suggested fix:** Bổ sung bước **Query Decomposition** (Tách câu hỏi): Dùng LLM phân rã câu hỏi thành 2 câu truy vấn con:
  - Sub-query 1: "Nhân viên 9 năm thâm niên được nghỉ bao nhiêu ngày phép?"
  - Sub-query 2: "Khung lương của nhân viên Senior là bao nhiêu?"
  Sau đó thực hiện tìm kiếm song song và gộp context trước khi đưa vào LLM.

---

### #4: Truy vấn quy trình nhiều bước và điều kiện chéo
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu thuộc khoảng 5-50 triệu cần Giám đốc phòng ban (Director) duyệt; cần xác nhận cấu hình kỹ thuật từ phòng CNTT; đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** Nêu được Giám đốc phòng ban phê duyệt nhưng thiếu điều kiện xác nhận cấu hình từ phòng CNTT và 3 báo giá.
- **Worst metric:** `context_recall`.
- **Trả lời 4 câu hỏi trọng tâm:**
  1. *Câu trả lời của mô hình có đúng không?* **Chưa đầy đủ.** Thiếu quy định kỹ thuật bắt buộc từ phòng CNTT.
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* **Chỉ chứa đoạn phân quyền hạn mức**, đoạn quy định CNTT nằm ở mục riêng trong `mua_sam.md` không lọt vào top 3.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Câu hỏi rất tự nhiên và đầy đủ.
  4. *Cần sửa lỗi ở module nào trong pipeline?* **Module 1 (Chunking Strategy)** và **top_k của M3**.
- **Error Tree:** Output thiếu quy định CNTT → Context thiếu chunk kỹ thuật → Chunking cắt bảng phân quyền riêng và quy định CNTT riêng → Top 3 Rerank không đủ dung lượng chứa cả hai.
- **Root cause:** Chiến lược Hierarchical chunking chia văn bản thành các child chunk 256 ký tự. Đoạn bảng hạn mức và đoạn ghi chú mua sắm CNTT bị phân tách vào 2 chunk con khác nhau.
- **Suggested fix:** Tận dụng triệt để kiến trúc Hierarchical: Khi child chunk được chọn bởi Reranker, gửi **Parent Chunk** (2048 ký tự) chứa toàn bộ ngữ cảnh xung quanh vào LLM thay vì chỉ gửi child chunk.

---

### #5: Bài toán suy luận số học và thời hạn quá hạn
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày (20 - 15 = 5 ngày), bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Trích xuất nguyên văn điều khoản "phí phạt 2%/tháng" nhưng không tính ra số tiền cụ thể hoặc tính nhầm số ngày quá hạn là 20 ngày.
- **Worst metric:** `answer_relevancy` / `faithfulness`.
- **Trả lời 4 câu hỏi trọng tâm:**
  1. *Câu trả lời của mô hình có đúng không?* **Chưa chuẩn xác về phép tính.** Không đưa ra được con số tính toán cụ thể cho 5 ngày quá hạn.
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* **Có đầy đủ quy định** về hạn mức 15 ngày và tỷ lệ phạt 2%/tháng trong `tam_ung.md`.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Câu hỏi rất rõ ràng về tình huống thực tế.
  4. *Cần sửa lỗi ở module nào trong pipeline?* **Tầng LLM Prompt Generation (System Prompt)**.
- **Error Tree:** Output tính toán sai → Context đúng → LLM thiếu khả năng suy luận số học từng bước (Arithmetic CoT) khi giải bài toán pro-rata.
- **Root cause:** Prompt hiện tại chỉ yêu cầu "Trả lời CHỈ dựa trên context", khiến mô hình có xu hướng trích nguyên văn công thức thay vì thực hiện phép tính suy luận cho người dùng.
- **Suggested fix:** Nâng cấp System Prompt cho phép Chain-of-Thought (CoT): "Nếu câu hỏi yêu cầu tính toán con số cụ thể, hãy trích dẫn công thức từ context và tính toán từng bước rõ ràng trước khi kết luận."

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
> *"Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?"*

**Error Tree walkthrough:**
1. **Output đúng?** $\rightarrow$ **SAI**. Mô hình trả lời "5 năm được cộng 1 ngày phép", trong khi chính sách mới nhất năm 2024 quy định chỉ cần "3 năm".
2. **Context đúng?** $\rightarrow$ **SAI**. Context cung cấp cho mô hình là đoạn trích từ `nghi_phep_nam_v2023.md`. Đoạn văn `nghi_phep_nam_v2024.md` bị loại khỏi top 3 sau bước Reranking.
3. **Query rewrite OK?** $\rightarrow$ Người dùng hỏi tự nhiên không kèm năm hiệu lực. Câu hỏi phản ánh đúng thực tế người dùng không biết nội bộ vừa đổi chính sách.
4. **Fix ở bước:**  
   - **Module 2 (Search):** Áp dụng Metadata Filtering để lọc các tài liệu có thuộc tính `status == "active"` hoặc `year == 2024`.
   - **Module 5 (Enrichment):** Contextual Prepend cần bổ sung thông tin trạng thái: *"Chính sách nghỉ phép năm 2024 (Đang áp dụng, thay thế bản 2023)..."* để Cross-Encoder dễ dàng phân biệt.

**Nếu có thêm 1 giờ, sẽ optimize:**
1. **Parent Retrieval Context:** Khi tìm kiếm khớp trên Child chunk (256 ký tự), hệ thống sẽ trả về Parent chunk (2048 ký tự) để gửi vào prompt của LLM. Điều này giải quyết triệt để vấn đề mất ngữ cảnh bảng biểu và các ghi chú đi kèm.
2. **Temporal & Metadata Routing:** Tạo bộ lọc thông minh tự động nhận diện tài liệu theo phiên bản/năm hiệu lực, gắn nhãn cảnh báo tài liệu lỗi thời (deprecated documents).
3. **Query Decomposition:** Thêm một prompt nhỏ trước M2 để phân tách các câu hỏi phức hợp thành các câu hỏi đơn, giải quyết dứt điểm các ca Multi-hop query.
