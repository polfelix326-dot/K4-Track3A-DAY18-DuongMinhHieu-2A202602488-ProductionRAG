# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Dương Minh Hiếu  
**Mã học viên:** 2A202602488  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Bảng đối chiếu chi tiết giữa các khái niệm lý thuyết cốt lõi trong bài giảng Production RAG và các hàm thực tế đã triển khai trong mã nguồn:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích chuyên sâu |
|----------------|:------:|:-------------|------------------------------------|
| **Advanced Chunking** *(Semantic, Hierarchical, Structure-Aware)* | M1 | `chunk_semantic()`<br>`chunk_hierarchical()`<br>`chunk_structure_aware()` | - `chunk_semantic()`: Dùng `all-MiniLM-L6-v2` và ngưỡng tương đồng cosine 0.85 để gom nhóm câu liền kề cùng chủ đề, tránh việc cắt câu giữa chừng.<br>- `chunk_hierarchical()`: Tạo cấu trúc Parent-Child (Parent 2048 chars, Child 256 chars). Tìm kiếm so khớp trên Child chunk để đạt độ đặc hiệu (precision), nhưng hoàn toàn có thể trả về Parent chunk để LLM có ngữ cảnh hoàn chỉnh.<br>- `chunk_structure_aware()`: Bắt các tiêu đề Markdown (`#`, `##`, `###`), giữ nguyên khối bảng biểu và danh sách, gán metadata `section`. |
| **Hybrid Search & Fusion** *(Lexical + Dense + RRF)* | M2 | `segment_vietnamese()`<br>`BM25Search.search()`<br>`DenseSearch.search()`<br>`reciprocal_rank_fusion()` | - Tiếng Việt có từ ghép ("nghỉ phép"), `underthesea.word_tokenize` nối bằng dấu `_`. Cần `replace("_", " ")` để BM25 tokenize chính xác theo khoảng trắng.<br>- `DenseSearch`: Dùng `BAAI/bge-m3` (1024-dim) và vector database Qdrant với phương thức `query_points()`.<br>- `reciprocal_rank_fusion()`: Áp dụng $RRF = \sum \frac{1}{k + rank + 1}$ ($k=60$), xếp hạng không phụ thuộc thang đo điểm số, kết hợp hoàn hảo từ khóa chính xác (BM25) và ngữ nghĩa sâu (Dense). |
| **Cross-Encoder Reranking** *(Deep Attention)* | M3 | `CrossEncoderReranker._load_model()`<br>`CrossEncoderReranker.rerank()` | - Khắc phục điểm yếu của Bi-Encoder: Cross-Encoder `BAAI/bge-reranker-v2-m3` nhận trực tiếp cặp `(query, document)` và tính attention chéo giữa từng từ.<br>- Sàng lọc từ top 20 candidate xuống top 3 kết quả đắt giá nhất gửi vào LLM prompt. Bổ sung `_CROSS_ENCODER_CACHE` để tái sử dụng model trong RAM, tránh overhead reload. |
| **Automated Evaluation & Error Tree** *(RAGAS)* | M4 | `evaluate_ragas()`<br>`failure_analysis()` | - Đánh giá tự động 4 chiều: Faithfulness (độ trung thực), Answer Relevancy (độ liên quan), Context Precision (độ chuẩn xác ngữ cảnh), Context Recall (độ phủ ngữ cảnh).<br>- Diagnostic Tree tự động phát hiện nguyên nhân gốc rễ và đề xuất giải pháp xử lý (Prompt, Chunking, Retrieval hay Reranking). |
| **Contextual Prepend & HyQA** *(Document Enrichment)* | M5 | `summarize_chunk()`<br>`generate_hypothesis_questions()`<br>`contextual_prepend()`<br>`_enrich_single_call()` | - `contextual_prepend()`: Gắn 1 câu bối cảnh trước đoạn trích giúp giảm 49% lỗi trích xuất theo nghiên cứu của Anthropic.<br>- `generate_hypothesis_questions()` (HyQA): Sinh câu hỏi giả định để cầu nối từ vựng giữa câu hỏi người dùng và văn bản.<br>- `_enrich_single_call()`: Gom cả 4 tác vụ vào 1 API call với `json_object` format để tối ưu chi phí và độ trễ. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

Trong quá trình xây dựng hệ thống Production RAG, tôi đã gặp và giải quyết 3 vấn đề kỹ thuật lớn:

1. **Lỗi nạp thư viện PyTorch trên môi trường Windows:**
   - **Lỗi kỹ thuật:** `OSError: [WinError 126] The specified module could not be found. Error loading ".../torch/lib/cublas64_12.dll" or one of its dependencies.`
   - **Nguyên nhân gốc rễ:** Môi trường Python toàn cục của hệ thống thiếu CUDA runtime DLLs tương thích với bản build PyTorch GPU.
   - **Cách xử lý:** Kích hoạt môi trường ảo riêng của dự án `.venv\Scripts\python.exe` với PyTorch phiên bản CPU (`2.14.1+cpu`) hoạt động độc lập và ổn định, loại bỏ hoàn toàn xung đột DLL.

2. **Vấn đề độ trễ khi khởi tạo CrossEncoder trong kiểm thử tự động:**
   - **Hiện tượng:** Bộ test `pytest tests/test_m3.py` mất hơn 8 phút do model `BAAI/bge-reranker-v2-m3` nặng hơn 2.2GB bị khởi tạo lại ở mỗi unit test case (`CrossEncoderReranker()`).
   - **Cách xử lý:** Bổ sung module-level cache `_CROSS_ENCODER_CACHE: dict[str, object] = {}`. Model chỉ nạp một lần duy nhất vào bộ nhớ và tái sử dụng cho tất cả các lượt rerank tiếp theo, giảm thời gian thực thi toàn bộ test suite từ 519s xuống còn 25s (nhanh hơn gấp 20 lần).

3. **Vấn đề tách từ tiếng Việt và tương thích với BM25:**
   - **Hiện tượng:** Tìm kiếm từ khóa "nghỉ phép" không khớp với văn bản chứa "Nhân viên được nghỉ phép năm".
   - **Nguyên nhân:** `underthesea.word_tokenize(..., format="text")` tạo ra chuỗi `"nghỉ_phép"`. Khi BM25 split khoảng trắng thì chuỗi này thành 1 token duy nhất `"nghỉ_phép"`, trong khi query của người dùng split thành 2 token `["nghỉ", "phép"]`.
   - **Cách xử lý:** Sử dụng `.replace("_", " ")` sau khi tokenize, đồng thời chuẩn hóa chữ thường (`.lower()`) cho cả tập ngữ liệu và câu truy vấn trong `BM25Search`.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống Trợ lý Pháp lý & Tra cứu Quy chế Doanh nghiệp (Enterprise Legal & Policy AI Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng Basic RAG (cắt đoạn thô theo kích thước 500 ký tự, Dense Search đơn thuần với OpenAI text-embedding-3-small, đưa thẳng top 5 kết quả vào GPT-4o-mini).
- **Vấn đề / Bottlenecks đang gặp:**
  - *Context Precision thấp:* Tìm kiếm hay bị trúng các điều khoản đã sửa đổi hoặc văn bản hướng dẫn cũ do từ vựng tương tự nhau (Version Conflict).
  - *Hallucination và cụt ý:* Bảng biểu chế độ phụ cấp và hạn mức tài chính bị cắt ngang giữa bảng, khiến LLM suy đoán sai số liệu.
  - *Chi phí token cao:* Gửi nhiều văn bản rác vào prompt làm tăng chi phí API và tăng độ trễ phản hồi (> 4 giây).

#### 2. Kế hoạch cải tiến áp dụng kiến thức Lab 18
1. **Chiến lược Chunking:**
   - Áp dụng **Structure-Aware Chunking** cho các văn bản quy chế, hợp đồng và chính sách: bảo toàn trọn vẹn từng Điều, Khoản và các Bảng biểu đi kèm.
   - Kết hợp cấu trúc **Hierarchical Chunking** (Parent 2048 / Child 256): Vector hóa trên Child chunk để bắt từ khóa chính xác, nhưng trích xuất Parent chunk đưa vào LLM để đảm bảo bối cảnh pháp lý đầy đủ.
2. **Hệ thống tìm kiếm Hybrid Search:**
   - Kết hợp **BM25 tiếng Việt** (đã chuẩn hóa qua `underthesea`) để bắt chính xác số hiệu văn bản (ví dụ "Nghị định 13/2023", "Thông tư 05"), điều khoản và các con số cụ thể.
   - Kết hợp **Dense Search** với `BAAI/bge-m3` để bắt ý nghĩa câu hỏi trừu tượng.
   - Gộp kết quả bằng thuật toán **RRF ($k=60$)**.
3. **Bộ lọc Metadata & Temporal Routing:**
   - Gắn nhãn metadata cho từng văn bản: `effective_date`, `expiration_date`, `department`, `is_active`.
   - Áp dụng Pre-filtering trước khi tìm kiếm để loại bỏ hoàn toàn các văn bản đã hết hiệu lực.
4. **Tầng Cross-Encoder Reranking:**
   - Dùng `BAAI/bge-reranker-v2-m3` rút gọn từ 20 đoạn văn tiềm năng xuống đúng 3 đoạn văn có mức độ liên quan cao nhất, giảm tải 70% lượng token gửi vào LLM.
5. **Đánh giá tự động với RAGAS:**
   - Xây dựng tập test set 50 câu hỏi đa dạng (lookup, version conflict, multi-hop).
   - Thiết lập CI/CD pipeline tự động chấm 4 chỉ số RAGAS mỗi khi cập nhật kho tri thức hoặc thay đổi prompt.

#### 3. Timeline triển khai (4 tuần)
- **Tuần 1:** Chuẩn hóa dữ liệu nguồn, chuyển đổi PDF scan sang Markdown chuẩn, triển khai Module Structure-Aware & Hierarchical Chunking.
- **Tuần 2:** Thiết lập cơ sở dữ liệu vector Qdrant, dựng Hybrid Search (BM25 + Dense) kèm Metadata Filtering theo phiên bản văn bản.
- **Tuần 3:** Tích hợp tầng Reranking (Cross-Encoder), cấu hình System Prompt tối ưu chống ảo giác (Temperature = 0, CoT cho câu hỏi số học).
- **Tuần 4:** Chạy RAGAS benchmark trên tập test set, hoàn thiện báo cáo phân tích lỗi (Failure Analysis) và đóng gói API dịch vụ FastAPI hoàn chỉnh.
