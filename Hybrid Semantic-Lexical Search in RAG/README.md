# Hybrid Semantic-Lexical Search in RAG

Kết hợp giữa search semantic và search lexical cho pipeline RAG.

## Semantic Search
- Chuyển hóa văn bản thành vector và sử dụng cosine similarity để tìm kiếm văn bản tương tự

### Ưu điểm:
- Tìm kiếm văn bản tương tự với query kể cả khi query không chứa từ khóa trong tài liệu
- Xử lý tốt được câu dài và phức tạp
### Nhược điểm:
- Kém chính xác hơn với các từ khóa hiếm

## Lexical Search
- Sử dụng BM25 để tìm kiếm văn bản theo từ khóa

### Ưu điểm:
- Tốc độ nhanh
- Tìm kiếm văn bản chính xác với query thông qua từ khóa (Như id hay series)
### Nhược điểm:
- Không hiểu được ngữ cảnh
- Độ chính xác thấp hơn nếu query không chứa từ khóa trong tài liệu
- Gặp lỗi từ đồng nghĩa

## Hybrid Search
- Kết hợp giữa semantic search và lexical search
- Sử dụng RRF (Reciprocal Rank Fusion) để kết hợp kết quả từ hai phương pháp
- Kết hợp được ưu điểm của cả 2 cách search và giảm thiểu các nhược điểm