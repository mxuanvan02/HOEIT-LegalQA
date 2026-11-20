# HOEIT-LegalQA: Bộ dữ liệu hỏi–đáp pháp luật tiếng Việt từ giáo trình

Repo chứa pipeline xây dựng bộ dữ liệu trắc nghiệm có nhãn Bloom. Dữ liệu: [maixuanvan/HOEIT-LegalQA](https://huggingface.co/datasets/maixuanvan/HOEIT-LegalQA).

> **Ranh giới phát hành:** repo này không cấp quyền phân phối lại giáo trình nguồn hay ảnh trang sách. Bộ dữ liệu không chứng minh tính đúng pháp lý, chất lượng sư phạm hay tính duy nhất của đáp án đúng cho đến khi có thẩm định chuyên gia độc lập.

## HOEIT-LegalQA

### Bản phát hành cuối cùng

- Nguồn: 48 giáo trình luật bậc đại học của Viện Đào tạo Mở và Công nghệ Thông tin, Đại học Huế.
- **Bản phát hành cuối: 4.668 câu trắc nghiệm bốn lựa chọn** (`data/{train,dev,test}.jsonl` trên HF), chia theo giáo trình nguồn: 3.440 / 540 / 688 (train/dev/test), không giao `doc_id`/`chunk_id` giữa các tập.
- Nhãn Bloom: Remember, Understand, Apply (30,46% / 31,83% / 37,70%).
- 40 miền pháp luật; vị trí đáp án đúng gần đều (23,5–26,1%).
- Các mô hình đã đánh giá: Gemma-2-9B, Llama-3-8B, Mistral-7B, Qwen2.5-7B.

Bản phát hành cuối là lớp đánh giá 14.210 câu **sau tầng kiểm toán nội dung**: loại thân câu suy biến (3.602), ngữ cảnh cắt cụt tại ngưỡng 3.000 ký tự (2.950), giải thích mâu thuẫn đáp án (943), rò rỉ tiếng Anh (28), rồi cân bằng hạng độ dài của đáp án đúng (đưa heuristic "chọn phương án dài nhất" từ 37,9–43,6% về đúng mức cơ hội 25,0%). Tỷ lệ giữ lại 32,9%. Chi tiết, số liệu kiểm chứng 33/33 và các đường cơ sở tham chiếu (random 25,0%; lexical overlap 43,5% trên test — báo cáo kèm mọi kết quả accuracy) nằm trong dataset card trên HF.

Benchmark phục vụ NLP pháp lý tiếng Việt, QA giáo dục luật, phân tích suy luận theo Bloom và đánh giá trắc nghiệm có ngữ cảnh nguồn. Đây không phải tư vấn pháp lý và không phải công cụ đánh giá năng lực rủi ro cao khi chưa có thẩm định chuyên gia độc lập.

### Pipeline

1. Số hóa PDF: `marker-pdf` chuyển giáo trình sang Markdown, trích tài sản trực quan theo trang.
2. Cấu trúc đa phương thức: chia Markdown thành các ngữ cảnh pháp lý truy nguyên được; `Vintern-1B-v3.5` mô tả các ngữ cảnh có ảnh.
3. Sinh QA: `Qwen2.5-7B-Instruct-AWQ` sinh câu trắc nghiệm tiếng Việt theo ba mức Bloom.
4. Lọc chất lượng: `Gemma-2-2B-IT` kiểm tra độ bám ngữ cảnh, phù hợp đa phương thức, mạch lạc pháp lý và nhất quán nhãn.
5. Kiểm toán nội dung: năm bước xác định (hạt giống 42) — `scripts/build.py` của bản phát hành HF; kiểm chứng độc lập bằng `scripts/verify.py` (33 phép kiểm).

Chuẩn bị benchmark là bước riêng: chuẩn hóa đáp án, làm sạch ngôn ngữ, chia tập theo tài liệu và cân bằng vị trí phương án có tính tất định.

### Chỉ số benchmark

Độ chính xác là tỷ lệ câu trả lời đúng trên toàn bộ câu được đánh giá (mẫu số = số câu của tập, không loại câu nào). Benchmark so sánh điều kiện không ngữ cảnh với điều kiện có đoạn trích nguồn đúng (gắn sẵn; không đánh giá truy hồi). Chênh lệch ngữ cảnh là hiệu số điểm phần trăm ghép cặp theo từng câu. Khoảng tin cậy của chênh lệch dùng bootstrap theo cụm giáo trình (B = 10.000); p-value dùng kiểm định McNemar có hiệu chỉnh Yates trên các ô lệch. Toàn bộ số liệu báo cáo trong bài tính lại được từ `scripts/compute_final_stats.py` (kèm bản phát hành HF) và ledger dự đoán theo từng câu.

## Dữ liệu và mã nguồn

| Artifact | Vị trí | Trạng thái |
| --- | --- | --- |
| Bộ dữ liệu HOEIT-LegalQA (4.668 câu, bản phát hành cuối) | <https://huggingface.co/datasets/maixuanvan/HOEIT-LegalQA> | công khai |
| Mã nguồn pipeline, kiểm toán và kiểm chứng | <https://github.com/mxuanvan02/HOEIT-LegalQA> | công khai (MIT cho mã nguồn) |
| Script tái tính số liệu đặc trưng dataset | `scripts/compute_final_stats.py` trong bản phát hành HF | kèm dữ liệu |

Số liệu công bố trong bài báo tái lập được từ bản phát hành HF bằng
`scripts/compute_final_stats.py`; kiểm chứng độc lập bằng `scripts/verify.py`
(33 phép kiểm, tách biệt khỏi mã xây dựng).

## Giấy phép

Mã nguồn trong repo này phát hành theo giấy phép MIT (xem tệp `LICENSE`).
Giấy phép MIT **không** bao gồm giáo trình nguồn, ảnh trang sách hay đoạn trích
nguyên văn trong bộ dữ liệu; các thành phần đó tuân theo giấy phép
textbook-derived-restricted công bố kèm bộ dữ liệu trên Hugging Face.

## Minh bạch và trích dẫn

Prompt runtime của pipeline nằm trong `src/03_qag_generator.py` (sinh câu hỏi) và `src/04_evaluate.py` (lọc tự động); tham số giải mã lấy từ `src/config.py`. Chỉ phát hành mã nguồn, fixture tổng hợp, schema không nhạy cảm và bản ghi dẫn xuất đã được phép phân phối lại. Không tải lên sách thô, bản scan, pixel hình vẽ, đoạn trích nguyên văn dài, thông tin xác thực hay đầu ra mô hình thô chưa rà soát.

```bibtex
@dataset{hoeitlegalqa2026,
  title   = {HOEIT-LegalQA: A Vietnamese legal question--answering dataset with automatically assigned Bloom labels},
  author  = {Mai, Xuan Van and Tran, Viet Long and Tran, Van Long and Nguyen, Van Khang and Dang, Nguyen Tri and Tran, Vo Hoang Nguyen and Nguyen, Tuong Tri},
  year    = {2026},
  publisher = {Hugging Face},
  url     = {https://huggingface.co/datasets/maixuanvan/HOEIT-LegalQA}
}
```

## Lời cảm ơn, tài trợ và khai báo

Nhóm tác giả cảm ơn Viện Đào tạo Mở và Công nghệ Thông tin, Đại học Huế đã cung cấp môi trường nghiên cứu và dữ liệu nghiên cứu (48 tệp PDF giáo trình luật). Nghiên cứu được tài trợ bởi Đề tài khoa học và công nghệ Đại học Huế, mã số DHH2026-19-09. Trong quá trình thực hiện, nhóm tác giả có sử dụng công cụ trí tuệ nhân tạo tạo sinh để hỗ trợ ngữ pháp trong viết bản thảo và hỗ trợ tạo mã nguồn; các tác giả chịu trách nhiệm toàn bộ về nội dung khoa học, dữ liệu và kết luận.

Quy trình Colab cũ cho OCR/QAG: xem [README_COLAB.md](README_COLAB.md).
