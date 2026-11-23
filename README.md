# HOEIT-LegalQA: Bộ dữ liệu hỏi–đáp pháp luật tiếng Việt từ giáo trình

Kho mã nguồn của quy trình xây dựng và rà soát bộ dữ liệu hỏi–đáp pháp luật tiếng Việt dạng trắc nghiệm bốn lựa chọn có nhãn theo ba mức của phân loại Bloom.

- **Bộ dữ liệu công khai (Hugging Face):** [maixuanvan/HOEIT-LegalQA](https://huggingface.co/datasets/maixuanvan/HOEIT-LegalQA)

> **Ranh giới phát hành:** Kho mã nguồn này không cấp quyền phân phối lại giáo trình nguồn hay ảnh trang sách. Bộ dữ liệu hiện là **tài nguyên nghiên cứu ứng viên** và không chứng minh tính đúng pháp lý, chất lượng sư phạm hay tính duy nhất của đáp án đúng khi chưa có thẩm định độc lập của chuyên gia.

## Tổng quan bản phát hành cuối

- **Nguồn:** 48 giáo trình luật bậc đại học của Viện Đào tạo Mở và Công nghệ Thông tin, Đại học Huế.
- **Quy mô:** **4.668 câu trắc nghiệm bốn lựa chọn** (`data/{train,dev,test}.jsonl` trên Hugging Face), chia theo giáo trình nguồn không giao nhau: 3.440 huấn luyện / 540 phát triển / 688 kiểm tra (26 / 11 / 11 giáo trình).
- **Nhãn Bloom (điều kiện sinh):** Nhớ 30,46% · Hiểu 31,83% · Vận dụng 37,70%.
- **Độ phủ & cân bằng:** 40 miền pháp luật; vị trí đáp án đúng gần đều (A 23,54% · B 24,55% · C 26,09% · D 25,81%); 5/4.668 câu (0,1%) có yếu tố đa phương thức.
- **Rà soát nội dung (sáu phép kiểm tất định):** từ 14.210 bản ghi trước rà soát, khôi phục nhãn miền NFC/NFD, loại 3.602 thân câu suy biến, 943 giải thích mâu thuẫn đáp án, 2.950 ngữ cảnh cắt cụt ($\ge 3.000$ ký tự), 28 rò rỉ tiếng Anh và cân bằng hạng độ dài đáp án đúng (đưa lối tắt chọn dài nhất từ 43,5–45,7% về đúng mức cơ hội 25,0%). Tỷ lệ giữ lại 32,9%.

## Quy trình bốn giai đoạn và tầng rà soát

1. **Số hóa PDF (`src/01_digitize.py`):** `marker-pdf` chuyển giáo trình sang Markdown, giữ cấu trúc tiêu đề, thông tin trang và đối tượng trực quan.
2. **Phân đoạn và neo tham chiếu (`src/02_structuring.py`):** chia Markdown thành ngữ cảnh pháp lý truy nguyên được; `Vintern-1B-v3.5` mô tả ảnh/bảng/lưu đồ.
3. **Sinh câu hỏi (`src/03_qag_generator.py`):** `Qwen2.5-7B-Instruct-AWQ` sinh câu trắc nghiệm tiếng Việt theo ba mức Bloom kèm giải thích theo tam đoạn luận pháp lý.
4. **Lọc tự động (`src/04_evaluate.py`):** `Gemma-2-2B-IT` chấm theo ba tiêu chí (bám ngữ cảnh, phù hợp đa phương thức, mạch lạc pháp lý).
5. **Rà soát nội dung tất định & kiểm chứng:** sáu phép kiểm tất định (`scripts/build.py` trên bản phát hành HF) và 33 phép kiểm độc lập (`scripts/verify.py`, đạt 33/33).

## Tóm tắt kết quả thực nghiệm (bài báo)

- **Đánh giá 4 mô hình trọng số mở** (`Llama-3-8B`, `Mistral-7B`, `Qwen2.5-7B`, `Gemma-2-9B`) trên 688 câu kiểm tra (*zero-shot*, giải mã tham lam `temperature = 0`, `max_tokens = 4`): cung cấp đoạn trích nguồn đúng tăng độ chính xác **8,14–20,93 đpt** (8,43–12,79 đpt nếu chỉ tính hai mô hình độc lập ngoài họ mô hình sinh và lọc); thiết lập này không đánh giá truy hồi.
- **Kiểm chứng khả năng phục hồi đáp án từ ngữ cảnh bằng hai mô hình quy mô lớn** trên hai mẫu phân tầng 250 câu rút độc lập (ẩn đáp án vàng): tỷ lệ tuyên bố ngữ cảnh không đủ căn cứ là **22,0%** (55/250) với `qwen4-max` (cùng họ với mô hình sinh, gọi theo lô 20 câu) và **8,0%** (20/250) với `claude-opus-5-5` (khác họ, gọi từng câu).

## Cấu trúc kho mã nguồn

```
src/                  Mã nguồn quy trình 4 giai đoạn (01–04) và cấu hình (config.py)
scripts/              Tiện ích xử lý, đánh giá benchmark và chuẩn bị dữ liệu
tests/                Bộ kiểm thử đơn vị (chạy bằng pytest)
run_pipeline.sh       Script chạy quy trình
README_COLAB.md       Hướng dẫn chạy trên Google Colab (tài liệu tham khảo cũ)
LICENSE               Giấy phép MIT cho mã nguồn
```

## Kiểm thử

```bash
pytest
```

## Giấy phép

Mã nguồn trong kho này phát hành theo giấy phép **MIT** (xem tệp `LICENSE`). Giấy phép MIT **không** bao gồm giáo trình nguồn, ảnh trang sách hay đoạn trích nguyên văn trong bộ dữ liệu; các thành phần đó tuân theo điều khoản `textbook-derived-restricted` (`LICENSE.md`) công bố kèm bộ dữ liệu trên Hugging Face.

## Trích dẫn

```bibtex
@dataset{hoeitlegalqa2026,
  title     = {HOEIT-LegalQA: A Vietnamese legal question--answering dataset with automatically assigned three-level Bloom labels},
  author    = {Mai, Xuan Van and Tran, Viet Long and Tran, Van Long and Nguyen, Van Khang and Dang, Tri Nguyen and Tran, Vo Hoang Nguyen and Nguyen, Tuong Tri},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/maixuanvan/HOEIT-LegalQA}
}
```

## Lời cảm ơn, tài trợ và khai báo

Nhóm tác giả cảm ơn Viện Đào tạo Mở và Công nghệ Thông tin, Đại học Huế đã cung cấp môi trường và dữ liệu nghiên cứu (48 giáo trình luật). Nghiên cứu được tài trợ bởi Đề tài khoa học và công nghệ Đại học Huế, mã số DHH2026-19-09. Nhóm tác giả có sử dụng công cụ trí tuệ nhân tạo tạo sinh để hỗ trợ ngữ pháp bản thảo và hỗ trợ tạo mã nguồn; các tác giả chịu trách nhiệm toàn bộ về nội dung khoa học, dữ liệu và kết luận.
