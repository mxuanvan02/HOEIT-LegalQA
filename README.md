# TQA Pipeline: HOEIT-LegalQA và ECM-TQAG

Repo này chứa hai nhánh riêng biệt về hỏi–đáp pháp luật tiếng Việt từ giáo trình:

- **HOEIT-LegalQA (benchmark):** pipeline xây dựng bộ dữ liệu trắc nghiệm có nhãn Bloom. Dữ liệu: [maixuanvan/HOEIT-LegalQA](https://huggingface.co/datasets/maixuanvan/HOEIT-LegalQA).
- **ECM-TQAG (giao thức thực nghiệm):** pipeline sinh và kiểm toán chuỗi bằng chứng. Vị trí dữ liệu dẫn xuất dự kiến: [maixuanvan/ECM-TQAG](https://huggingface.co/datasets/maixuanvan/ECM-TQAG); chưa phát hành artifact ECM nào cho đến khi các cổng sinh, chất lượng và quyền đều đạt.

Không được gộp hai nhánh. HOEIT-LegalQA là bản phát hành benchmark; ECM-TQAG là nghiên cứu thực nghiệm ràng buộc nguồn.

> **Ranh giới phát hành:** repo này không cấp quyền phân phối lại giáo trình nguồn, ảnh trang sách hay sổ ghi đầu ra thô của mô hình. Một bản ghi ECM parse được chỉ có nghĩa hợp đồng cấu trúc/nguồn gốc của nó đạt — không chứng minh tính đúng pháp lý, chất lượng sư phạm, tính duy nhất của đáp án đúng hay khả năng bám ảnh.

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

## ECM-TQAG: sinh QA đa phương thức theo chương trình đồ thị

### Hợp đồng phương pháp

Với mỗi chunk đã đóng băng, ECM-TQAG dùng ba điều kiện bằng chứng:

- **T:** chỉ văn bản trích xuất.
- **TL_struct:** văn bản + cấu trúc tài liệu khai báo.
- **TLV:** văn bản + cấu trúc + pixel ảnh đính kèm.

Mỗi điều kiện sinh một ứng viên theo ba giao thức:

- **Direct:** tạo câu trắc nghiệm từ một mệnh đề được nguồn hỗ trợ trực tiếp.
- **Answer-first:** khóa đáp án được hỗ trợ trước, rồi dựng câu hỏi và phương án nhiễu cùng miền.
- **ECM (Evidence-Chain Method):** một lời gọi planner, sau đó (khi plan qua các phép kiểm tất định) một lời gọi realization. Planner đề xuất đồ thị tài liệu ràng buộc nguồn và một yêu cầu motif trong danh mục đóng. Mã cục bộ khớp motif, biên dịch và thực thi một chương trình đồ thị hạn chế, dẫn xuất các nguyên tử đáp án đã khóa và vết nguồn gốc. Realizer **chỉ nhận cấu trúc đã khóa này**, không nhận văn bản gốc hay pixel ảnh, và tạo một câu trắc nghiệm.

Validator từ chối: ID/vai không hợp lệ, nút không neo vào bằng chứng đã đóng băng, motif không khớp, thiếu nút ảnh trong plan ECM–TLV, vết/neo không đầy đủ, và nguyên tử đáp án do executor dẫn xuất bị thay đổi. Ma trận đầy đủ là `8 chunks × 3 điều kiện × 3 giao thức = 72 ô`. Direct và Answer-first mỗi ô một lời gọi; ECM dùng planner cộng realization khi lập kế hoạch thành công, nên thiết kế đầy đủ tối đa **96 lời gọi API**.

### Cấu trúc repo

```text
scripts/research/build_ecm_8chunk_manifest.py  # dựng manifest 8×3 bất biến
scripts/research/run_qwen37_tqa_pilot.py       # pilot thu gọn hoặc ma trận đầy đủ
scripts/research/audit_strict_tqa_results.py   # kiểm toán nguồn gốc tất định
tests/test_qwen37_tqa_pilot.py                 # test hợp đồng ECM
research/artifacts/                            # manifest cục bộ, git-ignored
research/results/                              # sổ ghi/audit cục bộ, git-ignored
```

Tên file runner được giữ để tương thích với script hiện có; nó hỗ trợ cả đánh giá thu gọn lẫn ma trận đầy đủ.

### Yêu cầu

- Python 3.10+; riêng runner ECM chỉ dùng thư viện chuẩn.
- Khóa OpenRouter được cấp phép cho `qwen/qwen3.7-plus`.
- Manifest bất biến cục bộ và các file ảnh mà nó tham chiếu.

Cho pipeline OCR/chunking rộng hơn:

```bash
python -m pip install -r requirements.txt
```

Không bao giờ commit `.env`, khóa API, giáo trình thô, ảnh trích xuất hay sổ ghi thô.

### Dựng gói bằng chứng bất biến

Runner sinh tiêu thụ manifest đã đóng băng thay vì văn bản nguồn tùy ý. Điều này làm đầu vào T/TL_struct/TLV tường minh và kiểm tra kích thước/hash ảnh trước mỗi lời gọi.

```bash
python scripts/research/build_ecm_8chunk_manifest.py \
  --contexts data/output/interim/multimodal_contexts.json \
  --pilot-manifest research/artifacts/pilot_evaluation_manifest.json \
  --out research/artifacts/ecm_inputs_8chunks_v3.json
```

Hợp đồng đầu vào yêu cầu đúng tám chunk và đủ ba gói T, TL_struct, TLV cho mỗi chunk. T chỉ có văn bản; TL_struct thêm cấu trúc khai báo; TLV thêm ảnh đã kiểm chứng. Markdown của ảnh, tên file và dấu phân cách trang bị loại khỏi văn bản dùng chung; pixel chỉ đính kèm cho TLV. Khi việc tách OCR để lại một gói thiếu ngữ cảnh ngữ nghĩa, chỉ được bổ sung bằng ngữ cảnh nguồn liền kề có bằng chứng; không chèn văn bản láng giềng không liên quan.

### Kiểm tra trước khi gọi API

```bash
python -m unittest tests/test_qwen37_tqa_pilot.py -v

RUN_ID="ecm_graph_program_matrix72_YYYYMMDD"
OUT_DIR="research/results/${RUN_ID}"

python scripts/research/run_qwen37_tqa_pilot.py \
  --manifest research/artifacts/ecm_inputs_8chunks_v3.json \
  --out-dir "$OUT_DIR" \
  --experiment-id "$RUN_ID" \
  --model qwen/qwen3.7-plus \
  --api-key-env OPENROUTER_API_KEY \
  --seed 20260824 \
  --all --dry-run
```

Dry run phải báo cáo 72 ô và 96 lời gọi dự kiến; nó không đọc khóa API và không ghi thư mục kết quả.

### Chạy các chunk đã đóng băng vào sổ kết quả

Đặt khóa trong chính terminal khởi động runner; không bao giờ đưa vào Git, notebook, issue hay nhật ký chat.

```bash
export OPENROUTER_API_KEY='replac...cret'

RUN_ID="ecm_graph_program_matrix72_YYYYMMDD"
OUT_DIR="research/results/${RUN_ID}"
test ! -e "$OUT_DIR" || { echo "Refusing to overwrite $OUT_DIR"; exit 1; }

python scripts/research/run_qwen37_tqa_pilot.py \
  --manifest research/artifacts/ecm_inputs_8chunks_v3.json \
  --out-dir "$OUT_DIR" \
  --experiment-id "$RUN_ID" \
  --model qwen/qwen3.7-plus \
  --base-url https://openrouter.ai/api/v1/chat/completions \
  --api-key-env OPENROUTER_API_KEY \
  --seed 20260824 \
  --timeout-sec 180 \
  --retries 2 \
  --all
```

Thư mục không ghi đè chứa `results.jsonl` (sổ ô chỉ-append), `prompt_audit.jsonl` (hash prompt và biên nhận trường), và `summary.json`. Exit code khác 0 ghi nhận reject/error mà không xóa sổ; không chạy lại vào cùng thư mục đầu ra.

Với pilot thu gọn 9 ô, dùng cả ba giá trị `--condition` và một `--chunk-id` trong thư mục đầu ra riêng.

### Kiểm toán cơ học

```bash
python scripts/research/audit_strict_tqa_results.py \
  --results "$OUT_DIR/results.jsonl" \
  --manifest research/artifacts/ecm_inputs_8chunks_v3.json \
  --output "$OUT_DIR/mechanical_audit.json"
```

Phép kiểm audit replay lại: hình dạng bản ghi, các phương án phân biệt, tính nhất quán đáp án/lựa chọn khi áp dụng, nút và cạnh đồ thị kiểu ràng buộc nguồn, vị từ motif trong danh mục đóng, đầu ra compiler chương trình hạn chế chính xác, nguyên tử đáp án do executor dẫn xuất, biên nhận cấu trúc, vết chương trình, neo nút đồ thị đã khóa, và yêu cầu nút ảnh ECM–TLV. Đây không phải bộ phán quyết ngữ nghĩa, pháp lý, sư phạm hay bám ảnh; những mục đó cần quy trình thẩm định người/chuyên gia có tài liệu hóa.

### Kết quả đánh giá ECM-TQAG (ma trận 72 ô)

Ma trận Qwen3.7-plus hoàn chỉnh gồm 72 ô. Sau khi chạy lại 18 ô thuộc hai chunk thiếu ngữ cảnh bằng manifest bằng chứng bổ sung, 60 ô parse được (83,3%) và 12 ô bị từ chối (16,7%). Kết quả cuối báo cáo với nguồn gốc hỗn hợp: 54 ô không ảnh hưởng dùng manifest gốc và 18 ô chạy lại dùng manifest bổ sung. Hai tập con được kiểm toán riêng theo digest manifest của từng tập.

12 ô bị từ chối phân loại như sau:

| Tình huống | Số ô | Diễn giải |
|---|---:|---|
| Bằng chứng không đủ | 5 | Ngữ cảnh nguồn cung cấp không hỗ trợ cấu trúc hoàn chỉnh; bốn ô xảy ra ở gói chỉ có tiêu đề và ảnh. |
| Neo nguồn không đúng nguyên văn | 4 | Một nút hoặc neo đồ thị diễn giải lại thay vì tái tạo đúng một đoạn nguồn liên tục. |
| Đáp án ECM không neo vào nguyên tử executor | 2 | Phương án được chọn không được neo cơ học vào nguyên tử đáp án đã dẫn xuất. |
| Neo bằng chứng ECM không hợp lệ | 1 | Neo cuối không khớp nút đồ thị đã khóa. |

Đây là các kết quả cấu trúc/nguồn gốc, không phải phán quyết về tính đúng pháp lý hay chất lượng giáo dục. Bản ghi chi tiết giữ ở cục bộ vì đầu ra thô của mô hình, đoạn trích giáo trình và ảnh trang không được phép phân phối lại.

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

Prompt runtime nằm trong `scripts/research/run_qwen37_tqa_pilot.py`; phiên bản prompt và biên nhận SHA-256 lưu cục bộ theo từng ô. Chỉ phát hành mã nguồn, fixture tổng hợp, schema không nhạy cảm và bản ghi dẫn xuất đã được phép phân phối lại. Không tải lên sách thô, bản scan, pixel hình vẽ, đoạn trích nguyên văn dài, thông tin xác thực hay đầu ra mô hình thô chưa rà soát.

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
