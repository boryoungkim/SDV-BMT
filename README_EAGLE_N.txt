SDV-BMT EAGLE-N WORKING COPY — BẢN ĐỂ REVIEW / PORTING
12/11/2026

1. MỞ ĐÂU / SO SÁNH VỚI ĐÂU?
- Thư mục gốc của bạn: /home/bmt/caontd/SDV-BMT — GIỮ NGUYÊN.
- Bản sạch theo guideline: /home/bmt/caontd/SDV-BMT-v0.1.2-reference.
- Môi trường làm việc: /home/bmt/caontd/SDV-BMT-eagle-n (thư mục này).
- Evidence / diff: /home/bmt/caontd/eagle_n_integration_review.

Bản gốc cũ là 391c3c4 (v0.1.0-3). Guideline yêu cầu v0.1.2 / f3c8da0.
Vì vậy hãy so sánh theo HAI LỚP:
  A. SDV-BMT ↔ SDV-BMT-v0.1.2-reference: thay đổi upstream nâng baseline.
  B. SDV-BMT-v0.1.2-reference ↔ SDV-BMT-eagle-n: phần Eagle-N do ta bổ sung.
Nếu chỉ so A gốc với bản làm việc, các file DeepLab/codec/Cityscapes evaluator
của v0.1.2 dễ bị nhầm là code Eagle-N do ta viết.

Khi dùng Beyond Compare / Meld / VS Code folder diff, loại khỏi view:
  .git/, .venv/, native/build/, __pycache__/, .pytest_cache/, *.egg-info/.
Bộ guide riêng trong docs/porting/ được copy nguyên từ thư mục gốc, không phải
code plugin và không được commit vào repo công khai.

2. ĐÃ CÓ NHỮNG GÌ?
- Adapter bos_eagle_n kế thừa NativeRealtimeAdapter, strict DUT/program options.
- Đăng ký CLI trong bản sao; target native optional tách khỏi all/portable.
- Native backend C++: prepare/stage/submit/poll/collect/observe/cleanup.
- Executor C++ đã BUILD bằng SDK BOS: graph TTNN, chuẩn bị weights/programs,
  capture trace ở prepare, replay + H2D/D2H trong submit; không Python forward.
- Offline exporter: đọc đúng ONNX H04, giữ Conv/Relu/Add/Pool/GAP/Concat/Resize/
  ArgMax; chỉ folding shape expressions trên CPU; final uint32→uint8 ở collect.
- Đã xuất plan/weights thật của R18/R50, hash pin và đặt NGOÀI repository.
- Config DUT/workload/capture cho hai model; baseline là .yaml.in CHỜ NPU golden L4.
- Hash inputs, raw byte format, current source stamp, native qualification guard,
  staged-memory budget, N=1 slot guard, class range check trước uint8 cast.
- Tests adapter + native contract/mock + SDK opt-in. MOCK không đăng ký trong CLI
  production; production từ chối thư viện có real_device=false.
- Launcher container read-only; build/describe/validate/evaluate KHÔNG expose NPU.
  run yêu cầu native qualification và staging budget trước khi gắn device;
  hugepages dùng mount riêng trong caontd, không reserve hoặc sửa mount chung.

3. TRẠNG THÁI CẦN ĐỌC ĐÚNG
Đây là môi trường porting / plugin wired + SDK-buildable, CHƯA phải full native
model đã được nghiệm thu trên NPU hoặc G14/G15 hoàn tất.
- R18/R50 standalone Python trước đây đã PASS. Kết quả đó giữ nguyên ở thư mục cũ.
- Executor C++ mới đang thử ACTUAL single-image theo §8; chưa qua đủ
  operator/module/stage/dataset qualification gates. Kết quả mới nhất:
  FULL R18 qua SDV-BMT một ảnh ACTUAL đã PASS; khớp standalone NPU 100%.
- Native SDK build + describe thành công không chứng minh giá trị inference đúng.
- MOCK bundle/evaluation chỉ chứng minh framework integration, KHÔNG phải model
  accuracy, accelerator latency hoặc throughput.
- Chưa có dataset val500 tại vị trí canonical trong working copy.
- Chưa có native_qualification.json hay NPU capture golden. Không tạo receipt giả.
- Mặc định staged budget 4GiB: 500 item FP32 canonical đã có payload 11,72GiB;
  layout/dtype/padding còn có thể tăng memory. Phải giải quyết residency/budget
  thực trước full capture. Không tăng cap mù quáng hoặc sửa contract để qua.
- Device time None; clocks/firmware/silicon revision không có số query thì ghi
  not observable. Không dùng spec lý thuyết làm số runtime đã đo.

4. FILE MAP CHO REVIEW
DUT API / control:
  src/sdv_bmt/dut/eagle_n/adapter.py, __init__.py
  strict config, native identity, measurement declaration, backend_spec,
  realization evidence; Python không chạy per request.
Native measurement / DUT backend:
  native/src/main_eagle_n.cpp
  native/src/eagle_n/backend.hpp
  native/src/eagle_n/executor_abi.hpp
  hash/qualification gate, ownership/lifetime, host-staging quota, ErrorInfo,
  synchronous submit/poll, preallocated uint32/u8 outputs, cleanup.
NPU inference implementation:
  native/src/eagle_n/ttnn_executor.cpp
  TTNN C++ graph / bounded jobs / dtype adapters / trace replay; chưa NPU qualify.
Offline assets / launch:
  tools/eagle_n/export_plan.py — ONNX→native plan + aligned FP32 constants.
  tools/eagle_n/configure.py — config model/DUT/capture; baseline template until L4.
  tools/eagle_n/build_native.sh — build against existing BOS C++ SDK, no device.
  tools/eagle_n/in_container.py — pinned image, SDK read-only, private runtime.
  tools/eagle_n/sdk_identity.py — fingerprint SDK dependencies, phát hiện drift.
  tools/eagle_n/audit_readiness.py — liệt kê blockers mà không mở device.
  tools/eagle_n/review_diff.py — báo cáo folder diff / original-file verification.
Configs mới:
  configs/duts/eagle-n-{r18,r50,r101}.yaml
  configs/workloads/deeplabv3plus-{r18,r50,r101}-cityscapes-v1-eagle-n.yaml
  configs/experiments/deeplabv3plus-{r18,r50,r101}-eagle-n-capture.yaml
  configs/experiments/deeplabv3plus-{r18,r50,r101}-eagle-n-baseline.yaml.in
Tests:
  tests/unit/test_eagle_n_adapter.py
  tests/unit/test_eagle_n_native_contract.py
  tests/eagle_n/{contract_mock.cpp,contract_main.cpp,backend_guards.cpp}
  tests/device/test_eagle_n_native.py

Shared files thay đổi CHỈ TRONG WORKING COPY:
  native/Makefile: target optional eagle-n/eagle-n-sdk; source dependency cho
    identity/rebuild. all/portable không yêu cầu SDK và không link BOS.
  src/sdv_bmt/cli/main.py: import + DUT_ADAPTERS; ACTUAL notice thành generic
    để không gắn mọi thiết bị vào nhãn GPU reference.
KHÔNG sửa runner, scheduler, metric formulas, bundle schema, DeepLab workload,
input contract, postprocess/codec hoặc evaluator chung.
Xem docs/eagle_n/REQUEST_SHARED_CHANGES.txt trước khi đề xuất merge upstream.

5. ENVIRONMENT VÀ LỆNH KIỂM TRA KHÔNG CHẠY NPU
  cd /home/bmt/caontd/SDV-BMT-eagle-n
  .venv/bin/python --version                 # local venv Python 3.11
  .venv/bin/sdv-bmt --version               # 0.1.2
  make -C native portable PYTHON="$PWD/.venv/bin/python"
  native/build/sdv-bmt-native-tests
  SDV_BMT_MAKE_TARGETS=portable PYTHON="$PWD/.venv/bin/python" \
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -rs

SDK link được pin vào build_Release_distributed như standalone cũ; KHÔNG dùng
install_distributed vì _ttnncpp.so/libtt_metal.so ở đó khác checksum.

Native SDK library cần BOS Ubuntu/container toolchain; host sandbox có glibc
cũ không load được thư viện BOS. Đó là khác biệt môi trường, không phải NPU lỗi.
  sudo -v
  .venv/bin/python -B tools/eagle_n/in_container.py build
  .venv/bin/python -B tools/eagle_n/configure.py
  .venv/bin/python -B tools/eagle_n/in_container.py describe
Các lệnh này không expose /dev/bos/0. configure phải chạy lại sau khi binary/hash đổi.

Assets hiện tại ngoài repo:
  /home/bmt/caontd/eagle_n_integration_assets/libeagle_n_ttnn.so
  /home/bmt/caontd/eagle_n_integration_assets/r18/{r18.plan.json,r18.weights.f32,compile_record.json}
  (2026-11-18: assets r50/ (FP32-conv) và r101/ (untuned) cũ đã gỡ, chưa từng qualify; R-50/R-101 BF16 đi kèm bàn giao sau)
Không đặt ONNX, weights, images, SDK hoặc bundles vào source commit.

6. BƯỚC PORTING TIẾP — TRƯỚC KHI NÓI “CHẠY THÀNH CÔNG TRONG SDV-BMT”
- Review native execution path/Plan C + mixed precision + host uint8 Cast với
  Boryoung (D01 và boundary H10); ghi đây là implementation riêng từ H04.
- R18: native operator/tail/tie/layout probes → module → stage → full canonical
  image; kiểm tra trace reuse trên nhiều INPUT THỰC, không chỉ zero warm-up.
- Kiểm tra lifetime, input/output copy, no hot-path Python/logging/allocation/JIT,
  trace-memory fit và RSS. Build/describe/MOCK không thay bước này.
- Qualification receipt PHẢI là kết quả native thật, match plan/weights/executor
  hashes, không copy/đổi tên receipt Python cũ. Hiện bản sao cố ý chặn run khi thiếu.
- L1 canonical numeric + task quality; L2 describe/validate; L3 capture → bundle →
  evaluate; L4 NPU golden + capture lần2; L5 baseline5 lần đúng rate/boundary.
- R18 qua L5 rồi mới R50, sau đó R101 theo cùng plugin. 2Hz và timeout/deadline giữ nguyên;
  nếu NPU không theo kịp, báo/chốt rate, không tự đổi để pass. R101 cần cổng tài nguyên/tile riêng.

7. GUIDELINE / PHẠM VI
Common guide §2, §5–§10; device guide §5.1 và Step2. Ghi nguồn chi tiết ở
  docs/eagle_n/BOUNDARY_TABLE.txt
  docs/eagle_n/INTEGRATION_STATUS.json
User đã yêu cầu thực hiện trong bản sao riêng để review folder diff. Điều đó
không phải approval merge/publication, không tự chấp thuận các tiêu chí G15
đang pending. Không push, reset NPU, reboot, reload driver hoặc sửa SDK chung.

8. THỬ TRỰC TIẾP R18 THEO YÊU CẦU NGƯỜI DÙNG (12/11/2026)
Có thêm option trial_unqualified=true, mặc định false. Đây là thử tích hợp
ACTUAL trên một ảnh thật, không phải MOCK, không tạo qualification receipt giả.
Launcher giới hạn R18, selection.count=1, precondition_requests=0, experiment id
có chữ trial. Đường benchmark mặc định vẫn yêu cầu qualification.
Trong trial, executor KHÔNG zero warm-up/capture trace: submit H2D trên CQ0,
chạy graph TTNN trực tiếp, synchronize rồi D2H. JIT/host graph construction,
allocation và log operator nằm trong cửa sổ thử; không dùng latency này làm số
steady-state hay nghiệm thu native hot-path. Các phép toán mạng vẫn trên NPU.
Config/input riêng: /home/bmt/caontd/eagle_n_integration_assets/r18_trial/.
Ảnh là cùng ảnh đã có golden standalone, không phải bộ Cityscapes val500.
Workload trial tắt quality_evaluator vì không có ground truth của dataset;
đối chiếu label agreement với ORT/PyTorch/standalone NPU ở bước offline bằng
  .venv/bin/python -B tools/eagle_n/compare_trial.py BUNDLE --out OUTPUT_DIR
Đó không phải mIoU và không thay thế dataset qualification.
Lỗi phát hiện và sửa khi thử:
- SDK write_tensor host→device yêu cầu blocking=false. FIFO cùng CQ0 bảo đảm
  graph dùng dữ liệu sau write; synchronize/D2H cuối bảo đảm output hoàn tất.
- Decoder 128+12=140 kênh cần giữ row-major ở concat, flatten tile trước
  untilize, chia spatial depthwise có halo và pointwise thành M<=512. Các bản
  trước phát sinh CB 16.880.592 B > L1 1.572.864 B/core ở từng bước này.
- Bỏ giant reshape nhãn trên device: giữ output ArgMax và đọc logical vector
  uint32 bằng SDK trong trial, rồi chuyển uint8 ở collect. Logical unpack phía
  host có allocation trong cửa sổ trial; chưa phải native hot-path qualification.
- Kiểm tra packed output bytes đúng buffer uint32 trước memcpy D2H.
Các lỗi trên được ghi thành request error trong bundle; process exit=0 và
execution_status=completed KHÔNG có nghĩa request inference đã thành công.
Phải kiểm tra request_counts.completed/error và output rồi mới kết luận.

KẾT QUẢ TRIAL ĐÃ PASS:
/home/bmt/caontd/eagle_n_integration_evidence/run_20261112T081434_426852Z/verification/R18_TRIAL_RESULT.txt
1/1 completed; 100% match previous NPU; 99,87001419% match FP32 golden.
Đây là full graph một ảnh, chưa phải dataset/trace/steady-state qualification.


9. R-101 WIRED INTO THE WORKING COPY (13/11/2026)
R101 now has an H04-pinned offline plan and weights plus Eagle-N DUT/workload/
capture configs in this working copy. The ONNX SHA-256 matches
86d0f08c1852723892777c0216936e5b790d3b9345f0e83473fb21309c129745; the plan
contains 121 logical Conv operations. Precision is declared
`bf16-conv-fp32dst`, matching the native executor's BF16/HiFi4 branch (not the
R50 FP32 Conv path). R101 is included in the adapter, exporter, readiness audit,
and adapter contract-test matrix.

This is source/config integration only. The offline plan record says
`device_executed=false` and `EXPORTED_NOT_QUALIFIED`. The native binary has not
been rebuilt for this change, SDV-BMT has not opened the NPU for R101, and no
native qualification receipt, NPU golden or baseline exists. Capture is gated;
the baseline remains `.yaml.in` and cannot be run as a normal experiment.

To export/configure R101 without rewriting existing R18/R50 files:
  .venv/bin/python -B tools/eagle_n/export_plan.py \
    --onnx /home/bmt/caontd/eagle_n_porting/deeplab_r101/models/deeplabv3plus_r101.onnx \
    --variant r101 --out /home/bmt/caontd/eagle_n_integration_assets/r101
  .venv/bin/python -B tools/eagle_n/configure.py \
    --assets /home/bmt/caontd/eagle_n_integration_assets \
    --executor /home/bmt/caontd/eagle_n_integration_assets/libeagle_n_ttnn.so \
    --variant r101

R101 is the largest of the three (121 Conv). Before NPU use, rebuild the native
executor and perform operator/module/stage/full correctness and resource gates.
Do not enable `trial_unqualified` for R101. A standalone Python TTNN PASS does
not qualify this separate native C++ execution path.
