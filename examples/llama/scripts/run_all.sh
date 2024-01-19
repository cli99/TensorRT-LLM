files=(run_bench_fp8.sh run_bench_fp16.sh run_bench_w4a16_awq.sh run_bench_w4a16_gpqt.sh run_bench_w8a8sq.sh run_bench_w8a16_kv8.sh run_bench_w8a16.sh)

files=(run_bench_fp8.sh)

for file in ${files[@]}; do
    echo "Running $file"
    bash $file
done