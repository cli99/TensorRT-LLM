model_dir=../../hf-models/llama2/7b_hf
engine_dir=${model_dir}/trt_engines/w4a16_gptq/1-gpu/
benchmark_file=../../../benchmarks/python/benchmark.py
quantization=int4_weight_only_gptq
            # 'fp8', 'fp8_gemm', 'fp8_kv_cache', 'int8_sq_per_tensor',
            # 'int8_sq_per_token_channel', 'int8_weight_only', 'int4_weight_only',
            # 'int4_weight_only_gptq', 'int4_weight_only_gptq'

in_out_sizes=("64,512" "512,1024" "1024,64")
batch_sizes=(1 2 4 8)
print_header=True
for in_out_size in ${in_out_sizes[@]}; do
    for batch_size in ${batch_sizes[@]}; do
        in_out="${batch_size}:${in_out_size}"
        batch_size=$(echo $in_out | awk -F':' '{ print $1 }')
        in_out_dims=$(echo $in_out | awk -F':' '{ print $2 }')
        echo "BS: $batch_size, ISL/OSL: $in_out_dims"

        python ${benchmark_file} --model llama_7b --engine_dir ${engine_dir} --warm_up 1 --batch_size $batch_size --duration 0 --num_runs 20 --input_output_len $in_out_dims --quantization ${quantization} --csv --print_header ${print_header}
        if test ${print_header} = True; then
            print_header=False
        fi
    done
done
