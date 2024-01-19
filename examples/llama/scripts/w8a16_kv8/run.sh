model_dir=../../../hf-models/llama2/7b_hf
engine_dir=${model_dir}/trt_engines/w8a16_kv8/1-gpu/

# Run inference with sliding window/cache size 4096
python3 ${run_file} \
    --max_output_len=50 \
    --tokenizer_dir ${model_dir} \
    --engine_dir ${engine_dir} \
    --log_level info \
    --input_text "The quick brown fox jumps over the"
