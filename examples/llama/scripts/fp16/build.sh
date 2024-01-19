model_path=../../../hf-models/llama2/7b_hf
build_file=../../../../examples/llama/build.py

python ${build_file} --model_dir ${model_path} \
    --parallel_build \
    --dtype float16 \
    --remove_input_padding \
    --use_gpt_attention_plugin float16 \
    --enable_context_fmha \
    --use_gemm_plugin float16 \
    --max_batch_size 64 \
    --max_input_len 2048 \
    --max_output_len 2048 \
    --output_dir ${model_path}/trt_engines/fp16/1-gpu/
