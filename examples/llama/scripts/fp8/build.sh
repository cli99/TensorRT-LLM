model_path=../../../hf-models/llama2/7b_hf
quantize_file=../../../../examples/quantization/quantize.py
build_file=../../../../examples/llama/build.py

# Quantize HF model into FP8 and export a single-rank checkpoint
python ${quantize_file} --model_dir ${model_path} \
    --dtype float16 \
    --qformat fp8 \
    --export_path ${model_path}/quantized_fp8 \
    --calib_size 512

# Build model TP=1 using original HF checkpoint + PTQ scaling factors from the single-rank checkpoint
python ${build_file} --model_dir ${model_path} \
    --quantized_fp8_model_path ${model_path}/quantized_fp8/llama_tp1_rank0.npz \
    --dtype float16 \
    --use_gpt_attention_plugin float16 \
    --output_dir ${model_path}/trt_engines/fp8/1-gpu/ \
    --remove_input_padding \
    --enable_context_fmha \
    --enable_fp8 \
    --fp8_kv_cache \
    --strongly_typed \
    --world_size 1 \
    --tp_size 1 \
    --parallel_build \
    --max_batch_size 64 \
    --max_input_len 2048 \
    --max_output_len 2048
