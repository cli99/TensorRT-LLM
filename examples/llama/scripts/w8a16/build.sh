model_path=../../../hf-models/llama2/7b_hf
build_file=../../../../examples/llama/build.py
weight_only_precision='int8' # ['int8', 'int4', 'int4_awq', 'int4_gptq']

python ${build_file} --model_dir ${model_path} \
    --dtype float16 \
    --use_gpt_attention_plugin float16 \
    --output_dir ${model_path}/trt_engines/w8a16/1-gpu/ \
    --remove_input_padding \
    --enable_context_fmha \
    --world_size 1 \
    --tp_size 1 \
    --parallel_build \
    --max_batch_size 64 \
    --max_input_len 2048 \
    --max_output_len 2048 \
    --use_weight_only \
    --weight_only_precision ${weight_only_precision}
