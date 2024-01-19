# model_path=../../../hf-models/mistral/7b-v0.1/
model_path=../../../hf-models/llama2/7b_hf
gptq_export_dir=${model_path}/gptq # https://huggingface.co/TheBloke/Llama-2-7B-AWQ is hf format
build_file=../../../../examples/llama/build.py
weight_only_precision='int4_gptq' # ['int8', 'int4', 'int4_awq', 'int4_gptq']

if [ ! -d ${gptq_export_dir} ]; then
    echo "${gptq_export_dir} does not exist. Quantize HF LLaMA 7B checkpoint into INT4 GPTQ format"
    python run_gptq.py
fi

python ${build_file} --model_dir ${model_path} \
    --dtype float16 \
    --use_gpt_attention_plugin float16 \
    --remove_input_padding \
    --enable_context_fmha \
    --world_size 1 \
    --tp_size 1 \
    --parallel_build \
    --max_batch_size 64 \
    --max_input_len 2048 \
    --max_output_len 2048 \
    --output_dir ${model_path}/trt_engines/w4a16_gptq/1-gpu/ \
    --quant_ckpt_path ${gptq_export_dir}/gptq_model-4bit-128g.safetensors\
    --per_group \
    --use_weight_only \
    --weight_only_precision ${weight_only_precision}

