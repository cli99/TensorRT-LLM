# model_path=../../../hf-models/mistral/7b-v0.1/
model_path=../../../hf-models/llama2/7b_hf
awq_export_dir=${model_path}/awq/4bit-gs128-awq.pt # https://huggingface.co/TheBloke/Llama-2-7B-AWQ is hf format
build_file=../../../../examples/llama/build.py
quantize_file=../../../../examples/quantization/quantize.py
weight_only_precision='int4_awq' # ['int8', 'int4', 'int4_awq', 'int4_gptq']

if [ ! -d ${awq_export_dir} ]; then
    echo "${awq_export_dir} does not exist. Quantize HF LLaMA 7B checkpoint into INT4 AWQ format"
    python ${quantize_file} --model_dir ${model_path} \
                    --dtype float16 \
                    --qformat int4_awq \
                    --export_path ${awq_export_dir} \
                    --calib_size 32
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
    --output_dir ${model_path}/trt_engines/w4a16_awq/1-gpu/ \
    --quant_ckpt_path ${awq_export_dir}/llama_tp1_rank0.npz \
    --per_group \
    --use_weight_only \
    --weight_only_precision ${weight_only_precision}

