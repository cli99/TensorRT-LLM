model_path=../../../hf-models/llama2/7b_hf
build_file=../../../../examples/llama/build.py
weight_only_precision='int8' # ['int8', 'int4', 'int4_awq', 'int4_gptq']
tp_size=1
sq_output_dir=${model_path}/smooth_quant/sq0.8/

if [ ! -d ${sq_output_dir} ]; then
    echo "${sq_output_dir} does not exist. Smooth quantize HF LLaMA 7B checkpoint into INT8 format"
    # python3 ../../hf_llama_convert.py -i ${model_path} -o ${sq_output_dir} -sq 0.8 --tensor-parallelism ${tp_size} --storage-type fp16
fi


python ${build_file} --bin_model_dir ${sq_output_dir}/1-gpu/ \
    --dtype float16 \
    --use_gpt_attention_plugin float16 \
    --output_dir ${model_path}/trt_engines/w8a8sq/1-gpu/ \
    --remove_input_padding \
    --enable_context_fmha \
    --world_size 1 \
    --tp_size ${tp_size} \
    --parallel_build \
    --max_batch_size 64 \
    --max_input_len 2048 \
    --max_output_len 2048 \
    --use_smooth_quant \
    --per_token \
    --per_channel