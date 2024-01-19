# sudo apt-get update && apt-get -y install git git-lfs

# To build the TensorRT-LLM code.
python3 ./scripts/build_wheel.py --benchmarks --cuda_architectures "89-real" --clean --trt_root /usr/local/tensorrt

# Deploy TensorRT-LLM in your environment.
# pip install ./build/tensorrt_llm-0.7.1-cp310-cp310-linux_x86_64.whl


