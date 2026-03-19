git clone -b qwen3_omni https://github.com/wangxiongts/vllm.git
cd vllm
uv pip install -r requirements/build.txt
uv pip install -r requirements/cuda.txt

# Pre-built does not work

unset VLLM_USE_PRECOMPILED
unset VLLM_PRECOMPILED_WHEEL_LOCATION
export MAX_JOBS=24
export NVCC_THREADS=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6" #Targeting only A100 and A40 and leaving out H100 (makes it faster)

uv pip install -e . -v --no-build-isolation

pip install git+https://github.com/huggingface/transformers
pip install accelerate
pip install qwen-omni-utils -U
pip install -U flash-attn --no-build-isolation

Issues with the vllm branch:

sed -i 's/vision_config: DeepseekVLV2VisionConfig/vision_config: DeepseekVLV2VisionConfig = None/' vllm/transformers_utils/configs/deepseek_vl2.py

sed -i 's/    vision_config: VisionEncoderConfig/    vision_config: VisionEncoderConfig = None/' vllm/transformers_utils/configs/deepseek_vl2.py
sed -i 's/    projector_config: MlpProjectorConfig/    projector_config: MlpProjectorConfig = None/' vllm/transformers_utils/configs/deepseek_vl2.py
sed -i '917,922c\    def get_max_image_tokens(self) -> int:\n        hf_processor = self.info.get_hf_processor()\n        if not hasattr(hf_processor, "image_processor") or getattr(hf_processor.image_processor, "max_pixels", None) is None:\n            return 0\n        target_width, target_height = self.get_image_size_with_most_features()\n        return self.get_num_image_tokens(\n            image_width=target_width,\n            image_height=target_height,\n            image_processor=None,\n        )' vllm/model_executor/models/qwen2_vl.py
sed -i '927,928d' vllm/model_executor/models/qwen2_vl.py
sed -i 's/hf_processor = self.info.get_hf_processor()/hf_processor = self.get_hf_processor()/' vllm/model_executor/models/qwen2_vl.py
sed -i '908,914c\    def get_image_size_with_most_features(self) -> ImageSize:\n        hf_processor = self.get_hf_processor()\n        if not hasattr(hf_processor, "image_processor") or getattr(hf_processor.image_processor, "max_pixels", None) is None:\n            return ImageSize(width=0, height=0)\n        max_image_size, _ = self._get_vision_info(\n            image_width=9999999,\n            image_height=9999999,\n            image_processor=None,\n        )\n        return max_image_size' vllm/model_executor/models/qwen2_vl.py
sed -i '918d' vllm/model_executor/models/qwen2_vl.py


vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8901 --host 127.0.0.1 --dtype bfloat16 --max-model-len 24000 --allowed-local-media-path / -tp 2 
