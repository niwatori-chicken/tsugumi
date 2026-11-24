"""Inkling 2 Mini (thinkingmachines/Inkling-2-Mini, model_type inkling2_loop) on a single Blackhole card.

24-layer MoE decoder (128 experts, top 8) whose middle 12 layers run three times with shared weights (48 layer
applications per token), hybrid sliding (1024) / full attention with a learned relative position bias, attention
sinks, short causal convolutions and attention residuals over block states.

- config.py: the model configuration (config.json text_config)
- weights.py: checkpoint reader (NVFP4 / MXFP8 experts dequantized on load)
- ref.py: CPU reference (fp32, follows transformers' modeling_inkling2_loop.py) for checking the device engine
- engine.py: the ttnn engine (decode, one token per step)
- chat.py: prompt rendering and parsing of the reply (thinking / text / tool calls)
"""
