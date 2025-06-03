# video_chunk_sec=2
# model=gpt-4.1-mini
# mkdir -p outputs/openai_api/magqa
# python -u inference_independent_chunks.py \
#     --llm_pretrained ${model} \
#     --input_dir http://139.59.121.220/proactive_benchmark_frames/magqa-2fps --frame_fps 2 \
#     --video_chunk_sec ${video_chunk_sec} \
#     --test_fname data/magqa/0502_version-anno.json \
#     --output_fname outputs/${model}/magqa/${video_chunk_sec}sec.jsonl \
#     > outputs/${model}/magqa/${video_chunk_sec}sec.log 2>&1 &

# video_chunk_sec=2
# model=gemini-2.0-flash
# mkdir -p outputs/gemini_api/magqa
# python -u inference_independent_chunks.py \
#     --llm_pretrained ${model} \
#     --input_dir /var/www/html/proactive_benchmark_frames/magqa-2fps --frame_fps 2 \
#     --video_chunk_sec ${video_chunk_sec} \
#     --test_fname data/magqa/0502_version-anno.json \
#     --output_fname outputs/${model}/magqa/${video_chunk_sec}sec.jsonl \
#     > outputs/${model}/magqa/${video_chunk_sec}sec.log 2>&1 &