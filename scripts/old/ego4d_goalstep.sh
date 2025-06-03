# {
# for video_chunk_sec in 8 12
# do
#     mkdir -p outputs/internvl2_5-8b/ego4d_goalstep
#     CUDA_VISIBLE_DEVICES=4 \
#     python -u inference.py \
#     --llm_pretrained OpenGVLab/InternVL2_5-8B --bf16 true \
#     --input_dir /share3/public_share/Ego4D/v2/360_sec-clip_540ss_6fps --frame_fps 1 --max_num_frames 400 \
#     --video_chunk_sec ${video_chunk_sec} \
#     --test_fname ../qa_generate/outputs/ego4d_goalstep/online-val.json \
#     --output_fname outputs/internvl2_5-8b/ego4d_goalstep/${video_chunk_sec}sec.jsonl \
#     > outputs/internvl2_5-8b/ego4d_goalstep/${video_chunk_sec}sec.log 2>&1 &
#     wait

#     python reformat_to_mmduet_format.py \
#     --input_fname outputs/internvl2_5-8b/ego4d_goalstep/${video_chunk_sec}sec.jsonl \
#     --output_fname outputs/internvl2_5-8b/ego4d_goalstep/${video_chunk_sec}sec-pred.jsonl
# done
# } &

# {
# for video_chunk_sec in 8
# do
#     mkdir -p outputs/llava-ov/ego4d_goalstep
#     CUDA_VISIBLE_DEVICES=1 \·
#     python -u inference.py \
#     --llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov --bf16 true \
#     --input_dir /share3/public_share/Ego4D/v2/360_sec-clip_540ss_6fps --frame_fps 1 --max_num_frames 400 \
#     --video_chunk_sec ${video_chunk_sec} \
#     --test_fname ../qa_generate/outputs/ego4d_goalstep/online-val.json \
#     --output_fname outputs/llava-ov/ego4d_goalstep/${video_chunk_sec}sec.jsonl \
#     > outputs/llava-ov/ego4d_goalstep/${video_chunk_sec}sec.log 2>&1 &
#     wait

#     python reformat_to_mmduet_format.py \
#     --input_fname outputs/llava-ov/ego4d_goalstep/${video_chunk_sec}sec.jsonl \
#     --output_fname outputs/llava-ov/ego4d_goalstep/${video_chunk_sec}sec-pred.jsonl
# done
# } &


# --------------------
# openai api and gemini api
# --------------------
# python reformat_to_mmduet_format.py \
#     --input_fname outputs/openai_api/ego4d_goalstep/gpt-4.1-mini-8sec.jsonl \
#     --output_fname outputs/openai_api/ego4d_goalstep/gpt-4.1-mini-8sec-pred.jsonl

# python reformat_to_mmduet_format.py \
#     --input_fname outputs/gemini_api/ego4d_goalstep/gemini-2.0-flash-8sec.jsonl \
#     --output_fname outputs/gemini_api/ego4d_goalstep/gemini-2.0-flash-8sec-pred.jsonl
