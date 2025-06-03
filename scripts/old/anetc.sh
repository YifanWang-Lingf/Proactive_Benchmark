# mkdir -vp outputs/internvl2_5-8b/anetc
# GPUS=(3 3)
# VIDEO_CHUNK_SECS=(8 12)
# for i in 0 1
# do
#     {
#     video_chunk_sec=${VIDEO_CHUNK_SECS[i]}
#     CUDA_VISIBLE_DEVICES=${GPUS[i]} \
#     python -u inference.py --need_judge_answerable 0 \
#     --llm_pretrained OpenGVLab/InternVL2_5-8B --bf16 true \
#     --input_dir /share3/public_share/ActivityNet/videos/mp4_videos --frame_fps 1 --max_num_frames 400 \
#     --video_chunk_sec ${video_chunk_sec} \
#     --test_fname ../qa_generate/outputs/anetc/val-online.json \
#     --output_fname outputs/internvl2_5-8b/anetc/${video_chunk_sec}sec.jsonl \
#     > outputs/internvl2_5-8b/anetc/${video_chunk_sec}sec.log 2>&1

#     python reformat_to_mmduet_format.py \
#     --input_fname outputs/internvl2_5-8b/anetc/${video_chunk_sec}sec.jsonl \
#     --output_fname outputs/internvl2_5-8b/anetc/${video_chunk_sec}sec-pred.jsonl
#     } &
# done

# mkdir -vp outputs/llava-ov/anetc
# GPUS=(5 5)
# VIDEO_CHUNK_SECS=(8 12)
# for i in 0 1
# do
#     {
#     video_chunk_sec=${VIDEO_CHUNK_SECS[i]}
#     CUDA_VISIBLE_DEVICES=${GPUS[i]} \
#     python -u inference.py --need_judge_answerable 0 \
#     --llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov --bf16 true \
#     --input_dir /share3/public_share/ActivityNet/videos/mp4_videos --frame_fps 1 --max_num_frames 400 \
#     --video_chunk_sec ${video_chunk_sec} \
#     --test_fname ../qa_generate/outputs/anetc/val-online.json \
#     --output_fname outputs/llava-ov/anetc/${video_chunk_sec}sec.jsonl \
#     > outputs/llava-ov/anetc/${video_chunk_sec}sec.log 2>&1

#     python reformat_to_mmduet_format.py \
#     --input_fname outputs/llava-ov/anetc/${video_chunk_sec}sec.jsonl \
#     --output_fname outputs/llava-ov/anetc/${video_chunk_sec}sec-pred.jsonl
#     } &
# done

# {
# mkdir -vp outputs/internvl2_5-8b/anetc
# GPUS=(0 1 2 3)
# video_chunk_sec=16
# interval=250
# for i in 0 1 2 3
# do
#     CUDA_VISIBLE_DEVICES=${GPUS[i]} \
#     python -u inference.py --need_judge_answerable 0 \
#     --start_idx $((i*interval)) --end_idx $((i*interval+interval)) \
#     --llm_pretrained OpenGVLab/InternVL2_5-8B --bf16 true \
#     --input_dir /share3/public_share/ActivityNet/videos/mp4_videos --frame_fps 1 --max_num_frames 400 \
#     --video_chunk_sec ${video_chunk_sec} \
#     --test_fname ../qa_generate/outputs/anetc/val-online.json \
#     --output_fname outputs/internvl2_5-8b/anetc/${video_chunk_sec}sec.jsonl.$i \
#     > outputs/internvl2_5-8b/anetc/${video_chunk_sec}sec.log.$i 2>&1 &
# done
# wait

# cat outputs/internvl2_5-8b/anetc/${video_chunk_sec}sec.jsonl.* > outputs/internvl2_5-8b/anetc/${video_chunk_sec}sec.jsonl

# python reformat_to_mmduet_format.py \
#     --input_fname outputs/internvl2_5-8b/anetc/${video_chunk_sec}sec.jsonl \
#     --output_fname outputs/internvl2_5-8b/anetc/${video_chunk_sec}sec-pred.jsonl
# } &

# {
# mkdir -vp outputs/llava-ov/anetc
# GPUS=(4 5 6 7)
# video_chunk_sec=16
# interval=250
# for i in 0 1 2 3
# do
#     CUDA_VISIBLE_DEVICES=${GPUS[i]} \
#     python -u inference.py --need_judge_answerable 0 \
#     --start_idx $((i*interval)) --end_idx $((i*interval+interval)) \
#     --llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov --bf16 true \
#     --input_dir /share3/public_share/ActivityNet/videos/mp4_videos --frame_fps 1 --max_num_frames 400 \
#     --video_chunk_sec ${video_chunk_sec} \
#     --test_fname ../qa_generate/outputs/anetc/val-online.json \
#     --output_fname outputs/llava-ov/anetc/${video_chunk_sec}sec.jsonl.$i \
#     > outputs/llava-ov/anetc/${video_chunk_sec}sec.log.$i 2>&1 &
# done
# wait

# cat outputs/llava-ov/anetc/${video_chunk_sec}sec.jsonl.* > outputs/llava-ov/anetc/${video_chunk_sec}sec.jsonl

# python reformat_to_mmduet_format.py \
#     --input_fname outputs/llava-ov/anetc/${video_chunk_sec}sec.jsonl \
#     --output_fname outputs/llava-ov/anetc/${video_chunk_sec}sec-pred.jsonl
# } &
