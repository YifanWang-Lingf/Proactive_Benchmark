# {
# mkdir -vp outputs/internvl2_5-8b/tvqa
# GPUS=(0 1)
# interval=500
# for video_chunk_sec in 3 4 5
# do
#     for i in 0 1
#     do
#     CUDA_VISIBLE_DEVICES=${GPUS[i]} \
#     python -u inference.py \
#     --start_idx $((i*interval)) --end_idx $((i*interval+interval)) \
#     --llm_pretrained OpenGVLab/InternVL2_5-8B --bf16 true \
#     --input_dir /share3/public_share/TVQA/video/videos_val --frame_fps 2 --max_num_frames 400 \
#     --video_chunk_sec ${video_chunk_sec} \
#     --test_fname ../human_annotation/outputs/tvqa/tvqa-val-online-1question-open_ended-long_answers.json \
#     --output_fname outputs/internvl2_5-8b/tvqa/${video_chunk_sec}sec-wo_sub.jsonl.$i \
#     > outputs/internvl2_5-8b/tvqa/${video_chunk_sec}sec-wo_sub.log.$i 2>&1 &
#     done
#     wait

#     cat outputs/internvl2_5-8b/tvqa/${video_chunk_sec}sec-wo_sub.jsonl.* > outputs/internvl2_5-8b/tvqa/${video_chunk_sec}sec-wo_sub.jsonl

#     python reformat_to_mmduet_format.py \
#     --input_fname outputs/internvl2_5-8b/tvqa/${video_chunk_sec}sec-wo_sub.jsonl \
#     --output_fname outputs/internvl2_5-8b/tvqa/${video_chunk_sec}sec-wo_sub-pred.jsonl
# done
# } &

# {
# mkdir -vp outputs/internvl2_5-8b/tvqa
# GPUS=(2 3)
# interval=500
# for video_chunk_sec in 3 4 5
# do
#     for i in 0 1
#     do
#     CUDA_VISIBLE_DEVICES=${GPUS[i]} \
#     python -u inference.py \
#     --start_idx $((i*interval)) --end_idx $((i*interval+interval)) \
#     --llm_pretrained OpenGVLab/InternVL2_5-8B --bf16 true \
#     --input_dir /share3/public_share/TVQA/video/videos_val --frame_fps 2 --max_num_frames 400 \
#     --video_chunk_sec ${video_chunk_sec} \
#     --test_fname ../human_annotation/outputs/tvqa/tvqa-val-online-1question-open_ended-long_answers-with_sub.json \
#     --output_fname outputs/internvl2_5-8b/tvqa/${video_chunk_sec}sec-with_sub.jsonl.$i \
#     > outputs/internvl2_5-8b/tvqa/${video_chunk_sec}sec-with_sub.log.$i 2>&1 &
#     done
#     wait

#     cat outputs/internvl2_5-8b/tvqa/${video_chunk_sec}sec-with_sub.jsonl.* > outputs/internvl2_5-8b/tvqa/${video_chunk_sec}sec-with_sub.jsonl

#     python reformat_to_mmduet_format.py \
#     --input_fname outputs/internvl2_5-8b/tvqa/${video_chunk_sec}sec-with_sub.jsonl \
#     --output_fname outputs/internvl2_5-8b/tvqa/${video_chunk_sec}sec-with_sub-pred.jsonl
# done
# } &


# {
# mkdir -vp outputs/llava-ov/tvqa
# GPUS=(4 5)
# interval=500
# for video_chunk_sec in 3 4 5
# do
#     for i in 0 1
#     do
#     CUDA_VISIBLE_DEVICES=${GPUS[i]} \
#     python -u inference.py \
#     --start_idx $((i*interval)) --end_idx $((i*interval+interval)) \
#     --llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov --bf16 true \
#     --input_dir /share3/public_share/TVQA/video/videos_val --frame_fps 2 --max_num_frames 400 \
#     --video_chunk_sec ${video_chunk_sec} \
#     --test_fname ../human_annotation/outputs/tvqa/tvqa-val-online-1question-open_ended-long_answers.json \
#     --output_fname outputs/llava-ov/tvqa/${video_chunk_sec}sec-wo_sub.jsonl.$i \
#     > outputs/llava-ov/tvqa/${video_chunk_sec}sec-wo_sub.log.$i 2>&1 &
#     done
#     wait

#     cat outputs/llava-ov/tvqa/${video_chunk_sec}sec-wo_sub.jsonl.* > outputs/llava-ov/tvqa/${video_chunk_sec}sec-wo_sub.jsonl

#     python reformat_to_mmduet_format.py \
#     --input_fname outputs/llava-ov/tvqa/${video_chunk_sec}sec-wo_sub.jsonl \
#     --output_fname outputs/llava-ov/tvqa/${video_chunk_sec}sec-wo_sub-pred.jsonl
# done
# } &

# {
# mkdir -vp outputs/llava-ov/tvqa
# GPUS=(6 7)
# interval=500
# for video_chunk_sec in 3 4 5
# do
#     for i in 0 1
#     do
#     CUDA_VISIBLE_DEVICES=${GPUS[i]} \
#     python -u inference.py \
#     --start_idx $((i*interval)) --end_idx $((i*interval+interval)) \
#     --llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov --bf16 true \
#     --input_dir /share3/public_share/TVQA/video/videos_val --frame_fps 2 --max_num_frames 400 \
#     --video_chunk_sec ${video_chunk_sec} \
#     --test_fname ../human_annotation/outputs/tvqa/tvqa-val-online-1question-open_ended-long_answers-with_sub.json \
#     --output_fname outputs/llava-ov/tvqa/${video_chunk_sec}sec-with_sub.jsonl.$i \
#     > outputs/llava-ov/tvqa/${video_chunk_sec}sec-with_sub.log.$i 2>&1 &
#     done
#     wait

#     cat outputs/llava-ov/tvqa/${video_chunk_sec}sec-with_sub.jsonl.* > outputs/llava-ov/tvqa/${video_chunk_sec}sec-with_sub.jsonl

#     python reformat_to_mmduet_format.py \
#     --input_fname outputs/llava-ov/tvqa/${video_chunk_sec}sec-with_sub.jsonl \
#     --output_fname outputs/llava-ov/tvqa/${video_chunk_sec}sec-with_sub-pred.jsonl
# done
# } &

