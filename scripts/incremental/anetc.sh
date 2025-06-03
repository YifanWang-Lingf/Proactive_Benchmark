
{
mkdir -p outputs/llava-ov/anetc/incremental
for video_chunk_sec in 16
do
    CUDA_VISIBLE_DEVICES=2 \
    python -u inference_incremental_chunks.py \
    --llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov --bf16 true \
    --input_dir /share3/public_share/ActivityNet/videos/mp4_videos --frame_fps 1 --max_num_frames 100 \
    --video_chunk_sec ${video_chunk_sec} \
    --test_fname ../judge_questions/outputs/anetc/anetc_val-with_judge_questions-500_examples.json \
    --output_fname outputs/llava-ov/anetc/incremental/${video_chunk_sec}sec.jsonl \
    > outputs/llava-ov/anetc/incremental/${video_chunk_sec}sec.log 2>&1 &
    wait

    python reformat_to_mmduet_format.py \
    --input_fname outputs/llava-ov/anetc/incremental/${video_chunk_sec}sec.jsonl \
    --output_fname outputs/llava-ov/anetc/incremental/${video_chunk_sec}sec-pred.jsonl
done
} &
