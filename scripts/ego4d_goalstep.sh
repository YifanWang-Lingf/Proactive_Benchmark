# {
# GPUS=(0 1)
# interval=500
# for video_chunk_sec in 2 # 3 4
# do
#     for i in 0 1
#     do
#     # ----- 首先先跑inference -----
#     CUDA_VISIBLE_DEVICES=${GPUS[i]} \
#     python -u inference_independent_chunks.py \
#     --start_idx $((i*interval)) --end_idx $((i*interval+interval)) \
#     --llm_pretrained OpenGVLab/InternVL2_5-8B --bf16 true \
#     --input_dir ../qa_generate/outputs/ego4d_goalstep/videos --frame_fps 1 --max_num_frames 400 \
#     --video_chunk_sec ${video_chunk_sec} \
#     --test_fname ../qa_generate/outputs/ego4d_goalstep/0502_version-anno.json \
#     --output_fname outputs/internvl2_5-8b/ego4d_goalstep/${video_chunk_sec}sec.jsonl.$i \
#     > outputs/internvl2_5-8b/ego4d_goalstep/${video_chunk_sec}sec.log.$i 2>&1 &
#     done
#     wait

#     cat outputs/internvl2_5-8b/ego4d_goalstep/${video_chunk_sec}sec.jsonl.* > outputs/internvl2_5-8b/ego4d_goalstep/${video_chunk_sec}sec.jsonl

#     # ----- 然后转换一下上一个阶段输出的结果的格式，之后的evaluate阶段会统一用这种格式 -----
#     python reformat_to_mmduet_format.py \
#     --input_fname outputs/internvl2_5-8b/ego4d_goalstep/${video_chunk_sec}sec.jsonl \
#     --output_fname outputs/internvl2_5-8b/ego4d_goalstep/${video_chunk_sec}sec-pred.jsonl
# done
# } &

#########################################################################################################################

# PS MLLM的一些选择问题：
# "Qwen/Qwen2.5-VL-7B-Instruct" (当前shell中使用的)（貌似这个是主打的，镜像站只支持这个）或 "Qwen/Qwen2.5-VL-7B"
# "lmms-lab/LongVA-7B"（当前shell中使用的） 或 "lmms-lab/LongVA-7B-DPO"
# "internlm/internlm-xcomposer2d5-7b" (当前shell中使用的) 或 "internlm/internlm-xcomposer2d5-7b-chat"（对话微调版）


# -----下面的脚本是将终端阻塞以执行完所有任务，同时将输出显示在终端中-----
# -----强制将GPU编号按照物理编号执行，而不是按照算力排序-----
export CUDA_DEVICE_ORDER=PCI_BUS_ID
GPUS=(0 1)
interval=500

###### 模型文件夹名 ###### 
# model_dirname ="internvl2_5-8b" 
model_dirname="llava-ov"  
# model_dirname="qwen2_5_vl_7b" 
# model_dirname="longva_7b"
# model_dirname="internlm_xcomposer_2_5"
######################### 


for video_chunk_sec in 2    # 3 4
do
  for i in 0 1
  do
    # 指定每个GPU运行时日志记录文件名称，日志文件名例如：2sec.log.i，一个GPU一个文件名
    logfile="outputs/${model_dirname}/ego4d_goalstep/${video_chunk_sec}sec.log.$i"

    # ----- 首先先跑 inference -----
    ###### MLLM名称 ###### 
    # --llm_pretrained OpenGVLab/InternVL2_5-8B --bf16 true 
    # --llm_pretrained llava-hf/llava-onevision-qwen2-7b-ov-hf --bf16 true 
    # --llm_pretrained Qwen/Qwen2.5-VL-7B-Instruct --bf16 true 
    # --llm_pretrained lmms-lab/LongVA-7B --bf16 true
    # --llm_pretrained internlm/internlm-xcomposer2d5-7b --bf16 true
    #########################

    CUDA_VISIBLE_DEVICES=${GPUS[i]} python -u inference_independent_chunks.py \
        --start_idx $((i*interval)) --end_idx $((i*interval+interval)) \
        --llm_pretrained llava-hf/llava-onevision-qwen2-7b-ov-hf --bf16 true \
        --input_dir ../qa_generate/outputs/ego4d_goalstep/videos --frame_fps 1 --max_num_frames 400 \
        --video_chunk_sec ${video_chunk_sec} \
        --test_fname ../qa_generate/outputs/ego4d_goalstep/0502_version-anno.json \
        --output_fname outputs/${model_dirname}/ego4d_goalstep/${video_chunk_sec}sec.jsonl.$i \
    |& tee -a "$logfile"      

  done
  wait                       
  # 将上面的GPU输出文件合并为一个文件,例如 2sec.jsonl 

  cat outputs/${model_dirname}/ego4d_goalstep/${video_chunk_sec}sec.jsonl.* \
      > outputs/${model_dirname}/ego4d_goalstep/${video_chunk_sec}sec.jsonl


  # ----- 转格式 ----- 
  # 将上一个阶段输出的结果的格式转换为MMDuet格式，之后的evaluate阶段会统一用这种格式 
  # 注意这里的输出文件名是 2sec-pred.jsonl 

  python reformat_to_mmduet_format.py \
      --input_fname  outputs/${model_dirname}/ego4d_goalstep/${video_chunk_sec}sec.jsonl \
      --output_fname outputs/${model_dirname}/ego4d_goalstep/${video_chunk_sec}sec-pred.jsonl \
  |& tee -a "outputs/${model_dirname}/ego4d_goalstep/${video_chunk_sec}sec.reformat.log"

done



#########################################################################################################################