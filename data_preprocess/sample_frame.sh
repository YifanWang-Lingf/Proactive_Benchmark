# python sample_frame.py --fps 2 --num_workers 1 \
#     --anno_fname /share2/wangyq/projects/online_videollm/proactive_benchmark/qa_generate/outputs/magqa/0502_version-anno.json \
#     --src_video_folder /share4/wangyq/resources/shot2story/shot2story-videos/magqa_test_videos \
#     --dest_frame_folder /share3/wangyq/resources/proactive_benchmark_frames/magqa-2fps \
#     > nohup/sample_frame-magqa.log 2>&1 &

# python sample_frame.py --fps 1 \
#     --anno_fname /share2/wangyq/projects/online_videollm/proactive_benchmark/qa_generate/outputs/ego4d_goalstep/0502_version-anno.json \
#     --src_video_folder /share3/public_share/Ego4D/v2/360_sec-clip_540ss_6fps \
#     --dest_frame_folder /share3/wangyq/resources/proactive_benchmark_frames/ego4d_goalstep-1fps-grp \
#     >> nohup/sample_frame-ego4d_goalstep.log 2>&1 &

# python sample_frame.py --fps 1 \
#     --anno_fname /share2/wangyq/projects/online_videollm/proactive_benchmark/qa_generate/outputs/anetc/0502_version-anno.json \
#     --src_video_folder /share3/public_share/ActivityNet/videos/mp4_videos \
#     --dest_frame_folder /share3/wangyq/resources/proactive_benchmark_frames/anetc-1fps \
#     > nohup/sample_frame-anetc.log 2>&1 &

# python sample_frame.py --fps 1 \
#     --anno_fname /share2/wangyq/projects/online_videollm/proactive_benchmark/qa_generate/outputs/tvqa/0502_version-anno.json \
#     --src_video_folder /share3/public_share/TVQA/video/videos_val \
#     --dest_frame_folder /share3/wangyq/resources/proactive_benchmark_frames/tvqa-1fps-new_videos \
#     > nohup/sample_frame-tvqa.log 2>&1 &

# python sample_frame.py --fps 1 \
#     --anno_fname /share2/wangyq/projects/online_videollm/proactive_benchmark/qa_generate/outputs/ucfa/0502_version-anno.json \
#     --src_video_folder /share3/public_share/UCF_Crime_Annotation/test_anomaly_videos \
#     --dest_frame_folder /share3/wangyq/resources/proactive_benchmark_frames/ucfa-1fps \
#     > nohup/sample_frame-ucfa.log 2>&1 &

