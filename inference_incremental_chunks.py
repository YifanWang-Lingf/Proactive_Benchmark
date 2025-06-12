import os, json, math
from tqdm import tqdm
from transformers import HfArgumentParser
from dataclasses import dataclass

from models import parse_args
from models.arguments_live import LiveTestArguments
from baselines import load_model, inference
from utils.video_utils import get_video_length_sec


@dataclass
class ClipTestArguments(LiveTestArguments):
    video_chunk_sec: int = 3


if __name__ == '__main__':
    args, = HfArgumentParser(ClipTestArguments).parse_args_into_dataclasses()
    print(args)

    if 'llava-onevision' in args.llm_pretrained.lower():
        model_type = 'llava_ov'
    elif 'internvl' in args.llm_pretrained.lower():
        model_type = 'internvl_2d5'
    
    elif 'openai_api' == args.llm_pretrained:
        model_type = 'openai_api'
        
    # 新增 Qwen2.5 VL 7B模型支持
    elif 'qwen2.5-vl' in args.llm_pretrained.lower():
        model_type = 'qwen2_5_vl_7b'
    # 新增 LongAV 7B模型支持
    elif 'longva' in args.llm_pretrained.lower():
        model_type = 'longva_7b'
    # 新增 InternLM XComposer 2.5模型支持
    elif 'internlm' in args.llm_pretrained.lower():
        model_type = 'internlm_xcomposer_2.5'
    else:
        raise NotImplementedError("Unkown model type, please check the llm_pretrained argument")
    print("model_type:", model_type)
    baseline_model = load_model(model_type, args.llm_pretrained, args.attn_implementation)

    if os.path.exists(args.output_fname):
        tested_examples = [json.loads(line) for line in open(args.output_fname)]
        tested_questions = set([(e['question_id'], e['video_span'][0], e['video_span'][1]) for e in tested_examples])
    else:
        tested_questions = set()
    f_out = open(args.output_fname, 'a')
    test_data = json.load(open(args.test_fname))[args.start_idx: args.end_idx]

    for example_i, example in tqdm(enumerate(test_data), total=len(test_data)):
        try:
            video_length_sec = get_video_length_sec(os.path.join(args.input_dir, example['video']))
        except:
            print(f"get video length sec error for {example['question_id']}, skipping")
            continue

        last_reply_end_time = example['answer'][-1]['reply_timespan'][1]
        previous_turns_output = ''
        start_sec = 0
        for new_chunk_start_sec in range(0, int(video_length_sec), args.video_chunk_sec):
            if new_chunk_start_sec > last_reply_end_time: break
            end_sec = min(video_length_sec, new_chunk_start_sec + args.video_chunk_sec)
            if (example['question_id'], start_sec, end_sec) in tested_questions:
                print(f"question {example['question_id']} {start_sec} {end_sec} has been tested, skipping")
                continue

            original_question = example['conversation'][0]['content']
            additional_text_input = ""
            for input_turns in example['conversation']:
                if start_sec < input_turns['time'] <= end_sec:
                    additional_text_input += f"{input_turns['content']}\n"

            # if the frames are more than args.max_num_frames, we can change the frame_fps
            num_frames = int((end_sec - start_sec) * args.frame_fps)
            if num_frames > args.max_num_frames:
                change_ratio = math.ceil(num_frames / args.max_num_frames)
                frame_fps = args.frame_fps / change_ratio
            else:
                frame_fps = args.frame_fps

            video_data = {
                'video_file': os.path.join(args.input_dir, example['video']),
                'start_sec': start_sec, 'end_sec': end_sec,
                'frame_resolution': args.frame_resolution, 'frame_fps': frame_fps
            }
            text_data = {
                'original_question': original_question,
                'additional_text_input': additional_text_input,
                'previous_turns_output': previous_turns_output
            }

            try:
                response_dict = inference(
                    baseline_model=baseline_model,
                    model_type=model_type, video_data=video_data, text_data=text_data,
                    need_judge_answerable=False, debug_print=example_i < 5)

                previous_turns_output += response_dict['response']

                res = {
                    'video': example['video'], 'question_id': example['question_id'], 'video_span': [start_sec, end_sec],
                    'answerable': response_dict['answerable'], 'model_response': response_dict['response'],
                    'question': original_question, 'frame_fps': frame_fps
                }
                f_out.write(json.dumps(res) + '\n')
                f_out.flush()
            except Exception as e:
                print(f"error for {example['question_id']}, {start_sec} {end_sec}")
                # raise e
                break

    f_out.close()
