import copy, json, random, os, cv2, math
import torch
import numpy as np
from peft import PeftModel
from tqdm import tqdm
from transformers import HfArgumentParser, AutoModel, AutoTokenizer
from torch.utils.data import Dataset
from dataclasses import dataclass

from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from models.arguments_live import LiveTestArguments
import pseudo_proactive_eval.utils.internvl_utils as internvl_utils


@dataclass
class ClipTestArguments(LiveTestArguments):
    video_chunk_sec: int = 3
    need_judge_answerable: int = 1


def get_video_length_sec(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = frame_count / input_fps
    cap.release()
    return video_duration


def load_video(video_file, start_sec=0, end_sec=100000000000, output_resolution=384, output_fps=2):
    cap = cv2.VideoCapture(video_file)
    # Get original video properties
    pad_color = (0, 0, 0)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = frame_count / input_fps
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_width = output_height = output_resolution

    start_sec, end_sec = max(0, start_sec), min(video_duration, end_sec)
    start_frame, end_frame = start_sec * input_fps, end_sec * input_fps
    num_frames_total = math.ceil((end_sec - start_sec) * output_fps)
    frame_sec = [(i / output_fps) + start_sec for i in range(num_frames_total)]
    frame_list, cur_time, frame_index = [], start_sec, 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index < len(frame_sec) and cur_time >= frame_sec[frame_index]:
            if input_width > input_height:
                # Landscape video: scale width to the resolution, adjust height
                new_width = output_resolution
                new_height = int((input_height / input_width) * output_resolution)
            else:
                # Portrait video: scale height to the resolution, adjust width
                new_height = output_resolution
                new_width = int((input_width / input_height) * output_resolution)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            # pad the frame
            canvas = cv2.copyMakeBorder(
                resized_frame,
                top=(output_height - new_height) // 2,
                bottom=(output_height - new_height + 1) // 2,
                left=(output_width - new_width) // 2,
                right=(output_width - new_width + 1) // 2,
                borderType=cv2.BORDER_CONSTANT,
                value=pad_color
            )
            frame_list.append(np.transpose(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), (2, 0, 1)))
            frame_index += 1
        cur_time += 1 / input_fps
        if cur_time > end_sec:
            break
    cap.release()
    return torch.tensor(np.stack(frame_list))



if __name__ == '__main__':
    args, = HfArgumentParser(ClipTestArguments).parse_args_into_dataclasses()
    print(args)

    if 'llava' in args.llm_pretrained.lower():
        model_type = 'llava'
    elif 'intern' in args.llm_pretrained.lower():
        model_type = 'intern'
    else:
        raise NotImplementedError

    if model_type == 'llava':
        # llava onevision baseline
        tokenizer, model, image_processor, max_length = load_pretrained_model(args.llm_pretrained, None, "llava_qwen", device_map="auto", attn_implementation=args.attn_implementation)  # Add any other thing you want to pass in llava_model_args
        model.eval()
    elif model_type == 'intern':
        # internvideo baseline
        model = AutoModel.from_pretrained(
            args.llm_pretrained, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, use_flash_attn=args.attn_implementation.startswith('flash'),
            trust_remote_code=True).eval().cuda()
        generation_config = dict(max_new_tokens=256, do_sample=True)
        tokenizer = AutoTokenizer.from_pretrained(args.llm_pretrained, trust_remote_code=True, use_fast=False)

    if args.lora_pretrained is not None:
        print(f"loading lora ckpt from {args.lora_pretrained}, and setting mm_spatial_pool_stride to {args.video_pooling_stride}")
        model = PeftModel.from_pretrained(model, args.lora_pretrained, is_trainable=False)
        model.config.mm_spatial_pool_stride = args.video_pooling_stride

    if os.path.exists(args.output_fname):
        tested_examples = [json.loads(line) for line in open(args.output_fname)]
        tested_questions = set([(e['question_id'], e['video_span'][0], e['video_span'][1]) for e in tested_examples])
    else:
        tested_questions = set()

    conv_template = "qwen_1_5"
    f_out = open(args.output_fname, 'a')
    test_data = json.load(open(args.test_fname))[args.start_idx: args.end_idx]
    for example_i, example in tqdm(enumerate(test_data), total=len(test_data)):
        try:
            video_length_sec = get_video_length_sec(os.path.join(args.input_dir, example['video']))
        except:
            print(f"get video length sec error for {example['question_id']}, skipping")
            continue

        last_reply_end_time = example['answer'][-1]['time'][1]
        for start_sec in range(0, int(video_length_sec), args.video_chunk_sec):
            if start_sec > last_reply_end_time: break
            end_sec = min(video_length_sec, start_sec + args.video_chunk_sec)
            if (example['question_id'], start_sec, end_sec) in tested_questions:
                print(f"question {example['question_id']} {start_sec} {end_sec} has been tested, skipping")
                continue

            original_question = example['conversation'][0]['content']

            # DEPRECATED: 这段代码考虑的是一个video播放时会在不同时间提出多个不同问题的情况。
            # 但是我们现在测试benchmark时先不考虑这种情况。
            # 相反，我们现在要考虑的是 "在video开始时提出一个问题，但随着video进展，会出现字幕"的tvqa benchmark的场景。
            # for turn in example['conversation']:       # 寻找在end_sec之前的最后一个问题
            #     if turn['time'] < end_sec:
            #         original_question = turn['content']

            additional_text_input = ""
            for input_turns in example['conversation']:
                if start_sec < input_turns['time'] <= end_sec:
                    additional_text_input += f"{input_turns['content']}\n"

            if model_type == 'llava':
                try:
                    video_frames = load_video(os.path.join(args.input_dir, example['video']), start_sec=start_sec, end_sec=end_sec, output_resolution=args.frame_resolution, output_fps=args.frame_fps)
                    image_sizes = [frame.size() for frame in video_frames]
                    image_tensor = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().to(model.device)
                    modalities = ["video"] * len(video_frames)

                    if args.need_judge_answerable:
                        if not additional_text_input:
                            answerable_question = f"{DEFAULT_IMAGE_TOKEN}\nIs the following question answerable by the video? Only reply with yes or no.\n{original_question}"
                        else:
                            answerable_question = f"{DEFAULT_IMAGE_TOKEN}\n{additional_text_input}\nIs the following question answerable by the video and subtites? Only reply with yes or no.\n{original_question}"

                        conv = copy.deepcopy(conv_templates[conv_template])
                        conv.append_message(conv.roles[0], answerable_question)
                        conv.append_message(conv.roles[1], None)
                        prompt_question = conv.get_prompt()
                        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
                        cont = model.generate(
                            input_ids, images=[image_tensor], image_sizes=image_sizes,
                            do_sample=False, temperature=0, max_new_tokens=256, modalities=modalities,
                        )
                        answerable_response = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
                        if example_i < 5:
                            print("answerable_question:", answerable_question)
                            print("answerable_response:", answerable_response)
                    else:
                        answerable_response = 'yes'

                    if 'yes' in answerable_response.lower():
                        if not additional_text_input:
                            question = f"{DEFAULT_IMAGE_TOKEN}\n{original_question}"
                        else:
                            question = f"{DEFAULT_IMAGE_TOKEN}\n{additional_text_input}\n{original_question}"

                        conv = copy.deepcopy(conv_templates[conv_template])
                        conv.append_message(conv.roles[0], question)
                        conv.append_message(conv.roles[1], None)
                        prompt_question = conv.get_prompt()
                        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
                        cont = model.generate(
                            input_ids, images=[image_tensor], image_sizes=image_sizes,
                            do_sample=False, temperature=0, max_new_tokens=256, modalities=modalities,
                        )
                        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
                        if example_i < 5:
                            print("question:", question)
                            print("text_outputs:", text_outputs)
                    else:
                        text_outputs = None

                    res = {
                        'video': example['video'], 'question_id': example['question_id'], 'video_span': [start_sec, end_sec],
                        'answerable': answerable_response, 'model_response': text_outputs, 'question': original_question}
                    f_out.write(json.dumps(res) + '\n')
                    f_out.flush()
                except Exception as e:
                    print(f"error at {example['question_id']} {start_sec}: {e}")

            elif model_type == 'intern':
                pixel_values, num_patches_list = internvl_utils.load_video(
                    os.path.join(args.input_dir, example['video']),
                    bound=(start_sec, end_sec),
                    num_segments=int((end_sec - start_sec) * args.frame_fps)
                )
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
                video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])

                if args.need_judge_answerable:
                    if not additional_text_input:
                        answerable_question = video_prefix + "Is the following question answerable by the video? Only reply with Yes or No.\n" + original_question
                    else:
                        answerable_question = video_prefix + " " + additional_text_input + "\nIs the following question answerable by the video and subtitles? Only reply with Yes or No.\n" + original_question
                    answerable_response, history = model.chat(tokenizer, pixel_values, answerable_question, generation_config,
                                                num_patches_list=num_patches_list, history=None, return_history=True)
                    if example_i < 5:
                        print("answerable_question:", answerable_question)
                        print("answerable_response:", answerable_response)
                else:
                    answerable_response = 'yes'

                if 'yes' in answerable_response.lower():
                    if not additional_text_input:
                        question = video_prefix + original_question
                    else:
                        question = video_prefix + " " + additional_text_input + "\n" + original_question
                    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                                num_patches_list=num_patches_list, history=None, return_history=True)
                    if example_i < 5:
                        print("question:", question)
                        print("response:", response)
                else:
                    response = None
                res = {
                    'video': example['video'], 'question_id': example['question_id'], 'video_span': [start_sec, end_sec],
                    'answerable': answerable_response, 'model_response': response, 'question': original_question}
                f_out.write(json.dumps(res) + '\n')
                f_out.flush()
    f_out.close()