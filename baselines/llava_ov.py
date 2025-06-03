import sys, copy, os
sys.path.append("..")

import numpy as np
import torch

from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from utils.video_utils import load_video, get_video_length_sec

from transformers import AutoConfig, AutoTokenizer
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

def load_model(llm_pretrained, attn_implementation):
    
    
    # llava onevision baseline
    # 这里做出修改以解决报错，将：llava_qwen 改为 llava_onevision
    tokenizer, model, image_processor, max_length = load_pretrained_model(llm_pretrained, None, "llava_qwen", device_map="auto", attn_implementation=attn_implementation)  # Add any other thing you want to pass in llava_model_args
    baseline_model = {'model': model, 'tokenizer': tokenizer, 'image_processor': image_processor}
    return baseline_model


def inference(baseline_model, video_data, text_data,
              need_judge_answerable=True, debug_print=False):

    model, tokenizer, image_processor = baseline_model['model'], baseline_model['tokenizer'], baseline_model['image_processor']
    
    video_frames = load_video(
        video_file=video_data['video_file'],
        start_sec=video_data['start_sec'],
        end_sec=video_data['end_sec'],
        output_resolution=video_data['frame_resolution'],
        output_fps=video_data['frame_fps']
    )
    video_frames = torch.tensor(np.stack(video_frames))
    image_sizes = [frame.size() for frame in video_frames]
    image_tensor = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().to(model.device)
    modalities = ["video"] * len(video_frames)
    conv_template = 'qwen_1_5'

    original_question = text_data['original_question']
    additional_text_input = text_data.get('additional_text_input', '')
    previous_turns_output = text_data.get('previous_turns_output', '')

    if need_judge_answerable:
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
        if debug_print:
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
        if not previous_turns_output:
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
        else:
            conv.append_message(conv.roles[1], previous_turns_output)
            prompt_question = conv.get_prompt()
            sep_token = conv.sep
            last_sep_index = prompt_question.rfind(sep_token)
            prompt_question = prompt_question[:last_sep_index] + ' '
            # we do not need to add anything right after the last period mark '.'

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
        cont = model.generate(
            input_ids, images=[image_tensor], image_sizes=image_sizes,
            do_sample=False, temperature=0, max_new_tokens=256, modalities=modalities,
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        if debug_print:
            print("question:", question)
            print("model_input:", prompt_question)
            print("text_outputs:", text_outputs)
    else:
        text_outputs = None

    return {'answerable': answerable_response, 'response': text_outputs}
