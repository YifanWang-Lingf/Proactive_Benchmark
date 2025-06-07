import os, sys, copy
sys.path.append('..')
import torch
from transformers import HfArgumentParser, AutoModel, AutoTokenizer


from utils import internvl_utils


def load_model(llm_pretrained, attn_implementation):        # internvideo baseline
    model = AutoModel.from_pretrained(
        llm_pretrained, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, use_flash_attn=attn_implementation.startswith('flash'),
        trust_remote_code=True).eval().cuda()
    generation_config = dict(max_new_tokens=256, do_sample=True)
    tokenizer = AutoTokenizer.from_pretrained(llm_pretrained, trust_remote_code=True, use_fast=False)
    baseline_model = {'model': model, 'tokenizer': tokenizer, 'generation_config': generation_config}
    return baseline_model


def complete_truncated_assistant_turns(self, tokenizer, pixel_values, question, previous_turns_output, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

    if history is None and pixel_values is not None and '<image>' not in question:
        question = '<image>\n' + question

    if num_patches_list is None:
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    self.img_context_token_id = img_context_token_id

    template = copy.deepcopy(self.conv_template)
    template.system_message = self.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], previous_turns_output)
    query = template.get_prompt()
    sep_token = template.sep
    last_sep_index = query.rfind(sep_token)
    query = query[:last_sep_index]

    if verbose and pixel_values is not None:
        image_bs = pixel_values.shape[0]
        print(f'dynamic ViT batch size: {image_bs}')
        print('text input:', query)

    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response


def inference(baseline_model, video_data, text_data,
              need_judge_answerable=True, debug_print=False):
    model, tokenizer, generation_config = baseline_model['model'], baseline_model['tokenizer'], baseline_model['generation_config']
    pixel_values, num_patches_list = internvl_utils.load_video(
        video_data['video_file'],
        bound=(video_data['start_sec'], video_data['end_sec']),
        num_segments=int((video_data['end_sec'] - video_data['start_sec']) * video_data['frame_fps'])
    )

    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])

    original_question = text_data['original_question']
    additional_text_input = text_data.get('additional_text_input', '')
    previous_turns_output = text_data.get('previous_turns_output', '')

    if need_judge_answerable:
        if not additional_text_input:
            answerable_question = video_prefix + "Is the following question answerable by the video? Only reply with Yes or No.\n" + original_question
        else:
            answerable_question = video_prefix + " " + additional_text_input + "\nIs the following question answerable by the video and subtitles? Only reply with Yes or No.\n" + original_question
        answerable_response, history = model.chat(tokenizer, pixel_values, answerable_question, generation_config,
                                    num_patches_list=num_patches_list, history=None, return_history=True)
        if debug_print:
            print("answerable_question:", answerable_question)
            print("answerable_response:", answerable_response)
    else:
        answerable_response = 'yes'

    if 'yes' in answerable_response.lower():
        if not previous_turns_output:
            if not additional_text_input:
                question = video_prefix + original_question
            else:
                question = video_prefix + " " + additional_text_input + "\n" + original_question
            question += '\nYour answers can only contain video content. Do not add your own speculation or judgement. Do not add timestamps or frame numbers in your answer.'

            response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                        num_patches_list=num_patches_list, history=None, return_history=True)
        else:
            # model.chat function does not support complete truncated assistant turns, so we need to re-implement this
            response, history = complete_truncated_assistant_turns(
                model, tokenizer, pixel_values, question, previous_turns_output, generation_config,
                num_patches_list=num_patches_list, history=None, return_history=True, verbose=debug_print
            )

        if debug_print:
            print("question:", question)
            print("response:", response)
    else:
        response = None

    return {'answerable': answerable_response, 'response': response}
