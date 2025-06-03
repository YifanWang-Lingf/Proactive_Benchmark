from google import genai
from google.genai import types
import os
import numpy as np


def load_model(llm_pretrained, *args, **kwargs):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return {'client': client, 'model': llm_pretrained}


def inference(baseline_model, video_data, text_data,
              need_judge_answerable=True, debug_print=False):
    client, model = baseline_model['client'], baseline_model['model']

    start_sec, end_sec, video_fps = video_data['start_sec'], video_data['end_sec'], video_data['frame_fps']
    second_per_frame = 1 / video_fps

    start_sec = int(start_sec / second_per_frame) * second_per_frame
    end_sec = int(end_sec / second_per_frame) * second_per_frame

    # load the frames
    image_bytes_list = list()
    for frame_i in np.arange(start_sec, end_sec, second_per_frame):
        if 'magqa-2fps' in video_data['video_file']:
            frame_i = int(frame_i * 2)
        image_fname = video_data['video_file'] + '/%04d.jpg' % (frame_i + 1)
        if debug_print:
            print('loading image:', image_fname)
        with open(image_fname, 'rb') as f:
            image_bytes_list.append(f.read())

    original_question = text_data['original_question']
    additional_text_input = text_data.get('additional_text_input', '')
    previous_turns_output = text_data.get('previous_turns_output', '')

    if not previous_turns_output:
        user_input = f"Question:\n{original_question}"
        if additional_text_input:
            instruction = (
                f"You are provided with some frames sampled from a video between {start_sec} seconds and {end_sec} seconds at a frame rate of {video_fps} frames per second. You are also given a question, and subtitles of this video segment as additional information.\n"
                "Based on the video content, you need to output \"I have a new answer.\" or \"I have no new answer.\" in the first line of your response, indicating whether the question is answerable by the video. If you answer \"I have a new answer.\", you should output your answer in the second line.\n"
                "Your answers must be based solely on the video content and subtitles in additional information. Do not add your own speculation or judgement."
            )
            user_input += f"\nAdditional Information:\n{additional_text_input}"
        else:
            instruction = (
                f"You are provided with some frames sampled from a video between {start_sec} seconds and {end_sec} seconds at a frame rate of {video_fps} frames per second. You are also given a question.\n"
                "Based on the video content, you need to output \"I have a new answer.\" or \"I have no new answer.\" in the first line of your response, indicating whether the question is answerable by the video. If you answer \"I have a new answer.\", you should output your answer in the second line.\n"
                "Your answers must be based solely on the video content. Do not add your own speculation or judgement."
            )

    else:
        assert start_sec != 0
        # instruction = (
        #     f"You are provided with some frames sampled from a video between {start_sec} seconds and {end_sec} seconds at a frame rate of {video_fps} frames per second. You are also given a question, and the previous answers that you have already provided based on the video before {start_sec} seconds.\n"
        #     "Based on the video content, you need to output \"I have a new answer.\" or \"I have no new answer.\" in the first line of your response, indicating whether the question is answerable by the video. If you answer \"I have a new answer.\", you should output your answer in the second line.\n"
        #     "Your answers must be based solely on the video content. Do not add your own speculation or judgement."
        #     # "Based on the video content, you need to output \"I have a new answer.\" or \"I have no new answer.\" in the first line of your response, indicating whether there is a new answer in the video for the question that is different from all previous answers. If you answer \"I have a new answer.\", you should output your answer in the second line.\n"
        # )

        user_input = f"Question:\n{original_question}"
        if additional_text_input:
            instruction = (
                f"You are provided with some frames sampled from a video between {start_sec} seconds and {end_sec} seconds at a frame rate of {video_fps} frames per second. You are also given a question, the previous answer that you have already provided based on the video before {start_sec} seconds, and subtitles of this video segment as additional information.\n"
                "In the first line of your reply, you need to answer whether the question is answerable by the video. If the question is not answerable, output \"I have no answer.\" If the question is answerable and the answer is exactly the same as the previous answer, output \"I have the same answer.\" If there is a new answer in the video that is different from the previous answer, output \"I have a new answer.\", and output your answer in the second line.\n"
                "Your answers must be based solely on the video content and subtitles in additional information. Do not add your own speculation or judgement."
            )
            user_input += f"\nAdditional Information:\n{additional_text_input}"
        else:
            instruction = (
                f"You are provided with some frames sampled from a video between {start_sec} seconds and {end_sec} seconds at a frame rate of {video_fps} frames per second. You are also given a question, and the previous answer that you have already provided based on the video before {start_sec} seconds.\n"
                "In the first line of your reply, you need to answer whether the question is answerable by the video. If the question is not answerable, output \"I have no answer.\" If the question is answerable and the answer is exactly the same as the previous answer, output \"I have the same answer.\" If there is a new answer in the video that is different from the previous answer, output \"I have a new answer.\", and output your answer in the second line.\n"
                "Your answers must be based solely on the video content. Do not add your own speculation or judgement."
            )
        # user_input += f"\nPrevious Answers:\n{' '.join(previous_turns_output)}"
        user_input += f"\nPrevious Answer:\n{previous_turns_output[-1]}"

    conversation = [types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg') for image_bytes in image_bytes_list]
    conversation.append(instruction + '\n' + user_input)

    if debug_print:
        print("model input:", [i if type(i) == str else type(i) for i in conversation])

    response = client.models.generate_content(model=model, contents=conversation)

    if debug_print:
        print("model response:", response)

    splits = response.text.split('\n')
    splits = [i.strip() for i in splits if i.strip()]
    if len(splits) == 1:
        if splits[0].strip() in ['I have no answer.', 'I have no new answer.']:
            answerable = False
            response = ''
        elif splits[0].strip() == 'I have the same answer.':
            answerable = True
            response = previous_turns_output[-1]
    else:
        answerable, response = True, '\n'.join(splits[1:])

    return {'answerable': answerable, 'response': response}