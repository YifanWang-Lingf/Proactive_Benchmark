from openai import OpenAI
import numpy as np

NO_PREV_TURN_ADDITIONAL_TEXT_PROMPT = (
    "You are provided with some frames sampled from a video between {start_sec} seconds and {end_sec} seconds at a frame rate of {video_fps} frames per second. You are also given a question, and subtitles of this video segment as additional information.\n"
    "Based on the video content, you need to output \"I have a new answer.\" or \"I have no new answer.\" in the first line of your response, indicating whether the question is answerable by the video. If you answer \"I have a new answer.\", you should output your answer in the second line.\n"
    # "Your answers must be based solely on the video content and subtitles in additional information. Do not add your own speculation or judgement."
    "Your answers can only contain video content or subtitles in additional information. Do not add your own speculation or judgement. Do not add timestamps or frame numbers in your answer."
)

NO_PREV_TURN_PROMPT = (
    "You are provided with some frames sampled from a video between {start_sec} seconds and {end_sec} seconds at a frame rate of {video_fps} frames per second. You are also given a question.\n"
    "Based on the video content, you need to output \"I have a new answer.\" or \"I have no new answer.\" in the first line of your response, indicating whether the question is answerable by the video. If you answer \"I have a new answer.\", you should output your answer in the second line.\n"
    # "Your answers must be based solely on the video content. Do not add your own speculation or judgement."
    "Your answers can only contain video content. Do not add your own speculation or judgement. Do not add timestamps or frame numbers in your answer."
)

PREV_TURN_ADDITIONAL_TEXT_PROMPT = (
    "You are provided with some frames sampled from a video between {start_sec} seconds and {end_sec} seconds at a frame rate of {video_fps} frames per second. You are also given a question, the previous answer that you have already provided based on the video before {start_sec} seconds, and subtitles of this video segment as additional information.\n"
    "In the first line of your reply, you need to answer whether the question is answerable by the video. If the question is not answerable, output \"I have no answer.\" If the question is answerable and the answer is exactly the same as the previous answer, output \"I have the same answer.\" If there is a new answer in the video that is different from the previous answer, output \"I have a new answer.\", and output your answer in the second line.\n"
    # "Your answers must be based solely on the video content and subtitles in additional information. Do not add your own speculation or judgement."
    "Your answers can only contain video content or subtitles in additional information. Do not add your own speculation or judgement. Do not add timestamps or frame numbers in your answer."
)

PREV_TURN_PROMPT = (
    "You are provided with some frames sampled from a video between {start_sec} seconds and {end_sec} seconds at a frame rate of {video_fps} frames per second. You are also given a question, and the previous answer that you have already provided based on the video before {start_sec} seconds.\n"
    "In the first line of your reply, you need to answer whether the question is answerable by the video. If the question is not answerable, output \"I have no answer.\" If the question is answerable and the answer is exactly the same as the previous answer, output \"I have the same answer.\" If there is a new answer in the video that is different from the previous answer, output \"I have a new answer.\", and output your answer in the second line.\n"
    # "Your answers must be based solely on the video content. Do not add your own speculation or judgement."
    "Your answers can only contain video content. Do not add your own speculation or judgement. Do not add timestamps or frame numbers in your answer."
)


NO_PREV_TURN_PROMPT = (
    "You are provided with some frames sampled from a video between {start_sec} seconds and {end_sec} seconds at a frame rate of {video_fps} frames per second. You are also given a question.\n"
    "In the first line of your reply, you need to answer whether the question is answerable by the video. If the question is not answerable by the video, or the video is unrelated to the question, output \"I have no answer.\" If the question is answerable and the answer is exactly the same as the previous answer, output \"I have the same answer.\" If there is a new answer in the video that is different from the previous answer, output \"I have a new answer.\", and output your answer in the second line.\n"
    "Your answers can only contain video content. Do not add your own speculation or judgement. Do not add any timestamp or frame number in your answer."
)

PREV_TURN_PROMPT = (
    "You are provided with some frames sampled from a video between {start_sec} seconds and {end_sec} seconds at a frame rate of {video_fps} frames per second. You are also given a question, and the previous answer that you have already provided based on the video before {start_sec} seconds.\n"
    "In the first line of your reply, you need to answer whether the question is answerable by the video. If the question is not answerable by the video, or the video is unrelated to the question, output \"I have no answer.\" If the question is answerable and the answer is exactly the same as the previous answer, output \"I have the same answer.\" If there is a new answer in the video that is different from the previous answer, output \"I have a new answer.\", and output your answer in the second line.\n"
    "Your answers can only contain video content. Do not add your own speculation or judgement. Do not add any timestamp or frame number in your answer."
)




def load_model(llm_pretrained, *args, **kwargs):
    client = OpenAI()
    return {'client': client, 'model': llm_pretrained}


def inference(baseline_model, video_data, text_data,
              need_judge_answerable=True, debug_print=False):
    client, model = baseline_model['client'], baseline_model['model']

    start_sec, end_sec, video_fps = video_data['start_sec'], video_data['end_sec'], video_data['frame_fps']
    second_per_frame = 1 / video_fps

    start_sec = int(start_sec / second_per_frame) * second_per_frame
    end_sec = int(end_sec / second_per_frame) * second_per_frame
    
    # load the frame urls
    frame_urls = []
    for frame_i in np.arange(start_sec, end_sec, second_per_frame):
        if 'magqa-2fps' in video_data['video_file']:
            frame_i = int(frame_i * 2)
        frame_urls.append(video_data['video_file'] + '/%04d.jpg' % (frame_i + 1))

    original_question = text_data['original_question']
    additional_text_input = text_data.get('additional_text_input', '')
    previous_turns_output = text_data.get('previous_turns_output', '')

    if not previous_turns_output:
        user_input = f"Question:\n{original_question}"
        if additional_text_input:
            instruction = NO_PREV_TURN_ADDITIONAL_TEXT_PROMPT.format(start_sec=start_sec, end_sec=end_sec, video_fps=video_fps)
            user_input += f"\nAdditional Information:\n{additional_text_input}"
        else:
            instruction = NO_PREV_TURN_PROMPT.format(start_sec=start_sec, end_sec=end_sec, video_fps=video_fps)
    else:
        assert start_sec != 0

        user_input = f"Question:\n{original_question}"
        if additional_text_input:
            instruction = PREV_TURN_ADDITIONAL_TEXT_PROMPT.format(start_sec=start_sec, end_sec=end_sec, video_fps=video_fps)
            user_input += f"\nAdditional Information:\n{additional_text_input}"
        else:
            instruction = PREV_TURN_PROMPT.format(start_sec=start_sec, end_sec=end_sec, video_fps=video_fps)
        user_input += f"\nPrevious Answer:\n{previous_turns_output[-1]}"

    conversation = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": 
            [{"type": "input_image", "image_url": url} for url in frame_urls] + \
            [{"type": "input_text", "text": user_input}]
        }
    ]
    if debug_print:
        print("model input:", conversation)

    response = client.responses.create(model=model, input=conversation)
    if debug_print:
        print("model response:", response)

    splits = response.output_text.split('\n')
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