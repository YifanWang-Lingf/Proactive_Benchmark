import os, json
from tqdm import tqdm
from transformers import HfArgumentParser

from models import parse_args
from models.arguments_live import LiveTestArguments
from baselines import load_model, inference
from dataclasses import dataclass
from utils.video_utils import get_video_length_sec


@dataclass
class ClipTestArguments(LiveTestArguments):
    video_chunk_sec: int = 3
    need_judge_answerable: int = 1


if __name__ == '__main__':
    args, = HfArgumentParser(ClipTestArguments).parse_args_into_dataclasses()
    import debugpy
    
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
    # 加载模型，调用文件顺序为 本文件 <- baselines/__init__.py <- baselines/{llava_ov.py}
    baseline_model = load_model(model_type, args.llm_pretrained, args.attn_implementation)
    
    # args.output_fname = outputs/internvl2_5-8b/magqa/2sec.jsonl.0 是结果写入文件
    # 如果结果文件已经存在，则读取已测试的样本数据，避免重复测试，相当于是一个检查点操作
    if os.path.exists(args.output_fname):
        tested_examples = [json.loads(line) for line in open(args.output_fname)]
        tested_questions = set([(e['question_id'], e['video_span'][0], e['video_span'][1]) for e in tested_examples])
    else:
        tested_questions = set()
        
    # f_out 准备向结果文件中写入数据
    f_out = open(args.output_fname, 'a')
    
    # test_data 是json文件，只包含单张GPU需要处理的样本数据，是原始标注文件
    test_data = json.load(open(args.test_fname))[args.start_idx: args.end_idx]

    # 一次处理一条样本
    for example_i, example in tqdm(enumerate(test_data), total=len(test_data)):
        # 首先获取此样本中的视频的长度
        try:
            if 'duration' in example:
                video_length_sec = example['duration']
            else:
                video_length_sec = get_video_length_sec(os.path.join(args.input_dir, example['video']))
        except:
            print(f"get video length sec error for {example['question_id']}, skipping")
            continue

        last_reply_end_time = example['answer'][-1]['reply_timespan'][1]
        previous_turns_output = []
        
        # 对每一个video chunk，执行一次推理, for循环给出每个video chunk的开始时间 start_sec
        for start_sec in range(0, int(video_length_sec), args.video_chunk_sec):
            if start_sec > last_reply_end_time: break       # as replies after the last reply span will not count into the metrics
            
            # 然后计算出此video chunk的结束时间 end_sec
            end_sec = min(video_length_sec, start_sec + args.video_chunk_sec)
            
            # 跳过已经测试过的时间段
            if (example['question_id'], start_sec, end_sec) in tested_questions:
                print(f"question {example['question_id']} {start_sec} {end_sec} has been tested, skipping")
                continue
            
            # 获取原始问题和额外的文本输入的文本内容 <-- 单个字符串
            # original_question：example['conversation']<-字典 [0]<-列表 ['content']<-字典 = "What are people doing in the # office?"
            # additional_text_input：example['conversation']<-字典 [1]<-列表 ['content']<-字典 = "The office is busy with # people working."，多个额外输入文本会被拼接起来成为一个大的字符串
            
            original_question = example['conversation'][0]['content']
            additional_text_input = ""
            for input_turns in example['conversation']:
                if start_sec < input_turns['time'] <= end_sec:
                    additional_text_input += f"{input_turns['content']}\n"
                    
            # 获取视频数据的文件位置，此video chunk的开始和结束时间，以及帧分辨率和帧率，不同模型使用的视频数据接口应该是一样的
            video_data = {
                'video_file': os.path.join(args.input_dir, example['video']),
                'start_sec': start_sec, 'end_sec': end_sec,
                'frame_resolution': args.frame_resolution, 'frame_fps': args.frame_fps
            }
            
            # 获取文本数据，包括原始问题、额外的文本输入，不同模型使用的文本数据接口应该是一样的
            text_data = {
                'original_question': original_question,
                'additional_text_input': additional_text_input
            }

            # 如果是 OpenAI API 模型，还需要传入之前的输出
            if model_type in ['openai_api']:
                text_data['previous_turns_output'] = ' '.join(previous_turns_output)

            # 执行推理，传入MLLM模型，模型类别标号。视频数据，文本数据；返回的 response_dict 包含了模型的回答和是否可答的判断
            response_dict = inference(
                baseline_model=baseline_model,
                model_type=model_type, video_data=video_data, text_data=text_data,
                need_judge_answerable=args.need_judge_answerable, debug_print=example_i < 5)

            # 如果是 OpenAI API 模型，记录之前的输出            
            if model_type in ['openai_api']:
                previous_turns_output.append(response_dict['response'])

            # 打包归纳此video chunk的推理结果为字典，将其写入结果文件中(outputs/internvl2_5-8b/magqa/2sec.jsonl.0)
            res = {
                'video': example['video'], 'question_id': example['question_id'], 'video_span': [start_sec, end_sec],
                'answerable': response_dict['answerable'], 'model_response': response_dict['response'],
                'question': original_question
            }

            f_out.write(json.dumps(res) + '\n')
            f_out.flush()
    f_out.close()

# 没问题，测试通过了！