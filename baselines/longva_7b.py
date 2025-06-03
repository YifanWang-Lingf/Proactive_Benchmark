import torch
from longva.model.builder import load_pretrained_model  # :contentReference[oaicite:1]{index=1}
from longva.mm_utils import tokenizer_image_token, process_images     # :contentReference[oaicite:2]{index=2}
from longva.constants import IMAGE_TOKEN_INDEX                  # :contentReference[oaicite:3]{index=3}

from utils.video_utils import load_video
from PIL import Image
def load_model(llm_pretrained, attn_implementation):
    """
    - llm_pretrained: HuggingFace 仓库名 "lmms-lab/LongVA-7B"（当前shell中使用的） 或 "lmms-lab/LongVA-7B-DPO"
    - attn_implementation: 依旧传进来但对 LongVA-7B 无效（LongVA 不接受 flash-attn 参数）
    
    返回 dict，包含：
      - model:       LongVA-7B 的模型实例（已 .eval().cuda()）
      - tokenizer:   LongVA-7B 对应的 tokenizer（用于把 prompt → token_ids、query 模板）
      - image_processor: LongVA-7B 的图像／视频预处理器（用于把 PIL 图像列表 → tensor）
      - generation_config: dict，后面调用 model.generate 时的参数
    """
    # 1. load_pretrained_model 会返回：tokenizer, model, image_processor, (context_len)
    #    第三个参数 "llava_qwen" 表示“对话模式”——LongVA 会自动在 prompt 里插入 ChatML 标签
    tokenizer, model, image_processor, _ = load_pretrained_model(
        llm_pretrained,   # e.g. "lmms-lab/LongVA-7B"
        None,             # load_pretrained_model 本身不需要 attn_implementation
        "llava_qwen",
        device_map="auto",          # 自动拆分权重到 GPU/CPU
        torch_dtype=torch.bfloat16, # 半精度
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # 2. 把模型设为 eval 模式
    model.eval()

    # 3. LongVA 示例里通常直接在 generate 时传 “do_sample, temperature” 等，
    #    这里固定 max_new_tokens 256, do_sample=True
    generation_config = {
        "do_sample": True,
        "max_new_tokens": 256
    }

    return {
        "model": model,
        "tokenizer": tokenizer,
        "processor": image_processor,
        "generation_config": generation_config
    }



def inference(
    baseline_model,
    video_data,
    text_data,
    need_judge_answerable=True,
    debug_print=False
):
    """

    参数：
      - baseline_model: load_model 返回的 dict，包含 {"model","tokenizer","processor","generation_config"}
      - video_data: {
            "video_file": <str 视频路径>,
            "start_sec":   <int chunk 开始秒>,
            "end_sec":     <int chunk 结束秒>,
            "frame_resolution": <int>（边长，通常 384、320 等）,
            "frame_fps": <int>（抽帧帧率，通常 1、2、4）
        }
      - text_data: {
            "original_question": <str 原始问题>,
            "additional_text_input": <str 额外文字，可为空>,
            "previous_turns_output": <str 历史对话，仅 openai_api 时有意义>
        }
      - need_judge_answerable: bool，是否先做“可答性判定”
      - debug_print: bool，是否把 prompt/原始输出/后处理结果都打印出来

    返回：
      {
        "answerable": <str，“Yes”/“No”/或原始 decoded_yesno>,
        "response":   <str 最终回答（原始 raw_text） 或 None>
      }
    """
    device = "cuda"
    model = baseline_model["model"]
    tokenizer = baseline_model["tokenizer"]
    processor = baseline_model["processor"]
    generation_config = baseline_model["generation_config"]

    # -------------------------------------------------
    # 1. 调用 load_video，从 video_data["video_file"] 抽帧
    #    得到 List[np.ndarray]，每个元素形状为 (3, H, W)
    # -------------------------------------------------
    pixel_list = load_video(
        video_data["video_file"],
        start_sec=video_data["start_sec"],
        end_sec=video_data["end_sec"],
        output_resolution=video_data["frame_resolution"],
        output_fps=video_data["frame_fps"]
    )
    pil_frames = []
    if len(pixel_list) > 0:
        for frame_np in pixel_list:
            # 把 (3, H, W) 转为 (H, W, 3)，再转成 PIL.Image
            pil_frames.append(
                Image.fromarray(frame_np.transpose(1, 2, 0))
            )
    # 如果没截到任何帧，pil_frames 为空列表
    
    # --------- 显存优化 1 释放掉原始的视频序列 ---------
    del pixel_list
    # -----------------------------------------------
    
    original_question = text_data["original_question"]
    additional_text_input = text_data.get("additional_text_input", "").strip()
    previous_turns_output = text_data.get("previous_turns_output", "").strip()

    # -------------------------------------------------
    # 2. “可答性判定”阶段：生成 Prompt，然后喂给 LongVA-7B 做 yes/no 判断
    # -------------------------------------------------
    if need_judge_answerable:
        # 2.1 构造用户 prompt
        if not additional_text_input:
            prompt_yesno = (
                "Is the following question answerable by the video? "
                "Only reply with Yes or No.\n" + original_question
            )
        else:
            prompt_yesno = (
                additional_text_input + "\n"
                + "Is the following question answerable by the video and subtitles? "
                "Only reply with Yes or No.\n"
                + original_question
            )

        # 2.2 拼装成 ChatML 风格文本：含 system + user + <video> token + question + <assistant> token
        #     这里之所以要手动插入 system，是因为 LongVA-7B 需要“你是有帮助的助手”这类左侧提示，
        #     否则模型有时会先输出其它标签再跟答案，干扰判断。
        chatml_yesno = (
            "<|im_start|>system\n"
            "You are a helpful assistant. Reply with Yes or No only.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "<video>\n"
            f"{prompt_yesno}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # 2.3 把以上 ChatML 文本和所有帧一起编码成模型输入
        #     a) 先用 tokenizer_image_token 把 ChatML 文本 token 化（包含 <image> / <video> 等特殊符号）
        input_ids_yesno = tokenizer_image_token(
            chatml_yesno,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to(device)  # 形状 [1, seq_len]

        #     b) 再用 process_images 把 PIL 列表转成 tensor。process_images 会根据 model.config 把帧编码成视觉嵌入。
        #        这里直接把整个 pil_frames 列表当作“video frames”传入
        images_tensor_yesno = (
            process_images(pil_frames, processor, model.config)
            .to(device, dtype=torch.float16)
        )
        #  这一行做了：
        #    - process_images(...) 返回 [batch, 3, H, W] 或者 [num_frames, 3, H, W] 之类
        #    - reshape 以适配 model.generate 中的 images 参数

        # 2.4 调用 model.generate 生成“Yes/No”判断
        with torch.inference_mode():
            generated_yesno = model.generate(
                input_ids_yesno,
                images=images_tensor_yesno,
                image_sizes=[frame.size for frame in pil_frames],  
                # image_sizes 需要一个 list of (W, H)，表示每一帧的原始宽高
                modalities=["image"],  
                # LongVA-7B 官方示例里 image_tasks 用 modalities=["image"]
                max_new_tokens=generation_config["max_new_tokens"],
                do_sample=generation_config["do_sample"]
            )
        # 2.5 decode 成文本
        raw_yesno = tokenizer.batch_decode(
            generated_yesno, skip_special_tokens=True
        )[0]
        if debug_print:
            print("----- 可答性 原始 模型输出（raw_yesno） -----")
            print(raw_yesno)

        # 2.6 根据 raw_yesno 里是否包含 “yes”/“no” 来判断
        lower_yesno = raw_yesno.lower()
        if "yes" in lower_yesno:
            answerable_response = "Yes"
        elif "no" in lower_yesno:
            answerable_response = "No"
        else:
            # 如果 raw_yesno 里既不包含 yes 也不包含 no，就当做模型输出的原始回答
            answerable_response = raw_yesno.strip()

        if debug_print:
            print(">> 处理后 answerable_response =", answerable_response)
        
        
        # --------- 显存优化 2 释放掉第一轮可回答性推理的中间数据张量   ---------
        del input_ids_yesno, generated_yesno, raw_yesno
        del images_tensor_yesno
        torch.cuda.empty_cache()
        # --------------------------------------------------------------------
        
        
    else:
        answerable_response = "Yes"

    # -------------------------------------------------
    # 3. 正式回答阶段：只有当 answerable_response 含 “yes” 时才做
    # -------------------------------------------------
    if "yes" in answerable_response.lower():
        # 3.1 构造正式回答 prompt：如果有 previous_turns_output，把它拼第一行；否则只看 original_question + additional_text_input
        if not previous_turns_output:
            if not additional_text_input:
                prompt_final = original_question
            else:
                prompt_final = additional_text_input + "\n" + original_question
        else:
            if not additional_text_input:
                prompt_final = previous_turns_output + "\n" + original_question
            else:
                prompt_final = (
                    previous_turns_output
                    + "\n"
                    + additional_text_input
                    + "\n"
                    + original_question
                )

        # 3.2 ChatML 格式：system + user + <video>… + question + <assistant>
        chatml_final = (
            "<|im_start|>system\n"
            "You are a helpful assistant. Provide a concise answer based on the video.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "<video>\n"
            f"{prompt_final}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # 3.3 tokenize 文本 + process_images(帧) → 得到 input_ids_final, images_tensor_final
        input_ids_final = tokenizer_image_token(
            chatml_final,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to(device)

        images_tensor_final = (
            process_images(pil_frames, processor, model.config)
            .to(device, dtype=torch.float16)
        )

        # 3.4 调用 model.generate 生成最终回答
        with torch.inference_mode():
            generated_final = model.generate(
                input_ids_final,
                images=images_tensor_final,
                image_sizes=[frame.size for frame in pil_frames],
                modalities=["image"],
                max_new_tokens=generation_config["max_new_tokens"],
                do_sample=generation_config["do_sample"]
            )
        raw_final = tokenizer.batch_decode(
            generated_final, skip_special_tokens=True
        )[0]
        if debug_print:
            print("----- 正式问答 原始 模型输出（raw_final） -----")
            print(raw_final)

        # 3.5 不做任何剥离（Longva模型不会像Qwen系列回答会说一堆无关的，只会说回答），直接把 raw_final 作为 model_response 返回
        response = raw_final.strip()
        
        # --------- 显存优化 3 释放掉正式回答推理的中间数据张量   ---------
        del input_ids_final, generated_final, raw_final
        del images_tensor_final
        torch.cuda.empty_cache()
        # --------------------------------------------------------------
        
    else:
        response = None

    return {
        "answerable": answerable_response,
        "response": response
    }