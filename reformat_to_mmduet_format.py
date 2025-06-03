import argparse, json, os
import string


def have_letters(text):
    for letter in text:
        if letter.isalpha(): return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fname', type=str)
    parser.add_argument('--output_fname', type=str)
    args = parser.parse_args()
    print(args)

    data_list = open(args.input_fname).readlines()
    data_list = [json.loads(line.strip()) for line in data_list]

    f_out = open(args.output_fname, 'w')
    prev_question_id, prev_video = data_list[0]['question_id'], data_list[0]['video']
    model_response_list = list()
    for data in data_list:
        if data['question_id'] != prev_question_id:
            res = {
                'question_id': prev_question_id, 'video': prev_video, 'model_response_list': model_response_list
            }
            f_out.write(json.dumps(res) + '\n')
            model_response_list = list()
        prev_question_id, prev_video = data['question_id'], data['video']
        if 'model_response' not in data: print(data)        # DEBUG
        if data['model_response'] is not None and have_letters(data['model_response']):
            model_response_list.append({'time': data['video_span'][1], 'content': data['model_response']})
    res = {
        'question_id': prev_question_id, 'video': prev_video, 'model_response_list': model_response_list
    }
    f_out.write(json.dumps(res) + '\n')
    f_out.close()
