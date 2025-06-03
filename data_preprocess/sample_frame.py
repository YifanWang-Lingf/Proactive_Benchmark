import argparse, os, json
from multiprocessing import Pool


def func(command):
    video_fname, frame_folder, fps = command
    if os.path.exists(frame_folder):
        print(f'frame {frame_folder} already exists, skip this')
        return
    os.makedirs(frame_folder, exist_ok=True)
    max_side = 512
    scale_filter = (
        f"fps={fps},"
        f"scale='if(gt(max(iw,ih),{max_side}),"
        f"if(gte(iw,ih),{max_side},-2),iw)':"
        f"'if(gt(max(iw,ih),{max_side}),"
        f"if(gte(ih,iw),{max_side},-2),ih)'"
    )

    os.system(f'ffmpeg -v error -i {video_fname} -vf "{scale_filter}" {frame_folder}/%04d.jpg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_video_folder', type=str)
    parser.add_argument('--dest_frame_folder', type=str)
    parser.add_argument('--anno_fname', type=str)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=1000000000)
    parser.add_argument('--fps', type=float, default=1)
    parser.add_argument('--num_workers', type=int, default=5)
    args = parser.parse_args()
    print(args)

    data_list = json.load(open(args.anno_fname, 'r'))[args.start_idx:args.end_idx]
    videos = list(set([e['video'] for e in data_list]))
    # debug: check new videos
    videos = [e for e in videos if e not in os.listdir('/share3/wangyq/resources/proactive_benchmark_frames/tvqa-1fps')]
    print(f'videos: {len(videos)}')

    os.makedirs(args.dest_frame_folder, exist_ok=True)
    commands = [(
        os.path.join(args.src_video_folder, v),
        os.path.join(args.dest_frame_folder, v), 
        args.fps) for v in videos
    ]

    with Pool(args.num_workers) as p:
        p.map(func, commands)

