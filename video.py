import os
import ffmpeg

# raw_video_path = './videos/results_00.mp4'

def split_video(filename, destination, split_frames, newfile='newfile', subdir=False):
    new_files = ['control', 'stim', 'post']

    if subdir != False:
        if type(destination) != list:
            print('[CREATING SUBDIR]: ' + subdir)
            subdir = destination + '/' + subdir + '/'
            if not os.path.exists(subdir):
                os.makedirs(subdir)
        else:
            print('[ERROR]: destination is of type <list>')
    else:
        subdir = destination
    
    probe_result = ffmpeg.probe(filename)
    duration = probe_result.get("format", {}).get("duration", None)
    print('[ORIGINAL DURATION]:', duration)

    split_list = [0]
    for frame in split_frames:
        frame_to_sec = frame / 30
        split_list.append(frame_to_sec)
    split_list.append(duration)

    input_stream = ffmpeg.input(filename)

    pts = "PTS-STARTPTS"

    d_count = 0
    for condition in new_files:
        print('[CREATING VIDEO]:', condition.upper())
        print('[SPLITTING]: \n -START FRAME:', split_list[0], ' \n -END FRAME:', split_list[1])
        video = input_stream.trim(start=split_list[0], end=split_list[1]).setpts(pts)
        if type(destination) != list:
            output = ffmpeg.output(video, subdir + '/' + condition + newfile + '.mp4', format="mp4")
        else:
            output = ffmpeg.output(video, subdir[d_count] + '/' + newfile + '.mp4', format="mp4")
            d_count += 1
        output.run()
        split_list = split_list[1:]
        
        
# split_video(raw_video_path, "../finals", [18000, 36000])