#!/usr/bin/env python

import cv2
import glob
import argparse
import os
import sys
from datetime import datetime
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, concatenate_videoclips

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--videofile", help="Video file to parse", metavar='file')
parser.add_argument("-f", "--frame-skip", type=int, help="Frames to skip", metavar='', default = 45)
parser.add_argument("-m", "--maxsec", type=int, help="Max seconds to look at", metavar='', default = 0)
parser.add_argument("-i", "--image", help="Template image(s)", nargs='*', metavar='image...', default=[])
#parser.add_argument("-c", "--method", default = 'cv2.TM_CCOEFF_NORMED')
parser.add_argument("-s", "--show-image", help="Preview successes or not (boolean)", action="store_true", default=False)
parser.add_argument("-d", "--dump", help="Dump all frames (boolean)", action="store_true", default=False)
parser.add_argument("-u", "--use-frames", help="Use existing frames (boolean)", action="store_true", default=False)
parser.add_argument("-j", "--just-aggregate", help="Skip everything, just aggregate videos", nargs="+", metavar='video...')
parser.add_argument("-t", "--threshold", type=float, help="Threshold match value", metavar='', default = 1.0)
parser.add_argument("-g", "--generate-videos", help="Generate individual video files (boolean)", action="store_true", default=False)
parser.add_argument("-a", "--aggregate-video", help="Generate one aggregated video (boolean)", action="store_true", default=False)
parser.add_argument("-e", "--pre-delay", type=int, help="Seconds before detection in generated videos", metavar='', default = 10)
parser.add_argument("-o", "--post-delay", type=int, help="Seconds after detection in generated videos", metavar='', default = 7)
args = parser.parse_args()

videofile = args.videofile
frameskip = args.frame_skip
maxsecs = args.maxsec
goodimgfiles = args.image
#method = cv2.TM_CCOEFF_NORMED
method = cv2.TM_SQDIFF_NORMED
show_image = args.show_image
dump = args.dump
use_frames = args.use_frames
just_aggregate = args.just_aggregate
thresh = args.threshold
generate_videos = args.generate_videos
aggregate_video = args.aggregate_video
predelay = args.pre_delay
postdelay = args.post_delay

if just_aggregate:
    print 'Simply aggregating, ignoring all else'
    agg_vids = [VideoFileClip(v) for v in just_aggregate]
    agg_clip = concatenate_videoclips(agg_vids)
    now = datetime.now().isoformat().split('.')[0].replace(':','-')
    agg_clip.write_videofile("./aggs/agg-%s.mp4" % now)
    sys.exit(0)

if not videofile:
    raise ValueError('Need a video file to parse!')

if (not dump) and (not goodimgfiles):
    raise ValueError('Not dumping or comparing, not sure what you want from me.')

out_dir = './output'
video_name = os.path.basename(videofile).replace(" ", "_")
directory = out_dir + "/matches-" + video_name
adirectory = out_dir + "/frames-" + video_name

if not os.path.exists(directory):
    os.makedirs(directory)

if (not os.path.exists(adirectory)):
    if use_frames:
        raise ValueError('Frame directory not found: %s' % adirectory)
    if dump:
        os.makedirs(adirectory)

if generate_videos or aggregate_video:
    movie = VideoFileClip(videofile)
    movielen = movie.duration
    vdirectory = out_dir + "/videos-" + video_name
    video_name_short = os.path.splitext(video_name)[0]
    if not os.path.exists(vdirectory):
        os.makedirs(vdirectory)
    if aggregate_video:
        video_list = []

# can take this out if you just want to read in the first file in adir
vidcap = cv2.VideoCapture(videofile)
fps = int(round(vidcap.get(5)))
maxframes = fps * maxsecs

vid_w = vidcap.get(3)
vid_h = vidcap.get(4)

good_dict = {}
for goodimgfile in goodimgfiles:
    print goodimgfile
    
    good_read = cv2.imread(goodimgfile, 0)
    good_short = os.path.splitext(os.path.basename(goodimgfile))[0]
    
    gh = float(good_read.shape[0])
    gw = float(good_read.shape[1])
    
    magic_factor_w = (gw / 1280)
    magic_factor_h = (gh / 720)
    res_w = magic_factor_w * vid_w
    res_h = magic_factor_h * vid_h
    resiw = int(res_w)
    resih = int(res_h)
    
    good_resize = cv2.resize(good_read, (resiw, resih))
    good_edge = cv2.Canny(good_resize, 50, 150)
    
    good_dict[good_short] = {'edge': good_edge, 'just_found': False}
    
    good_dict_list = good_dict.keys()
    good_dict_str = ','.join(good_dict_list)

def matchImg(image, frame):
    print 'Frame: %s' % frame
    
    already_shown = False
    
    frame_total_secs = frame*1.0 / fps
    frame_hour = int(frame_total_secs / (60*60))
    frame_min = int((frame_total_secs / 60) % 60)
    frame_sec = frame_total_secs % 60
    frame_time = '%02d:%02d:%04.1f' % (frame_hour, frame_min, frame_sec)
    
    if dump and not use_frames:
        cv2.imwrite("%s/frame_%s_%s.jpg" % (adirectory, frame, frame_time), image)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    large_edge = cv2.Canny(gray_image, 50, 200)
    
    good_matches = []
    for goodimg in good_dict_list:
        goodimgdata = good_dict.get(goodimg)
        small_edge = goodimgdata['edge']
        just_found = goodimgdata['just_found']
        
        result = cv2.matchTemplate(small_edge, large_edge, method)
        
        minres = cv2.minMaxLoc(result)
        mn,mx,mnLoc,mxLoc = minres
        #print mn,mx
        
        if method == cv2.TM_SQDIFF_NORMED:
            thresh_pass = mn < thresh
            MPx,MPy = mnLoc
        else:
            thresh_pass = mx >= thresh
            MPx,MPy = mxLoc
        
        if thresh_pass:
            print '============== MATCH FOUND (%s) ===============' % goodimg
            
            if not just_found:
                good_matches.append(goodimg)
                
                trows,tcols = small_edge.shape[:2]
                edit_image = image.copy()
                cv2.rectangle(edit_image, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2)
                
                print "writing frame %s at time %s" % (frame, frame_time)
                cv2.imwrite("%s/%s_%s_%s.jpg" % (directory, goodimg, frame, frame_time), edit_image)
                
                if show_image and not already_shown:
                    cv2.imshow('output', edit_image)
                    cv2.waitKey(0)
                    already_shown = True
                    cv2.destroyWindow('output')
                    cv2.imshow('output', edit_image)
                    
                goodimgdata['just_found'] = True
                
        else:
            goodimgdata['just_found'] = False
            
    if (generate_videos or aggregate_video) and good_matches:
        sm_start = max(frame_total_secs - predelay, 0)
        sm_end = min(frame_total_secs + postdelay, movielen)
        sub_movie = movie.subclip(sm_start, sm_end)
        
        match_str = ','.join(good_matches)
        #if caption:
        txt_clip = TextClip(match_str, fontsize=55, color='white')
        txt_clip = txt_clip.set_position(('left', 'top')).set_duration(3)
        sub_movie = CompositeVideoClip([sub_movie, txt_clip])
        
        if generate_videos:
            sub_movie.write_videofile("%s/%s_%s.mp4" % (vdirectory, match_str, frame_time))
        else:
            video_list.append(sub_movie)
        
    return

success = True

if use_frames:
    while success:
        for f in sorted(glob.glob(adirectory + '/frame_*.jpg'), key=os.path.getmtime):
            count = int(f.split('/')[-1].split('_')[1])
            if (count <= maxframes) or (maxframes == 0):
                image = cv2.imread(f)
                matchImg(image, count)
            else:
                success = False
        
        success = False

else:        
    count = 0
    loop = frameskip
    while success:
        if (count <= maxframes) or (maxframes == 0):
            if loop == frameskip:
                success, image = vidcap.read()
                
                matchImg(image, count)
                    
                count += 1
                loop = 0
            else:
                success = vidcap.grab()
                count += 1
                loop += 1
        else:
            success = False


vidcap.release()

if aggregate_video:
    final_clip = concatenate_videoclips(video_list)
    final_clip.write_videofile("%s/%s.mp4" % (vdirectory, good_dict_str))

