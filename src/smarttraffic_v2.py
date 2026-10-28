"""SmartTraffic AI V2: YOLO11 motorcycle ByteTrack + helmet detection.
Run: python smarttraffic_v2.py --video raw.mp4 --helmet-model best.pt --output output
Requires: pip install ultralytics opencv-python-headless numpy
"""
import argparse, csv, time
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO


def class_ids(names):
    no_ids, yes_ids = [], []
    for idx, name in names.items():
        n = name.lower().replace('-', ' ').replace('_', ' ').strip()
        if ('helmet' in n and any(s in n for s in ('no ', 'without ', 'not ', 'non '))) or n in ('nohelmet', 'withouthelmet'):
            no_ids.append(int(idx))
        elif 'helmet' in n:
            yes_ids.append(int(idx))
    return yes_ids, no_ids


def rider_region(box, W, H):
    x1,y1,x2,y2 = map(float,box)
    bw,bh=x2-x1,y2-y1
    # The rider's head is generally above the motorcycle bbox; approximate, camera-dependent.
    return (max(0,x1-.30*bw), max(0,y1-1.25*bh), min(W,x2+.30*bw), min(H,y2+.12*bh))


def match_heads(heads, bikes, W, H):
    """Greedy one-to-one spatial assignment; returns (head_idx, bike_idx) pairs."""
    pairs=[]
    for hi,(hb,cls,score) in enumerate(heads):
        hx=(hb[0]+hb[2])/2; hy=(hb[1]+hb[3])/2
        for bi,(tid,bb,_) in enumerate(bikes):
            rx1,ry1,rx2,ry2=rider_region(bb,W,H)
            if not (rx1<=hx<=rx2 and ry1<=hy<=ry2):
                continue
            # normalized horizontal and vertical distance from expected head location
            bx=(bb[0]+bb[2])/2
            expected_y=bb[1]-.30*(bb[3]-bb[1])
            dx=abs(hx-bx)/max(1,bb[2]-bb[0])
            dy=abs(hy-expected_y)/max(1,bb[3]-bb[1])
            pairs.append((dx+0.45*dy,hi,bi))
    assigned_h=set(); assigned_b=set(); result=[]
    for _,hi,bi in sorted(pairs):
        if hi not in assigned_h and bi not in assigned_b:
            result.append((hi,bi));assigned_h.add(hi);assigned_b.add(bi)
    return result


def label(frame,txt,x,y,color,scale=.56):
    font=cv2.FONT_HERSHEY_SIMPLEX
    (tw,th),baseline=cv2.getTextSize(txt,font,scale,2)
    x=int(np.clip(x,0,max(0,frame.shape[1]-tw-12)))
    y=int(np.clip(y,th+12,frame.shape[0]-5))
    cv2.rectangle(frame,(x,y-th-9),(x+tw+10,y+baseline+4),(18,24,33),-1)
    cv2.putText(frame,txt,(x+5,y-3),font,scale,color,2,cv2.LINE_AA)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--video',required=True)
    p.add_argument('--helmet-model',required=True)
    p.add_argument('--output',default='/content/SmartTrafficAI_V2')
    p.add_argument('--bike-conf',type=float,default=.22)
    p.add_argument('--helmet-conf',type=float,default=.30)
    p.add_argument('--confirm-frames',type=int,default=4)
    p.add_argument('--imgsz',type=int,default=960)
    args=p.parse_args()
    out=Path(args.output); evidence=out/'evidence'; evidence.mkdir(parents=True,exist_ok=True)
    bike_model=YOLO('yolo11n.pt'); helmet_model=YOLO(args.helmet_model)
    yes_ids,no_ids=class_ids(helmet_model.names)
    if not no_ids: raise ValueError(f'No no-helmet class found: {helmet_model.names}')
    print('Helmet model classes:',helmet_model.names,'No-helmet IDs:',no_ids)
    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise RuntimeError('Cannot open input video')
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS) or 25
    if W<1 or H<1: raise RuntimeError('Invalid video dimensions')
    output_video=out/'smarttraffic_v2.mp4'
    writer=cv2.VideoWriter(str(output_video),cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H))
    if not writer.isOpened(): raise RuntimeError('Cannot open output video writer')
    state=defaultdict(lambda:{'streak':0,'last':-999,'confirmed':False})
    seen=set(); events=[]; frame_no=0; start=time.monotonic()
    try:
        while True:
            ok,raw=cap.read()
            if not ok:break
            frame_no+=1; frame=raw.copy()
            tr=bike_model.track(raw,persist=True,tracker='bytetrack.yaml',classes=[3],conf=args.bike_conf,imgsz=args.imgsz,verbose=False)[0]
            hr=helmet_model.predict(raw,conf=args.helmet_conf,imgsz=args.imgsz,verbose=False)[0]
            bikes=[]; heads=[]
            if tr.boxes is not None and tr.boxes.id is not None:
                for b,tid,cf in zip(tr.boxes.xyxy.cpu().numpy(),tr.boxes.id.int().cpu().tolist(),tr.boxes.conf.cpu().tolist()):
                    bikes.append((int(tid),b,float(cf))); seen.add(int(tid))
            if hr.boxes is not None:
                for b,cls,cf in zip(hr.boxes.xyxy.cpu().numpy(),hr.boxes.cls.int().cpu().tolist(),hr.boxes.conf.cpu().tolist()):
                    if cls in yes_ids or cls in no_ids: heads.append((b,int(cls),float(cf)))
            matches=match_heads(heads,bikes,W,H)
            bike_to_head={bi:hi for hi,bi in matches}
            for bi,(tid,b,cf) in enumerate(bikes):
                x1,y1,x2,y2=map(int,b); s=state[tid]
                hi=bike_to_head.get(bi)
                no_helmet=hi is not None and heads[hi][1] in no_ids
                # Only continuous matched no-helmet frames increase the streak.
                s['streak']=s['streak']+1 if no_helmet and s['last']==frame_no-1 else (1 if no_helmet else 0)
                s['last']=frame_no
                if s['streak']>=args.confirm_frames and not s['confirmed']:
                    s['confirmed']=True
                    filename=f'no_helmet_bike_{tid}_frame_{frame_no}.jpg'
                    # Save original frame with detection context in a separate labeled copy below.
                    events.append({'frame':frame_no,'seconds':round((frame_no-1)/fps,2),'track_id':tid,'violation':'No Helmet','evidence':filename})
                    cv2.imwrite(str(evidence/filename),raw)
                color=(45,70,235) if s['confirmed'] else (255,190,55)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                status='VIOLATION' if s['confirmed'] else ('NO HELMET?' if no_helmet else 'TRACKED')
                label(frame,f'BIKE #{tid} | {status}',x1,y1-6,color,.48)
            for b,cls,cf in heads:
                x1,y1,x2,y2=map(int,b); is_no=cls in no_ids
                color=(65,65,245) if is_no else (65,220,100)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                label(frame,f'{"NO HELMET" if is_no else "HELMET"} {cf:.2f}',x1,y1-7,color,.47)
            # Semi-transparent dashboard
            overlay=frame.copy(); panel_w=min(W,490); panel_h=min(H,132)
            cv2.rectangle(overlay,(0,0),(panel_w,panel_h),(15,22,32),-1)
            cv2.addWeighted(overlay,.83,frame,.17,0,frame)
            label(frame,'SMARTTRAFFIC AI  |  V2',12,30,(255,255,255),.75)
            label(frame,f'Visible bikes: {len(bikes)}  |  Unique IDs: {len(seen)}',12,69,(255,220,150),.53)
            label(frame,f'Confirmed violations: {len(events)}  |  Time: {(frame_no-1)/fps:.1f}s',12,105,(75,145,255),.53)
            writer.write(frame)
            if frame_no%100==0:print(f'Processed {frame_no} frames')
    finally:
        cap.release();writer.release()
    with (out/'violations.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['frame','seconds','track_id','violation','evidence']);w.writeheader();w.writerows(events)
    print('Finished:',frame_no,'frames;',len(seen),'unique track IDs;',len(events),'confirmed violations')
    print('Output:',output_video,'Elapsed:',round(time.monotonic()-start,1),'sec')
    print('Note: synthetic footage and heuristic rider matching are not ground-truth evaluation.')

if __name__=='__main__':main()
