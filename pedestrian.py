"""Person detection and pairwise image-space distances, independent of Colab."""
import argparse,json,itertools,math
from pathlib import Path

def distances(boxes,pixels_per_meter=None):
    if pixels_per_meter is not None and pixels_per_meter<=0:raise ValueError('pixels_per_meter must be positive')
    centers=[]
    for box in boxes:
        if len(box)!=4 or not all(math.isfinite(x) for x in box):raise ValueError('Boxes must contain four finite numbers')
        x,y,w,h=box
        if w<=0 or h<=0:raise ValueError('Box dimensions must be positive')
        centers.append((x+w/2,y+h/2))
    return [dict(pair=[i,j],distance=math.dist(centers[i],centers[j])/(pixels_per_meter or 1),unit='m (calibrated approximation)' if pixels_per_meter else 'px') for i,j in itertools.combinations(range(len(boxes)),2)]

def detect(frame,model=None,config=None,threshold=.5):
    import cv2
    if model:
        net=cv2.dnn.readNetFromTensorflow(str(model),str(config) if config else '')
        net.setInput(cv2.dnn.blobFromImage(frame,size=(300,300),swapRB=True,crop=False));raw=net.forward();height,width=frame.shape[:2];boxes=[]
        for detection in raw.reshape(-1,7):
            if int(detection[1])==1 and detection[2]>=threshold:
                x1,y1,x2,y2=detection[3:7]*[width,height,width,height];x1=max(0,float(x1));y1=max(0,float(y1));x2=min(width,float(x2));y2=min(height,float(y2))
                if x2>x1 and y2>y1:boxes.append([x1,y1,x2-x1,y2-y1])
        return boxes
    hog=cv2.HOGDescriptor();hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    if frame.shape[0]<128 or frame.shape[1]<64:return []
    boxes,weights=hog.detectMultiScale(frame,winStride=(8,8),padding=(8,8),scale=1.05)
    if len(boxes)==0:return []
    scores=[float(w) for w in weights];indices=cv2.dnn.NMSBoxes(boxes.tolist(),scores,0.0,.4)
    return [list(map(int,boxes[int(i)])) for i in indices.flatten()] if len(indices) else []

def annotate(frame,boxes,measurements):
    import cv2
    result=frame.copy();centers=[]
    for x,y,w,h in boxes:
        x,y,w,h=map(int,(x,y,w,h));cv2.rectangle(result,(x,y),(x+w,y+h),(112,230,154),2);centers.append((x+w//2,y+h//2))
    for measure in measurements:
        a,b=[centers[i] for i in measure['pair']];cv2.line(result,a,b,(88,200,240),1);mid=((a[0]+b[0])//2,(a[1]+b[1])//2)
        cv2.putText(result,f"{measure['distance']:.1f} {measure['unit']}",mid,cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    return result

if __name__=='__main__':
    import cv2
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('image',type=Path);p.add_argument('--output',type=Path,default=Path('annotated.jpg'));p.add_argument('--boxes',type=Path,help='Use supplied pixel-coordinate boxes instead of running a detector');p.add_argument('--model',type=Path,help='Optional SSD MobileNet frozen .pb graph');p.add_argument('--config',type=Path,help='Matching OpenCV .pbtxt graph configuration');p.add_argument('--pixels-per-meter',type=float);p.add_argument('--threshold',type=float,default=.5)
    args=p.parse_args();frame=cv2.imread(str(args.image))
    if frame is None:p.error('Could not read input image')
    if args.model and not args.config:p.error('--model requires the matching --config .pbtxt file')
    try:
        boxes=json.loads(args.boxes.read_text()) if args.boxes else detect(frame,args.model,args.config,args.threshold)
        measures=distances(boxes,args.pixels_per_meter)
    except (ValueError,cv2.error) as e:p.error(str(e))
    result=annotate(frame,boxes,measures)
    if not cv2.imwrite(str(args.output),result):p.error('Could not write output image')
    print(json.dumps(dict(detector='supplied_boxes' if args.boxes else 'ssd_mobilenet' if args.model else 'opencv_hog',boxes=boxes,distances=measures,output=str(args.output)),indent=2))
