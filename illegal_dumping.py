import argparse
import time
from pathlib import Path
from datetime import datetime
import pytz
import ipywidgets as widgets
from IPython.display import display

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, non_max_suppression_kpt, load_model
from utils.plots import output_to_keypoint, plot_skeleton_kpts, colors, plot_one_box_kpt, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}


##########################################################################################
def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    if label == 0: #person
        color = (255,255,255)
        
    else: # trash
        color = (153, 153, 103)
        
    return tuple(color)

#................................ CircleQueue .............................
class CircularQueue():
    
    def __init__(self, max = 3):
        self.max = max
        self.queue = [0] * self.max
        self.size = self.front = 0
        self.rear = None
    
    def is_empty(self):
        return self.size == 0
    
    def is_full(self):
        if self.rear == None:
            return False
        return self.next_index(self.rear) == self.front
    
    def next_index(self, idx):
        return (idx+1) % self.max
    
    def enqueue(self, data):
        if self.is_full():
            raise Exception("Queue is Full")
        if self.rear == None:
            self.rear = 0
        else:
            self.rear = self.next_index(self.rear)
        
        self.queue[self.rear] = data
        self.size += 1
        return self.queue[self.rear]
    
    def dequeue(self):
        if self.is_empty():
            raise Exception("Queue is Empty")
        self.queue[self.front] = None
        self.front = self.next_index(self.front)
        return self.queue[self.front]
    
    def display(self):
        print(self.queue)
    
    def peek(self):
        if self.is_empty():
            raise Exception("Queue is Empty")
        return self.queue[(self.front) % self.max]
    
    def out_peek(self):
        if self.is_empty():
            raise Exception("Queue is Empty")
        return self.queue[((self.front+1) % self.max):]

def draw_boxes(img, bbox, names ,object_id, identities=None, offset=(0, 0, 0, 0), id_with_trash=None, alarm=None):

    height, width, _ = img.shape

    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[1]
        y1 += offset[2]
        y2 += offset[3]
        cat = int(object_id[i]) if object_id is not None else 0
        id = int(identities[i]) if identities is not None else 0
        cen = (int((x1+x2)/2), int(y2))
        wit = x2-x1

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int(y2))

        # create new buffer for new object
        if id not in data_deque:  
            data_deque[id] = deque(maxlen= opt.trailslen)

        color = compute_color_for_labels(object_id[i])
        
        if id_with_trash == int(id):
            label = str(id) + ":"+ names[cat]
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.putText(img, f"{label}(dumping candidate)", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (204,153,102), 2)
            cv2.ellipse(img, cen, axes=(int(wit), int(0.35 * wit)), angle=0.0, 
                        startAngle=-45, endAngle=235, color = (204,153,102), thickness=5)
            
        else:
            label = str(id) + ":"+ names[cat]
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            if names[cat] == 'trash':
                cv2.rectangle(img, (int(x1), int(y1+h)), (int(x1+w), int(y1)), (153, 153, 103), -1, cv2.LINE_AA)
                cv2.putText(img, label, (int(x1), int(y1+h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            else:  
                cv2.putText(img, label, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.ellipse(img, cen, axes=(int(wit), int(0.35 * wit)), angle=0.0,
                        startAngle=-45, endAngle=235, color = color, thickness=2)
            (d_w, d_h), _ = cv2.getTextSize("DUMPER", cv2.FONT_HERSHEY_SIMPLEX, 3, 3)
            if alarm:
                cv2.putText(img, "DUMPER", (int(width-d_w), int(d_h)), cv2.FONT_HERSHEY_SIMPLEX, 
                        3, (255, 102, 102), 5)     

        data_deque[id].appendleft(center)
        
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(opt.trailslen / float(i + i)) * 1.5)
            if id_with_trash == int(id):
                color = (204,153,102)
            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
         
    return img

def detect(save_img=False):
    korea_time = 'Asia/Seoul'
    tz = pytz.timezone(korea_time)
    dump_frame = []
    names, source, weights, weights2, view_img, save_txt, imgsz, trace = None, opt.source, opt.weights, opt.weights2, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace 
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # initialize deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    model2 = attempt_load(weights2, map_location=device)
    _ = model2.eval()
    names2 = model2.module.names if hasattr(model2, 'module') else model2.names
    stride2 = int(model2.stride.max()) 

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        model2(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    cq = CircularQueue(30)
    #print('CircularQueue CEHCK CircularQueue CHECK')
    #print(cq.display())
    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device).float()
        #img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]
                model2(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
            pred2, _ = model2(img, augment=opt.augment)

        t2 = time_synchronized()

        pred = non_max_suppression_kpt(pred,   #Apply non max suppression
                                            0.9,   # Conf. Threshold.
                                            0.65, # IoU Threshold.
                                            nc=model.yaml['nc'], # Number of classes.
                                            nkpt =model.yaml['nkpt'], # Number of keypoints.
                                            kpt_label=True)
        pred2 = non_max_suppression(pred2, opt.conf_thres, opt.iou_thres, classes=1, agnostic=opt.agnostic_nms)

        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        for i, det in enumerate(zip(pred, pred2)):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            #print('im0.shape',im0.shape)
            if len(det[0]):
                # Rescale boxes from img_size to im0 size
                #det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                det[0][:, :4] = scale_coords(img.shape[2:], det[0][:, :4], im0.shape, kpt_label=False)
                det[0][:, 6:] = scale_coords(img.shape[2:], det[0][:, 6:], im0.shape, kpt_label=True, step=3)
                det[1][:, :4] = scale_coords(img.shape[2:], det[1][:, :4], im0.shape, kpt_label=False)

                # Print results
                for c in det[0][:, -1].unique():
                    n = (det[0][:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                xywh_bboxs = []
                confs = []
                oids = []
                xywh_bboxs2 = [] ## 인간
                confs2 = []
                oids2 = []
                #kpts = []

                # Write results
                for i, (*xyxy, conf, cls) in enumerate(reversed(det[0][:, :6])):
                    x_c2, y_c2, bbox_w2, bbox_h2 = xyxy_to_xywh(*xyxy)
                    xywh_obj2 = [x_c2, y_c2, bbox_w2, bbox_h2]
                    xywh_bboxs2.append(xywh_obj2)
                    confs2.append([conf.item()]) ## 인간
                    oids2.append(int(cls)) ## 인간
                    kpts = det[0][i, 6:]
                    for i2, (*xyxy2, conf2, cls2) in enumerate(reversed(det[1][:, :6])):
                        ## 인간, 쓰레기
                        x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy2)
                        xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                        xywh_bboxs.append(xywh_obj)
                        ## 인간(pose_estimation)- pred
                        confs.append([conf2.item()]) ## 인간, 쓰레기
                        oids.append(int(cls2)) ## 인간, 쓰레기   

                        if save_img or view_img:  # Add bbox to image
                            plot_one_box_kpt(xyxy2, im0s, color=colors[int(cls)],
                                                line_thickness=1, kpt_label=True, kpts=kpts, steps=3, 
                                                orig_shape=im0s.shape[:2])

                xywhs = xywh_bboxs + xywh_bboxs2
                confs = confs + confs2
                oids = oids + oids2
            
                xywhs = torch.Tensor(xywhs)
                confss = torch.Tensor(confs)
                
                outputs = deepsort.update(xywhs, confss, oids, im0)
                
                count = 0
                count_trash = 0
                for i in outputs:
                    if i[5] == 1: count_trash += 1

                count = len(det[0])
                tracked_dets_ = outputs.copy()

                #........CACULATE THE DISTANCE BETWEEN TRASH AND PEOPLE........
                        ## FUNCTION 
                def central_bbox(bbox):
                    CX = (bbox[0]+bbox[2])//2
                    CY = (bbox[1]+bbox[3])//2
                    
                    return [CX, CY]

                def manhattan_distance(pt1, pt2):
                    distance = 0
                    for i in range(len(pt1)):
                        distance += abs(pt1[i] - pt2[i])
                    
                    return distance

                cl = []
                for i in range(len(outputs)):
                    cl.append(outputs[i][5])
                #...........................................................
                # 0-1. t시점 frame 내 사람과 쓰레기 *모두* detect 될 경우만 고려
                if set(cl) == {0.0, 1.0}: # [0.0, 0.0, 1.0] 이렇게 나올 경우 있으므로 set
                    # 1. result 에서 손목 좌표 가져오기
                    result_ = det[0].tolist().copy()
                    for i in range(len(result_)): # int형으로 변형
                        result_[i] = [round(j) for j in result_[i]]
                    
                    #print('------------------------------------- int(RESULT) -------------------------------------')
                
                    #print(len(result_))
                    result_.sort(key = lambda x:x[1])
                    #print(result_)

                    wrist = [] # 손목 좌표 담을 리스트
                    #count = 0 # 한 frame 내 사람은 몇 명?
                    for i in range(len(result_)):
                        if result_[i][5] == 0: # 사람인 경우
                            tmp = []
                            wrist_l = result_[i][33:35]
                            wrist_r = result_[i][36:38]
                            tmp.append(wrist_l)
                            tmp.append(wrist_r)
                            wrist.append(tmp)
                    #print('------------------------------------- LEFT wrist & RIGHT wrist -------------------------------------')
                    #print('writs : ', wrist, '# of people : ', count )                

                    # 2. tracked_dets 에서 쓰레기의 central bbox 좌표 가져오기
                    ctr_bb = [] # 쓰레기 중앙 좌표 담을 리스트
                    #count_trash = 0 # 한 frame 내 쓰레기는 몇 개?
                    for j in range(len(tracked_dets_)): 
                        if tracked_dets_[j][5] == 1: # 쓰레기인 경우
                            trash_ctr = tracked_dets_[j][:4] 
                            ctr_bb.append(central_bbox(trash_ctr))
                    #print('------------------------------------- TRASH BBOX -------------------------------------')
                    #print('trash bbox : ', ctr_bb, '# of trash : ' ,count_trash)

                    # 3. 손목과 쓰레기 간 거리 구하기 (*주의 : 사람 수 고려할 것)
                    if count > 0 and count_trash > 0 and len(ctr_bb) > 0: # 쓰레기와 사람이 있는 경우, 
                        distance = [] # 손목과 쓰레기 간 거리 담을 리스트
                        adj_distance = [] # DUMPER CANDIDATE : 거리가 가깝다면, 따로 보관할 리스트
                        idx = [] # DUMPER CANDIDATE의 idx 저장
                        tmp = [] # TO PICK UP THE BEST DUMPER CANDIDATE !!!!!!!! 
                        for i in range(count): # 사람 수 고려 
                            for j in range(count_trash):
                                # 왼쪽과 오른쪽 손목 중 더 가까운 거리 고려
                                #print(wrist[i][0], wrist[i][1] , ctr_bb[j])
                                cp = min(manhattan_distance(wrist[i][0], ctr_bb[j]), manhattan_distance(wrist[i][1], ctr_bb[j]))
                                distance.append(cp)
                                tmp.append(cp)
                        # 쓰레기와 가장 가까운 거리에 있는 사람은 누구? 그 사람이 DUMPER
                        dump_min = min(tmp)
                        if dump_min < 300:
                            adj_distance.append(dump_min)
                            if count >1 and count_trash >1:
                                for i in range(0, len(distance), count_trash):
                                    if i <= distance.index(dump_min) < i+count_trash:
                                        try:
                                            idx.append(i//2)
                                        except ZeroDivisionError:
                                            idx.append(int(0))
                            elif count == 1:
                                idx.append(int(0))
                            elif count_trash == 1:
                                idx.append(distance.index(dump_min))

                        #print('------------------------------------- DUMPER CANDIDATE -------------------------------------')
                        #print('distance : ', distance, 'adjacent distance : ', adj_distance, 'dumper idx : ', idx)

                        # 4. 쓰레기와 가까운 사람 ID 출력
                        # 4-1. 쓰레기를 가지고 있는 사람(dumper)이 있다면
                        if len(idx) > 1: 
                            ppl = [] # tracked_dets에서 사람 정보만 가져와 담을 리스트
                            for p in range(len(tracked_dets_)):
                                if tracked_dets_[p][5] == 0:
                                    ppl.append(tracked_dets_[p])
                            #print('ONLY people tracked_dets', ppl)

                            # idx 를 순회하면서, ONLY people tracked_dets에 담긴 idx번째 사람의 고유 ID 탐색
                            for k in idx:
                                id_with_trash = int(ppl[k][-2]) # 쓰레기 지닌 사람의 고유 ID
                                #print('------------------------------------- DUMPER ID ! -------------------------------------')
                                #print(int(ppl[k][-2]))
                        elif len(idx) == 1:
                            ppl = [] # tracked_dets에서 사람 정보만 가져와 담을 리스트
                            for p in range(len(tracked_dets_)):
                                if tracked_dets_[p][5] == 0:
                                    ppl.append(tracked_dets_[p])
                            #print('ONLY people tracked_dets', ppl)

                            id_with_trash = int(ppl[0][-2]) # 쓰레기 지닌 사람의 고유 ID
                            #print('------------------------------------- DUMPER ID ! -------------------------------------')
                            #print(int(ppl[0][-2]))

                        # 4-2. 쓰레기를 가지고 있는 사람이 없다면 (일정 거리 이상)
                        else:
                            id_with_trash = None # 고유 아이디는 없음

                # 0-2. t시점 frame 내 사람과 쓰레기 *모두* detect 되지 않는 경우 (거리를 구할 수 없으므로)
                else:
                    id_with_trash = None # 고유 아이디는 없음
                    
                if len(outputs) > 0:
                    
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    #print('사람이든 쓰레기든 DETECT된 것들 : 어떤 ID ?' '어떤 CLASS ?')
                    #print('ID : ', identities)
                    id_with_trash = id_with_trash
                    #print('ID WITH TRASH : ', id_with_trash)
                    object_id = outputs[:, -1]
                    alarm = False
                    if id_with_trash:
                        if cq.is_full():
                            cq.dequeue()
                        cq.enqueue(0)
                
                    elif not id_with_trash:
                        if cq.is_full():
                            ct = list(cq.queue)
                            if ct.count(0) > 0 and ct.count(0) < ct.count(1):
                                alarm = True
                                dump_frame.append(frame)
                                
                            cq.dequeue()
                        cq.enqueue(1)

                
                    draw_boxes(im0, bbox_xyxy, names2, object_id, identities, id_with_trash= id_with_trash, alarm= alarm)
                    # capture dump_trash
                    if len(dump_frame) == 1:
                        now = datetime.now(tz)
                        file_name = now.strftime("%Y-%m-%d %H:%M:%S") + ".jpg"
                        cv2.imwrite(f'/home/work/.data/capture/{file_name}', im0)
                
                    id_with_trash = None
                    
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    #print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model.pt path(s)')
    parser.add_argument('--weights2', nargs='+', type=str, default='epoch_001.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    #parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--trailslen', type=int, default=64, help='trails size (new parameter)')
    opt = parser.parse_args() 
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7-w6-pose.pt']:
                detect()
                strip_optimizer(opt.weights)
            for opt.weights2 in ['epoch_001.pt']:
                detect()
                strip_optimizer(opt.weights2)
        else:
            detect()
