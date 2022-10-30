# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python path/to/val.py --weights yolov5s.pt                 # PyTorch
                                      yolov5s.torchscript        # TorchScript
                                      yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s.xml                # OpenVINO
                                      yolov5s.engine             # TensorRT
                                      yolov5s.mlmodel            # CoreML (MacOS-only)
                                      yolov5s_saved_model        # TensorFlow SavedModel
                                      yolov5s.pb                 # TensorFlow GraphDef
                                      yolov5s.tflite             # TensorFlow Lite
                                      yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse                  # 解析命令行参数模块
import json                      # 实现字典列表和JSON字符串之间的相互解析
import os                        # 与操作系统进行交互的模块 包含文件路径操作和解析
import sys                       # sys系统模块 包含了与Python解释器和它的环境有关的函数
from pathlib import Path         # Path将str转换为Path对象 使字符串路径易于操作的模块
from threading import Thread     # 线程操作模块

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.general import (LOGGER, box_iou, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    # 获得对应图片的长和宽
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        # xyxy格式->xywh, 并对坐标进行归一化处理
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        # 保存预测坐标，类别，置信度
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
    # coco格式json文件大概包含信息如上
    # 获取图片id
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    # box转换为xywh格式
    box = xyxy2xywh(predn[:, :4])  # xywh
    """
    值得注意的是，之前所说的xyxy格式为左上角右下角的坐标
    xywh是中心点坐标和长和宽
    而coco的json格式中的框坐标格式为xywh,此处的xy为左上角坐标
    也就是coco的json格式的坐标格式为：左上角坐标+长宽
    所以下面一行代码就是将：中心点坐标->左上角
    """
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    """
    image_id：图片id, 即属于哪张图
    category_id: 类别, coco91class()从索引0~79映射到索引0~90
    bbox：框的坐标
    score：置信度
    """
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        ):
    """
        :param model: 测试的模型，训练时调用val传入
        :param dataloader: 测试集的dataloader，训练时调用val传入
        :param save_dir: 保存在测试时第一个batch的图片上画出标签框和预测框的图片路径
        :param plots: 是否绘制各种可视化，比如测试预测，混淆矩阵，PR曲线等
        :param wandb_looger: wandb可视化工具, train的时候传入
        :param compute_loss: 计算损失的对象实例, train的时候传入
        """
    # Initialize/load model and set device
    # 判断是否在训练时调用val，如果是则获取训练时的设备
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model

        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        # 选择设备
        device = select_device(device, batch_size=batch_size)

        # Directories
        # 获取保存日志路径
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        # 加载模型
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
        stride, pt, jit, onnx, engine = model.stride, model.pt, model.jit, model.onnx, model.engine
        # 检查输入图片分辨率是否能被32整除
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        # 如果设备不是cpu且opt.half=True，则将模型由Float32转为Float16，提高前向传播的速度
        half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            model.model.half() if half else model.model.float()
        elif engine:
            batch_size = model.batch_size
        else:
            half = False
            batch_size = 1  # export.py models default to batch-size 1
            device = torch.device('cpu')
            LOGGER.info(f'Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    # 判断是否为coco数据集
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    # 设置iou阈值，从0.5~0.95，每间隔0.05取一次
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()  # iou个数

    # Dataloader
    if not training:
        # 创建一个全0数组测试一下前向传播是否正常运行
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz), half=half)  # warmup
        pad = 0.0 if task in ('speed', 'benchmark') else 0.5
        rect = False if task == 'benchmark' else pt  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride, single_cls, pad=pad, rect=rect,
                                       workers=workers, prefix=colorstr(f'{task}: '))[0]
    # 初始化测试的图片数量
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)  # 声明混淆矩阵对象实例
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}  # 获取类别的名字
    """
    获取coco数据集的类别索引
    这里要说明一下，coco数据集有80个类别(索引范围应该为0~79)，
    但是他的索引却属于0~90
    coco80_to_coco91_class()就是为了与上述索引对应起来，返回一个范围在0~90的索引数组
    """
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    # 设置tqdm进度条的显示信息
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    # 初始化指标，时间
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # 初始化测试集的损失
    loss = torch.zeros(3, device=device)
    # 初始化json文件的字典，统计信息，ap, wandb显示图片
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        """
        time_synchronized()函数里面进行了torch.cuda.synchronize()，再返回的time.time()
        torch.cuda.synchronize()等待gpu上完成所有的工作
        总的来说就是这样测试时间会更准确 
        """
        t1 = time_sync()
        if pt or jit or engine:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # 图片也由Float32->Float16
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1  # 计算数据拷贝，类型转换以及除255的时间

        # Inference
        # 前向传播
        # out为预测结果, train_out训练结果
        out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
        dt[1] += time_sync() - t2 # 计算推理时间

        # Loss
        # 如果是在训练时进行的val，则通过训练结果计算并返回测试集的box, obj, cls损失
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # NMS
        # 将归一化标签框反算到基于原图大小，如果设置save-hybrid则传入nms函数
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)  # 进行nms
        dt[2] += time_sync() - t3

        # Metrics
        # 为每一张图片做统计, 写入预测信息到txt文件, 生成json文件字典, 统计tp等
        for si, pred in enumerate(out):
            # 获取第si张图片的标签信息, 包括class,x,y,w,h
            # targets[:, 0]为标签属于哪一张图片的编号
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []   # 获取标签类别
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1  # 统计测试图片数量
            # 如果预测为空，则添加空的信息到stats里
            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            # 反算坐标 基于input-size -> 基于原图大小
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # 获得xyxy格式的框
                # 反算坐标 基于input-size -> 基于原图大小
                # 值得注意的是在进行nms之前，将归一化的标签框反算为基于input-size的标签框了
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    # 混淆矩阵的处理
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                # 反算坐标 基于input-size -> 基于原图大小
                # 值得注意的是在进行nms之前，将归一化的标签框反算为基于input-size的标签框了
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            # 每张图片的结果统计到stats里
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # 保存预测结果为txt文件
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json:
                # 保存coco格式的json文件字典
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        # Plot images
        # 画出前三个batch的图片的ground truth和预测框并保存
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(im, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(im, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute metrics
    # 将stats列表的信息拼接到一起
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        # 根据上面得到的tp等信息计算指标
        # 精准度TP/TP+FP，召回率TP/P，map，f1分数，类别
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        # nt是一个列表，测试集每个类别有多少个标签框
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    # 打印指标结果
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    # 细节展示每一个类别的指标
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    # 打印前处理时间，前向传播耗费的时间、nms的时间
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        # 绘制混淆矩阵
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Save JSON
    # 采用之前保存的json格式预测结果，通过cocoapi评估指标
    # 需要注意的是 测试集的标签也需要转成coco的json格式
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        # 获取标签json文件路径
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        # 获取预测框的json文件路径并保存
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            # 加载标签json文件, 预测json文件
            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            # 创建评估器
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()  # 评估
            eval.accumulate()
            # 展示结果
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    # 返回测试指标结果
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    """
    opt参数详解
    data: 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息
    weights: 模型的权重文件地址 weights/yolov5s.pt
    batch_size: 前向传播的批次大小 默认32
    imgsz: 输入网络的图片分辨率 默认640
    conf-thres: object置信度阈值 默认0.001
    iou-thres: 进行NMS时IOU的阈值 默认0.6
    task: 设置测试的类型 有train, val, test, speed or study几种 默认val
    device: 测试的设备
    single-cls: 数据集是否只用一个类别 默认False
    augment: 测试是否使用TTA Test Time Augment 默认False
    verbose: 是否打印出每个类别的mAP 默认False
    下面三个参数是auto-labelling(有点像RNN中的teaching forcing)相关参数详见:https://github.com/ultralytics/yolov5/issues/1563 下面解释是作者原话
    save-txt: traditional auto-labelling
    save-hybrid: save hybrid autolabels, combining existing labels with new predictions before NMS (existing predictions given confidence=1.0 before NMS.
    save-conf: add confidences to any of the above commands
    save-json: 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签） 默认False
    project: 测试保存的源文件 默认runs/test
    name: 测试保存的文件地址 默认exp  保存在runs/test/exp下
    exist-ok: 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹
    half: 是否使用半精度推理 默认False
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()  # 解析上述参数
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')  # |或 左右两个变量有一个为True 左边变量就为True
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    # 检查环境
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    # task = 'val', 'test', 'study', 'speed'
    # task = ['val', 'test']时就正常测试验证集、测试集
    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        # 评估模型速度
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)
        # task == 'study'时，就评估模型在各个尺度下的指标并可视化
        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
