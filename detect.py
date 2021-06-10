import torch
import numpy as np
import cv2
from face_detection.data import cfg_re50
from face_detection.layers.functions.prior_box import PriorBox
from face_detection.utils.nms.py_cpu_nms import py_cpu_nms
from face_detection.models.retinaface import RetinaFace
from face_detection.utils.box_utils import decode, decode_landm

CONFIDENCE_THRESHOLD = 0.2
TOP_K = 5000
NMS_THRESHOLD = 0.6
KEEP_TOP_K = 750
VIS_THRES = 0.5

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def detect(img_raw):
    #preprocessing
    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    # img = img.to(device)
    # scale = scale.to(device)

    #Forward
    loc, conf, landms = net(img)  

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    # priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale 
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    # scale1 = scale1.to(device)
    landms = landms * scale1
    landms = landms.cpu().numpy()

    inds = np.where(scores > CONFIDENCE_THRESHOLD)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    order = scores.argsort()[::-1][:TOP_K]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, NMS_THRESHOLD)
    # keep = nms(dets, args.NMS_THRESHOLD,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    dets = dets[:KEEP_TOP_K, :]
    landms = landms[:KEEP_TOP_K, :]

    dets = np.concatenate((dets, landms), axis=1)

    img_copy = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    cropped_faces = []
    for b in dets:
        if b[4] < VIS_THRES:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        # cv2.rectangle(img_copy, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cropped_faces.append(img_copy[b[1]:b[3], b[0]:b[2], :])

    return cropped_faces


torch.set_grad_enabled(False)
cfg = cfg_re50
# net and model
net = RetinaFace(cfg=cfg, phase = 'test')
net = load_model(net, 'weights/Resnet50_Final.pth', True)
net.eval()
print('Finished loading model!')
