import torch

from ..utils import box_utils
from .data_preprocessing import PredictionTransform
from ..utils.misc import Timer
from torch.utils.data import DataLoader


class Predictor:
    def __init__(self, net, size, mean=0.0, std=1.0, nms_method=None,
                 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
        self.net = net
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method

        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net.to(self.device)
        self.net.eval()

        self.timer = Timer()

    def predict(self, imageList, batch_size=1, top_k=-1, prob_threshold=None):
        cpu_device = torch.device("cpu")

        dataArray = []

        for image in imageList:
            height, width, _ = image.shape
            trans_image = self.transform(image)
            dataElement = {"img": trans_image, "h": height, "w": width}
            dataArray.append(dataElement)

        dataloader = DataLoader([x["img"] for x in dataArray], batch_size=batch_size)

        for i, data in enumerate(dataloader):
            dimages = data
            gimages = dimages.to(self.device)

            self.timer.start()
            scores, boxes = self.net.forward(gimages)
            print("Inference time: ", self.timer.end())

        result_batch = boxes.size(dim=0)
        score_batch = scores.shape

        batch_res_list = []

        cb = boxes.to(cpu_device)
        cs = scores.to(cpu_device)

        for i in range(0, int(result_batch), 1):
            iboxes = cb[i]
            iscores = cs[i]
            if not prob_threshold:
                prob_threshold = self.filter_threshold
            picked_box_probs = []
            picked_labels = []
            for class_index in range(1, iscores.size(1)):
                probs = iscores[:, class_index]
                mask = probs > prob_threshold
                probs = probs[mask]
                if probs.size(0) == 0:
                    continue
                subset_boxes = iboxes[mask, :]
                box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
                box_probs = box_utils.nms(
                    box_probs,
                    self.nms_method,
                    score_threshold=prob_threshold,
                    iou_threshold=self.iou_threshold,
                    sigma=self.sigma,
                    top_k=top_k,
                    candidate_size=self.candidate_size,
                )
                picked_box_probs.append(box_probs)
                picked_labels.extend([class_index] * box_probs.size(0))
            if not picked_box_probs:
                return torch.tensor([]), torch.tensor([]), torch.tensor([])
            picked_box_probs = torch.cat(picked_box_probs)
            picked_box_probs[:, 0] *= dataArray[i]["w"]
            picked_box_probs[:, 1] *= dataArray[i]["h"]
            picked_box_probs[:, 2] *= dataArray[i]["w"]
            picked_box_probs[:, 3] *= dataArray[i]["h"]
            result = (
                picked_box_probs[:, :4],
                torch.tensor(picked_labels),
                picked_box_probs[:, 4],
            )
            batch_res_list.append(result)

        return batch_res_list
