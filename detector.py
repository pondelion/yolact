from typing import Optional, List

import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np

from yolact_model import Yolact
from yolact_data.config import set_cfg
from yolact_utils.augmentations import FastBaseTransform
from yolact_layers.output_utils import postprocess


def _cdn(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()


class Detector:
    
    def __init__(
        self,
        cfg_name: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        thresh: float = 0.5,
    ):
        self._cfg_name = cfg_name
        self._weights_path = None
        self._device = device
        set_cfg(cfg_name)
        self._model = Yolact()
        self._model.to(self._device)
        if self._device == 'cuda':
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        self._thresh = 0.5
        self._TOK_K = 30

    def load_weights(self, weights_path: str):
        if self._weights_path == weights_path:
            print(f'weights {weights_path} is already loaded, skipping.')
            return
        self._model.load_weights(weights_path)
        self._model.eval()

    def detect_imgfile(
        self,
        img_filepath: str,
        thresh: Optional[float] = None,
    ) -> List[np.ndarray]:
        frame = torch.from_numpy(cv2.imread(img_filepath)).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = self._model(batch)
        if thresh is None:
            thresh = self._thresh
        
        img_gpu = frame / 255.0
        h, w, _ = frame.shape
        classes, scores, boxes, masks = postprocess(
            preds, w, h, crop_masks=False, score_threshold=thresh
        )
        sorted_indices = scores.argsort(0, descending=True)[:self._TOK_K]
        si = sorted_indices
        return _cdn(classes[si]), _cdn(scores[si]), _cdn(boxes[si]), _cdn(masks[si])
