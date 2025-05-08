<h1>Adapting SAM for Surgical Instrument Tracking and Segmentation in Endoscopic Submucosal Dissection Videos </h1>

<h2>Abstract</h2>
The precise tracking and segmentation of surgical instruments have led to a remarkable enhancement in the efficiency of surgical procedures. However, the challenge lies in achieving accurate segmentation of surgical instruments while minimizing the need for manual annotation and reducing the time required for the segmentation process. To tackle this, we propose a novel framework for surgical instrument segmentation and tracking. Specifically, with a tiny subset of frames for segmentation, we ensure accurate segmentation across the entire surgical video. Our method adopts a two-stage approach to efficiently segment videos. Initially, we utilize the Segment-Anything (SAM) model, which has been fine-tuned using the Low-Rank Adaptation (LoRA) on the EndoVis17 Dataset. The fine-tuned SAM model is applied to segment the initial frames of the video accurately. Subsequently, we deploy the XMem++ tracking algorithm to follow the annotated frames, thereby facilitating the segmentation of the entire video sequence. This workflow enables us to precisely segment and track objects within the video. Through extensive evaluation of the in-distribution dataset (EndoVis17) and the out-of-distribution datasets (EndoVis18 \& the endoscopic submucosal dissection surgery (ESD) dataset), our framework demonstrates exceptional accuracy and robustness, thus showcasing its potential to advance the automated robotic-assisted surgery.

## Citation

If you find our code or paper useful, please cite as


      @misc{yu2024adapting,
      title={Adapting SAM for Surgical Instrument Tracking and Segmentation in Endoscopic Submucosal Dissection Videos}, 
      author={Jieming Yu and Long Bai and Guankun Wang and An Wang and Xiaoxiao Yang and Huxin Gao and Hongliang Ren},
      year={2024},
      eprint={2404.10640},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
      }



<h2>Acknowledgements</h2>
Thanks [SAM LoRA](https://github.com/MathieuNlp/Sam_LoRA.git) and [XMem++](https://github.com/max810/XMem2.git) 


