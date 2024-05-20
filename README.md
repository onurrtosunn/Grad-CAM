## YOLOv7 Heatmap Visualization with Grad-CAM

This repository provides a tool for visualizing the decision-making process of a YOLOv7 model using various Class Activation Mapping (CAM) techniques such as Grad-CAM, Grad-CAM++, and XGrad-CAM. The visualization helps in understanding which parts of an image the model focuses on while making predictions.

### Requirements

- Python 3.x
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- tqdm
- PIL (Pillow)
- pytorch_grad_cam


### Usage

1. **Prepare the weights and configuration file**:
    - Ensure you have the YOLOv7 weights file (`yolov7.pt`) and the corresponding configuration file (`cfg/training/yolov7.yaml`).

2. **Edit the parameters if needed**:
    - Update the `get_params` function in the script if you need to change the default parameters such as weight file, configuration file, device, CAM method, target layer, backward type, confidence threshold, and ratio.

3. **Run the script**:
    - Use the following command to generate heatmaps for the specified image:
    ```sh
    python gradcam.py
    ```
    - By default, it will process `inference/images/image3.jpg` and save the results in the `result` directory.

### Parameters

- **weight**: Path to the YOLOv7 weights file.
- **cfg**: Path to the YOLOv7 configuration file.
- **device**: The device to run the model on (`'cpu'` or `'cuda'`).
- **method**: The CAM method to use (`'GradCAM'`, `'GradCAMPlusPlus'`, or `'XGradCAM'`).
- **layer**: The target layer for the CAM method (e.g., `'model.model[-2]'`).
- **backward_type**: Type of backward operation (`'class'` or `'conf'`).
- **conf_threshold**: Confidence threshold for detections.
- **ratio**: Ratio of the top predictions to visualize.

### Example

```python
def get_params():
    params = {
        'weight': 'yolov7.pt',
        'cfg': 'cfg/training/yolov7.yaml',
        'device': 'cpu',
        'method': 'GradCAM',  # GradCAMPlusPlus, GradCAM, XGradCAM
        'layer': 'model.model[-2]',
        'backward_type': 'class',  # class or conf
        'conf_threshold': 0.6,  # 0.6
        'ratio': 0.02  # 0.02-0.1
    }
    return params

if __name__ == '__main__':
    model = yolov7_heatmap(**get_params())
    model('inference/images/image3.jpg', 'result')
