# gradCAM_layer
gradCAM visualization for a specific layer

#### Visualize grad-CAM image
I used a [pretrained model](https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k) from hugging face in code script.

Extract grad-CAM image, here taking `dog.png` as an example, you can choose another image with input image size 448 x 448
```bash
python grad_cam.py --image-path=dog.png --use-cuda
```

<img src="images/grad_ImageNetV2.png"  width="300" height="400">

Source code: https://github.com/jacobgil/pytorch-grad-cam