## Dependencies

Python3.7+

Python-OpenCV

Pytorch1.0+

PyQT5

## Run

```
python3 Main.py
```

![2019-09-03 21-09-36屏幕截图](/img.asserts/2019-09-03 21-09-36屏幕截图.png)

“开始识别”

![one_mh1567517475081](/img.asserts/one_mh1567517475081.png)

You will be able to see the person detected and the distance between the person and the camera  from the picture.

At present, the model we are using is trained by Mobilenet2-SSD. If you want to change it into another model, you can modify it manually in the file Detect.py.

It should be noted that through our VOC test set, the evaluation of the mobilenet 2-ssd.pth model is about 67%, while that of the mobilenet 3-large-ssd.pth model is about 55%. We will make further improvements in the future.

## Update

We updated the interface, and now you can manually select the network to load, but if you want to modify and put it into your model, you need to implement it manually.

![1567656250909](/img.asserts/1567656250909.png)

 

**Make sure you choose the network before you start identifying it.**