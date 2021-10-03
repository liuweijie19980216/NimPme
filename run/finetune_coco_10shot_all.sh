python3 -m tools.train_net --num-gpus 1 \
        --config-file /home/liuwj/Repository/few-shot-object-detection/configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_10shot.yaml
        --opts MODEL.WEIGHTS /home/liuwj/Repository/few-shot-object-detection/checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all1/model_reset_surgery.pth
