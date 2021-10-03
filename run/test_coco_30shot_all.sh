python3 -m tools.test_net --num-gpus 1 \
        --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_30shot.yaml \
        --eval-only |tee coco_30shot.txt