python3 -m tools.ckpt_surgery \
        --src1 checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
        --method randinit \
        --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all1 -- coco

python3 -m tools.ckpt_surgery \
        --src1 checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
        --method remove \
        --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all1 --coco


python3 -m tools.ckpt_surgery \
        --src1 checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
        --src2 checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_10shot/model_final.pth \
        --method combine \
        --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all1 --coco
