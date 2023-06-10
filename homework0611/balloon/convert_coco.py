import os.path as osp
import mmcv
from mmengine.config import Config
from mmengine.utils import track_iter_progress
import mmengine
import json

def convert_balloon_to_coco(ann_file, out_file, image_prefix):    
    data_infos = Config.fromfile(ann_file)
    # data_infos = json.load(open(ann_file))
    
    annotations = [] 
    images = []
    obj_count = 0
    for idx, v in enumerate(track_iter_progress(data_infos.values())):
        filename = v['filename']
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]
        
        images.append(dict(id = idx, 
                           file_name = filename, 
                           height=height,
                           width = width))
        
        bboxes = []
        labels = []
        masks = []
        for _, obj in v['regions'].items(): 
            assert not obj['region_attributes'] 
            obj = obj['shape_attributes']
            px = obj['all_points_x']
            py = obj['all_points_y']  
            poly = [(x+0.5, y+0.5) for x,y in zip(px,py)]            
            poly = [p for  x in poly for p in x]
            
            x_min, y_min, x_max, y_max = (min(px), min(py), max(px),max(py)) 
            
            data_anno = dict(image_id = idx,
                             id = obj_count,
                             category_id = 0, 
                             bbox = [x_min, y_min, x_max-x_min, y_max-y_min],
                             area = (x_max - x_min)*(y_max - y_min),
                             segmentation = [poly], 
                             iscrowd =0)
            annotations.append(data_anno) 
            obj_count += 1
            
    coco_format_json = dict(images = images,
                            annotations = annotations,
                            categories=[{'id':0, 'name':'balloon'}])    
    mmengine.dump(coco_format_json, out_file)
    # 对验证集数据进行处理是，将下面路径中的train 替换成val 即可
    # # 注意数据集 balloon 的路径自行调整ann_file = './balloon/train/via_region_data.json'
    # out_file = './balloon/train/annotation_coco.json'
    # image_prefix = './balloon/train'
    # convert_balloon_to_coco(ann_file, out_file, image_prefix)
    
convert_balloon_to_coco('data/balloon/train/via_region_data.json', 'data/balloon/train/train_ball.json', 'data/balloon/train')
convert_balloon_to_coco('data/balloon/val/via_region_data.json', 'data/balloon/val/val_ball.json', 'data/balloon/val')