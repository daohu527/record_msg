# record_msg
record message parse helper function. It needs to be used in combination with [**cyber_record**](https://github.com/daohu527/cyber_record)

## Parser record
First read the record file through `cyber_record`, then `record_msg` provides 3 interfaces to help parsing cyber record file.

#### csv format
you can use `to_csv` to format objects so that they can be easily saved in csv format.
```python
import csv
from record_msg.parser import to_csv

f = open("message.csv", 'w')
writer = csv.writer(f)

def parse_pose(pose):
  '''
  save pose to csv file
  '''
  line = to_csv([pose.header.timestamp_sec, pose.pose])
  writer.writerow(line)

f.close()
```

#### image
you can use `ImageParser` to parse and save images in `output_path`.
```python
from record_msg.parser import ImageParser

image_parser = ImageParser(output_path='../test')
for topic, message, t in record.read_messages():
  if topic == "/apollo/sensor/camera/front_6mm/image":
    image_parser.parse(message)
    # or use timestamp as image file name
    # image_parser.parse(image, t)
```

#### lidar
you can use `PointCloudParser` to parse and save pointclouds in `output_path`.
```python
from record_msg.parser import PointCloudParser

pointcloud_parser = PointCloudParser('../test')
for topic, message, t in record.read_messages():
  if topic == "/apollo/sensor/lidar32/compensator/PointCloud2":
    pointcloud_parser.parse(message)
    # other modes, default is 'ascii'
    # pointcloud_parser.parse(message, mode='binary')
    # pointcloud_parser.parse(message, mode='binary_compressed')
```
