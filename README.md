# record_msg
record message parse helper function

## Parser record
`record_msg` provides 3 interfaces to help parsing cyber record file.

#### csv format
you can use `to_csv` to format objects so that they can be easily saved in csv format.
```python
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
image_parser = ImageParser(output_path='../test')
for topic, message, t in record.read_messages():
  if topic == "/apollo/sensor/camera/front_6mm/image":
    image_parser.parse(image)
    # or use timestamp as image file name
    # image_parser.parse(image, t)
```

#### lidar
todo
