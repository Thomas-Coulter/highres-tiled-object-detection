#Object Detection Testing using custom CNN

Purpose is to test "tiling" subimages and feed them into the detector rather than letting the images be compressed to fit the 640x640 thereby losing resolution of the subjects

#*Test Image and Outcomes*

**INPUT IMAGE**
![1](https://github.com/user-attachments/assets/f98b5540-1cea-4f52-b3a9-11bb734c0f95)

**DETECTED IMAGE, 640X640 COMPRESSION HAPPENS PRIOR TO INFERENCE**
![1a](https://github.com/user-attachments/assets/f3d2915c-761a-4127-9b17-47c54712274d)

**DETECTED IMAGE, DIVIDED INTO 640X640 TILES AND REASSEMBLED WITH LOCATIONS MAPPED TO ORIGINAL IMAGE**
![1b](https://github.com/user-attachments/assets/916331fd-080f-45bb-b3a6-c0e514591c22)
