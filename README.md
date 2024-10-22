# image-restoration

Project of the course *Applied Deep Learning*

- Topic: Restoring grainy black and white images, which actually consists of two potentially separate tasks: denoising and colorization.
- Project type: a combination of "Bring your own data" and "Bring your own method".
- Inspiration: In 2018, Peter Jackson, the director of The Lord of the Ring movies, released a new movie called "They Shall Not Grow Old". He took old, grainy, black and white footages of the first world war, denoised them, colored them, added sound to them, and edited them into a whole movie that tells a story. The result was stunning. We always think of older historical events as black and white, as grainy. We have only seen low quality footage. But those people did not live in a black and white world. Their world had as much color as we have now. This alone makes a huge difference in immersing the audience into the story.
- Approach: I intend to synthesize my own data. The [DIV2K dataset](https://paperswithcode.com/dataset/div2k) can be used as the dataset. It has 1000 images. Then, I would have to synthesize black and white grainy images from them. Then, I would like to try using convolutional autoencoders to denoise the images. Then, I would like to try using a UNet for colorization. It would also be interesting to try an end-to-end approach to combine both tasks into one.
- Papers:
  
   https://paperswithcode.com/paper/image-restoration-using-convolutional-auto
  
   https://paperswithcode.com/paper/colorful-image-colorization
- Time estimation:
1. dataset synthesis and augmentation: 30 hrs
2. denoising model: 15 hrs
3. colorization model: 15 hrs
4. fine-tuning everything: 5 hrs
5. final report and presentation: 10 hrs
