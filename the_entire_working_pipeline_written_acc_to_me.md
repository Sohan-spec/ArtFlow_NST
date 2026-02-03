pipeline:-



-loads image in OpenCV format->BGR, changes to RGB, shape is \[h,w,c], range 0-255



-compute new height and width (resizing essentially) and divide by 255, to keep in b/w 0.0 to 1.0



-vgg 19 contains 5 blocks of conv+relu+maxpool

-1st and 2nd- 2 \* conv + relu +1 maxpool

3rd 4th and 5th- 4 \* conv+relu + 1 maxpool



-change that image to tensor->\[c,h,w], multiply 255 (for tensor calcs), normalize each R,G and B pixel (in this case just x-mean) (the range essentially becomes like -123 to 131 approx.)



-add your batch dimension (1)



-this tensor goes into VGG with its default weights, slides a 3x3 filter over the image covering all hxw pixels and computes dot products (also this is done for all 3 layers at once R G and B), filter is 3 x 3 x3



-inside the conv layers we give internal batches (aka filters=no. of output channels) (64,128,256 etc) and so the filter thing is done 64 times per conv layer in the first block



-in the first layer for all 3 channels it does it 64 times, and then it like stacks each and every one (so it essentially becomes just one), in the next layer it takes all 64 times input as a single input and then does it another 64 times and the following goes on to all the layers



-then ofc after the conv layer is done, bias is added, relu is used



-maxpool is applied after each block and essentially just halves the height and width, while keeping the strongest features



-feature maps are taken after each blocks 1st conv + relu layer (in my design), it's basically the image (RGB) is stacked with 64 diff filters on top, each filter has diff weights so that it can identify diff features from that image, so each filter produces hxw matrix but as initially we had 3 channels now we'll have 64 channels



-now the style image is basically undergoing the gram matrix treatment, where in we have \[c,h,w] and then it converts it to \[c,h\*w], then we do a transpose of that matrix and multiply the \[c,h\*w] with the \[c,h\*w]^t matrix which gives us the correlation b/w how often each feature map occurs together



-then we pass all 3 images onto the vgg and do forward propagation, only optimizing goes through every iteration for back prop, content and style just initially



-in the end we calculate content loss, using MSE, it compares each feature map of our optimizing image to our content image layer by layer, by doing a mean square loss



-then for style loss, we check for each feature map in our optimizing image gram matrix and our style image gram matrix and find out the mean square loss again



-we then calculate total variation loss, where we check for adjacent pixels and we do their minus (by row and by column) and add them(the row and column keeping modulus ofc)



-then gradients are calculated delta loss/delta x which tells each pixel it's worth basically (how much it decreasing/increasing affects others)



- then by matching the feature maps of the content image and the optimizing image we get the same content, by matching the gram matrices of the style image and our optimizing image we get our style and voila there you have it

