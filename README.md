# EMDL

Embedded and mobile deep learning research notes

## Docs

### Paper

1. [DeepMon: Mobile GPU-based Deep Learning Framework for Continuous Vision Applications](https://www.sigmobile.org/mobisys/2017/accepted.php) [MobiSys '17]

1. [DeepEye: Resource Efficient Local Execution of Multiple Deep Vision Models using Wearable Commodity Hardware](http://fahim-kawsar.net/papers/Mathur.MobiSys2017-Camera.pdf) [MobiSys '17]

1. [MobiRNN: Efficient Recurrent Neural Network Execution on Mobile GPU](https://arxiv.org/abs/1706.00878) [EMDL '17]

1. [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) [arXiv '17, Google ]

1. [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149) [ICLR'16 (oral)]

1. [Quantized Convolutional Neural Networks for Mobile Devices](https://arxiv.org/abs/1512.06473) [CVPR '16]

1. [DeepSense: A GPU-based deep convolutional neural network framework on commodity mobile devices](http://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=4278&context=sis_research) [WearSys '16]

1. [DeepX: A Software Accelerator for Low-Power Deep Learning Inference on Mobile Devices](http://niclane.org/pubs/deepx_ipsn.pdf) [IPSN '16]

1. [EIE: Efficient Inference Engine on Compressed Deep Neural Network](https://arxiv.org/abs/1602.01528) [ISCA '16]

1. [MCDNN: An Approximation-Based Execution Framework for Deep Stream Processing Under Resource Constraints](http://haneul.github.io/papers/mcdnn.pdf) [MobiSys '16]

1. [Sparsification and Separation of Deep Learning Layers for Constrained Resource Inference on Wearables](http://niclane.org/pubs/sparsesep_sensys.pdf) [SenSys ’16]

1. [An Early Resource Characterization of Deep Learning on Wearables, Smartphones and Internet-of-Things Devices](http://niclane.org/pubs/iotapp15_early.pdf) [IoT-App ’15]

1. [CNNdroid: GPU-Accelerated Execution of Trained Deep Convolutional Neural Networks on Android](https://arxiv.org/abs/1511.07376) [MM '16]

### Guide

1. [Squeezing Deep Learning Into Mobile Phones](https://www.slideshare.net/anirudhkoul/squeezing-deep-learning-into-mobile-phones)

1. [Efficient Convolutional Neural Network Inference on Mobile GPUs](https://www.slideshare.net/embeddedvision/efficient-convolutional-neural-network-inference-on-mobile-gpus-a-presentation-from-imagination-technologies)

1. [Deep learning **systems**](http://dlsys.cs.washington.edu/schedule), UW course schedule(focused on systems design, not learning)

## Code

### General

1. [ARM-software/ComputeLibrary: The ARM Computer Vision and Machine Learning library is a set of functions optimised for both ARM CPUs and GPUs using SIMD technologies](https://github.com/ARM-software/ComputeLibrary), [Intro](https://developer.arm.com/technologies/compute-library)

1. [Apple CoreML](https://developer.apple.com/documentation/coreml)

### OpenCL, Vulkan, RenderScript

1. [SaschaWillems/Vulkan: Examples and demos for the new Vulkan API](https://github.com/SaschaWillems/Vulkan)

1. [ARM-software/vulkan-sdk: ARM Vulkan SDK](https://github.com/ARM-software/vulkan-sdk)

1. [alexhultman/libvc: Vulkan Compute for C++ (experimentation project)](https://github.com/alexhultman/libvc)

1. [Deep Learning in a Single File for Smart Devices — mxnet](https://github.com/dmlc/mxnet/tree/master/amalgamation)

1. [TensorFlow Android Camera Demo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android)

1. [bwasti/AICamera: Demonstration of using Caffe2 inside an Android application.](https://github.com/bwasti/AICamera)

1. [mtmd/Mobile_ConvNet: RenderScript based implementation of Convolutional Neural Networks for Android phones](https://github.com/mtmd/Mobile_ConvNet)

1. [harvardnlp/nmt-android: Neural Machine Translation on Android](https://github.com/harvardnlp/nmt-android)

1. [hollance/TensorFlow-iOS-Example: Source code for my blog post "Getting started with TensorFlow on iOS"](https://github.com/hollance/TensorFlow-iOS-Example)

### Tutorial

1. [ARM® Mali™ GPU OpenCL Developer Guide](http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.100614_0303_00_en/ada1432742770595.html), [pdf](http://infocenter.arm.com/help/topic/com.arm.doc.100614_0303_00_en/arm_mali_gpu_opencl_developer_guide_100614_0303_00_en.pdf)

1. [Optimal Compute on ARM MaliTM GPUs](http://www.cs.bris.ac.uk/home/simonm/montblanc/OpenCL_on_Mali.pdf)

1. [GPU Compute for Mobile Devices](http://www.iwocl.org/wp-content/uploads/iwocl-2014-workshop-Tim-Hartley.pdf)

1. [Compute for Mobile Devices Performance focused](http://kesen.realtimerendering.com/Compute_for_Mobile_Devices5.pdf)

1. [Hands On OpenCL](https://handsonopencl.github.io/)

1. [Adreno OpenCL Programming Guide](https://developer.qualcomm.com/download/adrenosdk/adreno-opencl-programming-guide.pdf)

1. [Better OpenCL Performance on Qualcomm Adreno GPU](https://developer.qualcomm.com/blog/better-opencl-performance-qualcomm-adreno-gpu-memory-optimization)

### Others

1. [mil-tokyo/webdnn: Fastest DNN Execution Framework on Web Browser](https://github.com/mil-tokyo/webdnn)

## Hardware

### GPU

1. [Bifrost GPU architecture and ARM Mali-G71 GPU](https://www.hotchips.org/wp-content/uploads/hc_archives/hc28/HC28.22-Monday-Epub/HC28.22.10-GPU-HPC-Epub/HC28.22.110-Bifrost-JemDavies-ARM-v04-9.pdf)

1. [Midgard GPU Architecture](http://malideveloper.arm.com/downloads/ARM_Game_Developer_Days/PDFs/2-Mali-GPU-architecture-overview-and-tile-local-storage.pdf), [ARM Mali-T880 GPU](https://www.hotchips.org/wp-content/uploads/hc_archives/hc27/HC27.25-Tuesday-Epub/HC27.25.50-GPU-Epub/HC27.25.531-Mali-T880-Bratt-ARM-2015_08_23.pdf)


### Driver

1. [Adreno] [csarron/qcom_vendor_binaries: Common Proprietary Qualcomm Binaries](https://github.com/csarron/qcom_vendor_binaries)
1. [Mali] [Fevax/vendor_samsung_hero2ltexx: Blobs from s7 Edge G935F](https://github.com/Fevax/vendor_samsung_hero2ltexx)
