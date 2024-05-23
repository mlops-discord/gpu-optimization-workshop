# GPU Optimization Workshop (May 2024)
Slides, notes, and materials for the workshop

- RSVP link: https://lu.ma/1wu5ppl5
- Host: [@chiphuyen](https://github.com/chiphuyen)'s [Discord community](https://discord.gg/C8duCmvngk)
- [YouTube's recording](https://www.youtube.com/watch?v=v_q2JTIqE20)

## Pre-event note
* The talks are pretty technical, given that this is a workshop on GPU optimization. The speakers try their best to make their topics accessible, but you’ll make more out of the workshop if you familiarize yourself with the basic concepts in advance. (See Reading materials)
* The event will be livestreamed on YouTube, but questions should be asked on Discord, not YouTube.
* Given that we have 2000+ people signing up for the event, we expect there will be a lot of interesting live discussions on Discord.
* Workshop TAs who will be helping us run the workshop:
    * [Roland Tannous](https://www.linkedin.com/in/rolandjosephtannous/)
    * [Chris Alexiuk](https://www.linkedin.com/in/csalexiuk/)
    * [Matúš Jurák](https://www.linkedin.com/in/mat%C3%BA%C5%A1-jur%C3%A1k-8bb680139/)

## Schedule
**[12:00] Crash course on GPU optimization ([Mark Saroufim](https://www.linkedin.com/in/marksaroufim/) @ Meta)**

_Mark is a PyTorch core developer and cofounder of CUDA MODE. He also ran the really fun NeurIPS[ LLM Efficiency challenge](https://neurips.cc/virtual/2023/competition/66594) last year. Previously, he was at Graphcore and Microsoft._

Mark will give an overview of why GPUs, the metrics that matter, and different GPU programming models (thread-based CUDA and block-based Triton). He promises this will be a painless guide to writing CUDA/Triton kernels! This talk will give us the basics to understand the rest of the workshop.

**[12:45] High-performance LLM serving on GPUs ([Sharan Chetlur](https://www.linkedin.com/in/sharan-chetlur-1bb35912/) @ NVIDIA)**

_Sharan is a principal engineer working on TensorRT-LLM at NVIDIA. He’s been working on CUDA since 2012, optimizing the performance of deep learning models from a single GPU to a full data center scale. Previously, he was the Director of Engineering at Cerebras._

Sharan will discuss how to build performant, flexible solutions to optimize LLM serving given the rapid evolution of new models and techniques. The talk will cover optimization techniques such as token concatenation, different strategies for batching, and cache.

**[13:20] Block-based GPU Programming with Triton ([Philippe Tillet](https://www.linkedin.com/in/philippe-tillet-809b5536/) @ OpenAI)**

_Philippe is currently leading the Triton team at OpenAI. Previously, he was at pretty much all major chip makers including NVIDIA, AMD, Intel, and Nervana._

Philippe will explain how Triton works and how its block-based programming model differs from the traditional single instruction, multiple threads (SIMT) programming model that CUDA follows. Triton aims to be higher-level than CUDA while being more expressive (lower-level) than common graph compilers like XLA and Torch-Inductor.

**[14:00] Scaling data processing from CPU to distributed GPUs ([William Malpica](https://www.linkedin.com/in/william-malpica-68577a44/) @ Voltron Data)**

_William is a co-founder of Voltron Data and the creator of BlazingSQL. He helped scale Theseus, a GPU-native query engine, to handle 100TB queries!_

Most people today use GPUs for training and inference. A category of workloads that GPUs excel at but are underutilized for is data processing. In this talk, William will discuss why large-scale data processing should be done on GPUs instead of CPUs and how different tools like cuDF, RAPIDS, and Theseus leverage GPUs for data processing.

## Reading materials 

Please read the schedule below carefully. If there are terms you’re not familiar with, you might want to look them up in advance. Examples:

1. **Memory bound vs. compute bound**: whether the bottleneck is in GPU’s memory or in computation capabilities.
2. **Thread-based vs. block-based**: different programming models for GPU programming. CUDA is thread-based and Triton is block-based.

Tools that will be discussed in the workshop:

1. [Development repository for the Triton language and compiler](https://github.com/triton-lang/triton)
    1. [Introducing Triton: Open-source GPU programming for neural networks](https://openai.com/index/triton/)
    2. [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) 
2. [TensorRT](https://github.com/NVIDIA/TensorRT) and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
3. Check out Mark’s lecture on [profiling CUDA in PyTorch](https://www.youtube.com/watch?v=LuhJEEJQgUM&ab_channel=CUDAMODE).
    3. [Model Inference Optimization Checklist](https://pytorch.org/serve/performance_checklist.html)
    4. [Accelerating Generative AI with PyTorch: Segment Anything, Fast](https://pytorch.org/blog/accelerating-generative-ai/) 
4. [rapidsai/cudf - GPU DataFrame Library](https://github.com/rapidsai/cudf) 
5. [Benchmarking Report: Theseus Engine | Voltron Data](https://voltrondata.com/benchmarks/theseus) 

Recommended resources:
1. [How CUDA Programming Works - Stephen Jones, NVIDIA](https://www.youtube.com/watch?v=QQceTDjA4f4&ab_channel=ChristopherHollinworth) (great lecture)
2. [The Best GPUs for Deep Learning in 2023 — An In-depth Analysis](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) (Tim Dettmers) 
3. [CUDA MODE Discord](https://discord.gg/cudamode). They have a great [lecture series on GPU optimization](https://github.com/cuda-mode/lectures/tree/main).


