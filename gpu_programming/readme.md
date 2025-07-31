[IMPROTANT-TODO] Learn GPU programming and fundamental machine learning through it to prepare for the hackathon. performance & kernel engineering to optimize machine learning/deep learning models on TPUs, GPUs & Trainium. 

These attributes are nice to have:

    PhD in Computer Science and Engineering with a specialization in Computer Architecture, Parallel Computing, Compilers, or other Systems

    Participation in competitive programming competitions

    Experience building compilers

    Experience working with hardware developers
Here you find a collection of CUDA related material (books, papers, blog-post, youtube videos, tweets, implementations etc.). We also collect information to higher level tools for performance optimization and kernel development like Triton and torch.compile() ... whatever makes the GPUs go brrrr.


Material (code, slides) for the individual lectures can be found in the lectures repository.


We are hiring! If you are interested in efficient architecture or making training and inference on thousands of GPUs much faster, please feel free to dm me or @WeizhuChen

GPU kernel programming

Let's write some fast numeric code. Concerns of compute, memory, cache, and data movement come together in the pursuit of performance. (Links below don't include computer graphics, that's a huge topic.)

Getting started

    PMPP Book

CCCL (Thrust, CUB, libcudacxx)

CUDA Matmul blog post

Extending ML frameworks

    PyTorch: Custom ops with CUDA

JAX: Custom ops with CUDA

Triton

Pallas (JAX)

ggml-cuda

tfjs-backend-webgl

tfjs-backend-webgpu

Modern toolchains

    SPIRV-Cross

Tour of WGSL

Apache TVM

ML model formats

Scientific computing

    JuliaGPU

Futhark

JaxNeRF
 



1st Contact with CUDA

    An Easy Introduction to CUDA C and C++
    An Even Easier Introduction to CUDA
    CUDA Toolkit Documentation
    Basic terminology: Thread block, Warp, Streaming Multiprocessor: Wiki: Thread Block, A tour of CUDA
    GPU Performance Background User's Guide
    OLCF NVIDIA CUDA Training Series, talk recordings can be found under the presentation footer for each lecture; exercises
    GTC 2022 - CUDA: New Features and Beyond - Stephen Jones
    Intro video: Writing Code That Runs FAST on a GPU
    12 hrs CUDA tutorial: Introduction of CUDA and writing kernels in CUDA

2nd Contact

    CUDA Refresher

Hazy Research

The MLSys-oriented research group at Stanford led by Chris Re, with alumni Tri Dao, Dan Fu, and many others. A goldmine.

    Building Blocks for AI Systems: Their collection of resources similar to this one, many great links
    Data-Centric AI: An older such collection
    Blog
    ThunderKittens: (May 2024) A DSL within CUDA, this blog post has good background on getting good H100 performance
    Systems for Foundation Models, and Foundation Models for Systems: Chris Re's keynote from NeurIPS Dec 2023

Papers, Case Studies

    A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA Hopper Architecture using the CUTLASS Library
    How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog
    Anatomy of high-performance matrix multiplication

Books

    Programming Massively Parallel Processors: A Hands-on Approach
    Cuda by Example: An Introduction to General-Purpose Gpu Programming; code
    The CUDA Handbook
    The Book of Shaders guide through the abstract and complex universe of Fragment Shader (not cuda but GPU related)
    Art of HPC 4 books on HPC more generally, does not specifically cover GPUs but lessons broadly apply

Cuda Courses

    HetSys: Programming Heterogeneous Computing Systems with GPUs and other Accelerators
    Heterogeneous Parallel Programming Class (YouTube playlist) Prof. Wen-mei Hwu, University of Illinois
    Official YouTube channel for "Programming Massively Parallel Processors: A Hands-on Approach", course playlist: Applied Parallel Programming
    Programming Parallel Computers; covers both CUDA and CPU-parallelism. Use Open Course Version and you can even submit your own solutions to the exercises for testing and benchmarking.

CUDA Grandmasters
Tri Dao

    x: @tri_dao, gh: tridao
    Dao-AILab/flash-attention, paper
    state-spaces/mamba, paper: Mamba: Linear-Time Sequence Modeling with Selective State Spaces, minimal impl: mamba-minimal

Tim Dettmers

    x: @Tim_Dettmers, gh: TimDettmers
    TimDettmers/bitsandbytes, docs: docs
    QLoRA: Efficient Finetuning of Quantized LLMs

Sasha Rush

    x: @srush_nlp, gh: srush
    Sasha Rush's GPU Puzzles, dshah3's CUDA C++ version & walkthrough video
    Mamba: The Hard Way, code: srush/annotated-mamba

Practice

    Adnan Aziz and Anupam Bhatnagar GPU Puzzlers

PyTorch Performance Optimization

    Accelerating Generative AI with PyTorch: Segment Anything, Fast
    Accelerating Generative AI with PyTorch II: GPT, Fast
    Speed, Python: Pick Two. How CUDA Graphs Enable Fast Python Code for Deep Learning
    Performance Debugging of Production PyTorch Models at Meta

PyTorch Internals & Debugging

    TorchDynamo Deep Dive
    PyTorch Compiler Troubleshooting
    PyTorch internals
    Pytorch 2 internals
    Understanding GPU memory: 1: Visualizing All Allocations over Time, 2: Finding and Removing Reference Cycles
    Debugging memory using snapshots: Debugging PyTorch memory use with snapshots
    CUDA caching allocaator: https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html
    Trace Analyzer: PyTorch Trace Analysis for the Masses
    Holistic Trace Analysis (HTA), gh: facebookresearch/HolisticTraceAnalysis

Code / Libs

    NVIDIA/cutlass

Essentials

    Triton compiler tutorials
    CUDA C++ Programming Guide
    PyTorch: Custom C++ and CUDA Extensions, Code: pytorch/extension-cpp
    PyTorch C++ API
    pybind11 documentation
    NVIDIA Tensor Core Programming
    GPU Programming: When, Why and How?
    How GPU Computing Works | GTC 2021 (more basic than the 2022 version)
    How CUDA Programming Works | GTC 2022
    CUDA Kernel optimization Part 1 Part 2
    PTX and ISA Programming Guide (V8.3)
    Compiler Explorer: Inspect PTX: div 256 -> shr 8 example

Profiling

    Nsight Compute Profiling Guide
    mcarilli/nsight.sh - Favorite nsight systems profiling commands for PyTorch scripts
    Profiling GPU Applications with Nsight Systems

Python GPU Computing

    PyTorch
    Trtion, github: openai/triton
    numba @cuda.jit
    Apache TVM
    JAX Pallas
    CuPy NumPy compatible GPU Computing
    NVidia Fuser
    Codon @gpu.kernel, github: exaloop/codon
    Mojo (part of commercial MAX Plattform by Modular)
    NVIDIA Python Bindings: CUDA Python (calling NVRTC to compile kernels, malloc, copy, launching kernels, ..), cuDNN FrontEnd(FE) API, CUTLASS Python Interface

Advanced Topics, Research, Compilers

    TACO: The Tensor Algebra Compiler, gh: tensor-compiler/taco
    Mosaic compiler C++ DSL for sparse and dense tensors algebra (built on top of TACO), paper, presentation

News

    SemiAnalysis

Technical Blog Posts

    Cooperative Groups: Flexible CUDA Thread Programming (Oct 04, 2017)
    A friendly introduction to machine learning compilers and optimizers (Sep 7, 2021)

Hardware Architecture

    NVIDIA H100 Whitepaper
    NVIDIA GH200 Whitepaper
    AMD CDNA 3 Whitepaper
    AMD MI300X Data Sheet
    Video: Can SRAM Keep Shrinking? (by Asianometry)

GPU-MODE Community Projects
ring-attention

    see our ring-attention repo

pscan

    GPU Gems: Parallel Prefix Sum (Scan) with CUDA, PDF version (2007), impl: stack overflow, nicer impl: mattdean1/cuda
    Accelerating Reduction and Scan Using Tensor Core Units
    Thrust: Prefix Sums, Reference: scan variants
    CUB, part of cccl: NVIDIA/cccl/tree/main/cub
    SAM Algorithm: Higher-Order and Tuple-Based Massively-Parallel Prefix Sums (licensed for non commercial use only)
    CUB Algorithm: Single-pass Parallel Prefix Scan with Decoupled Look-back
    Group Experiments: johnryan465/pscan, andreaskoepf/pscan_kernel

Triton Kernels / Examples

    unsloth that implements custom kernels in Triton for faster QLoRA training
    Custom implementation of relative position attention (link)
    Tri Dao's Triton implementation of Flash Attention: flash_attn_triton.py
    YouTube playlist: Triton Conference 2023
    LightLLM with different triton kernels for different LLMs


Other Libraries and Resources

    A Practioner's Guide to Triton is a great gentle intro to Triton (here's the accompanying notebook).
        Note from Umer: It feels weird to include my own work, but many people said the lecture was very helpful!

    The Triton Tutorials are a great intermediate resources.

    Flash Attention has a number of useful Triton kernels.

    Unsloth contains many ready-to-use Triton kernels especially for finetuning applications

    flash-linear-attention has a massive number of Linear attention or subquadratic attention replacement architectures, written using several different approaches of parallelization in Triton.

    xformers contains many Triton kernels throughout, including some attention kernels such as Flash-Decoding.

    Applied AI contains kernels such as a Triton MoE, fused softmax for training and inference

    AO contains kernels for GaLoRe, HQQ and DoRA

    torch.compile can codegenerate Triton kernels from PyTorch code

    Liger Kernel is a collection of Triton kernels designed specifically for LLM training

Competition

    PMPP practice problems: Starting on Sunday Feb 21, 2025.
    AMD $100K kernel competition
    BioML kernels


TPU Kernel Engineer x Performance Engineer x GPU Kernels Engineer 

About the Role

As a TPU Kernel Engineer, you'll be responsible for identifying and addressing performance issues across many different ML systems, including research, training, and inference. A significant portion of this work will involve designing and optimizing kernels for the TPU. You will also provide feedback to researchers about how model changes impact performance. Strong candidates will have a track record of solving large-scale systems problems and low-level optimization.

Running machine learning (ML) algorithms at our scale often requires solving novel systems problems. As a Performance Engineer, you'll be responsible for identifying these problems, and then developing systems that optimize the throughput and robustness of our largest distributed systems. Strong candidates here will have a track record of solving large-scale systems problems and will be excited to grow to become an expert in ML also.



As a Kernels Engineer for Kernel Libraries, you will write high-performance kernels for the training and inference workloads. You will work with other engineers across the platform team to accelerate our biggest training runs. You will also work backward from the capabilities of the GPUs to make model architectures amenable to efficient training and inference. If you are excited about maximizing HBM, optimizing for instruction issue rate, shuffling within a warp, managing the precious register space, and keeping the tensor cores at high utilization, this is the perfect opportunity!

We are looking for a kernel-focused engineer to lead efforts in writing, porting, and optimizing GPU kernels used in inference workloads. This role requires deep familiarity with CUDA or equivalent kernel programming environments, and a strong intuition for performance tuning across modern GPU architectures.

As a Software Engineer, you will help build AI systems that can perform previously impossible tasks or achieve outstanding levels of performance. This requires good engineering (for example designing, implementing, and optimizing state-of-the-art AI models), writing bug-free machine learning code (surprisingly difficult!), and building the science behind the algorithms employed. In all the projects this role pursues, the ultimate goal is to push the field forward.

The Research Acceleration team builds high-quality research tools and frameworks to increase research productivity across OpenAI, with the goal of accelerating progress towards AGI. For example, we develop Triton, a language and compiler for writing custom GPU kernels. The aim of Triton is to provide an open-source environment to write fast code at higher productivity than CUDA. 

We frequently collaborate with other teams to speed up the development of new state-of-the-art capabilities. For example, we recently collaborated with our Codegen research team on the Codex model, which can generate code in Python and many other languages.

Do you love research tools, compilers, and collaborating on cutting-edge AI models? If so, this role is for you! We are looking for people who are self-directed and enjoy determining the most meaningful problem to solve in order to accelerate our research.

As a software engineer on the Scaling team, you’ll help build and optimize the low-level stack that orchestrates computation and data movement across OpenAI’s supercomputing clusters. Your work will involve designing high-performance runtimes, building custom kernels, contributing to compiler infrastructure, and developing scalable simulation systems to validate and optimize distributed training workloads.

You will work at the intersection of systems programming, ML infrastructure, and high-performance computing, helping to create both ergonomic developer APIs and highly efficient runtime systems. This means balancing ease of use and introspection with the need for stability and performance on our evolving hardware fleet.

This role is based in San Francisco, CA, with a hybrid work model (3 days/week in-office). Relocation assistance is available.



You may be a good fit if you:

    Have significant experience optimizing ML systems for TPUs, GPUs, or other accelerators
    Have significant software engineering or machine learning experience, particularly at supercomputing scale
    Are results-oriented, with a bias towards flexibility and impact
    Pick up slack, even if it goes outside your job description
    Enjoy pair programming (we love to pair!)
    Want to learn more about machine learning research
    Care about the societal impacts of your work
    Develop high-performance GPU/CPU kernels and make trade-offs that maximize end-to-end hardware utilization
    Utilize knowledge of hardware features and performance characteristics to make aggressive optimizations
    Work with our other platform teams to deploy your kernels, manage our training uptime, and find opportunities for optimization
    Develop low-precision algorithms that deliver high performance with little loss of ML accuracy
    Work with ML engineers to develop model architectures that are amenable to efficient training and inference
    Work with hardware vendors and advise on HW/SW co-design
    Design, implement, and optimize CUDA kernels for inference-critical operations (e.g., fused matmuls, custom activation functions, memory layout transforms).
    Analyze performance bottlenecks and optimize kernel execution for throughput and latency.
    Contribute to and extend internal GPU libraries and runtime tools.
    Work closely with hardware-specific profiling tools (e.g., Nsight, nvprof) to guide improvements.
    Collaborate with researchers to port or re-architect new model operations for production use.
    Contributions to an AI framework such as PyTorch or Tensorflow, or compilers such as GCC, LLVM, or MLIR
    Engineer and optimize compute and data kernels, ensuring correctness, high performance, and portability across simulation and production environments.
    Familiarity with GPUs and performance tooling (e.g., CUDA, Nsight, perf, flamegraphs) is a plus e

Strong candidates may also have experience with: 

    High performance, large-scale ML systems
    Designing and implementing kernels for TPUs or other ML accelerators
    Understanding accelerators at a deep level, e.g. a background in computer architecture
    ML framework internals
    Language modeling with transformers
    GPU/Accelerator programming
    OS internals

    Are a strong coder with excellent skills in C/C++ and Python
    Have a deep understanding of GPU, CPU, or other AI accelerator architectures
    Have experience writing and optimizing compute kernels with CUDA or similar languages
    Are familiar with LLM architectures and training infrastructure
    Have experience driving ML accuracy with low-precision formats
    Get a great deal of satisfaction with every percentage point in performance improvement
    Have deep expertise in CUDA, and have written high-performance GPU code used in production systems.
    Understand GPU memory hierarchies, warp scheduling, and kernel-level tuning.
    Have experience debugging low-level perf issues with Nsight, cupti, or ROCm tools.
    Are excited to build optimized compute primitives that scale across fleets of accelerators.
    Enjoy working closely with model researchers to bring new operators to life.

    Contributions to GPU kernel libraries or frameworks (Triton, cuBLAS, cutlass).
    Experience with mixed precision or tensor core optimization.
    Familiarity with MIG, multi-instance GPU configurations, or NUMA-aware execution.


Representative projects:

    Implement low-latency, high-throughput sampling for large language models
    Adapt existing models for low-precision inference
    Build quantitative models of system performance
    Design and implement custom collective communication algorithms
    Debug kernel performance at the assembly level
    Implement GPU kernels to adapt our models to low-precision inference
    Write a custom load-balancing algorithm to optimize serving efficiency
    Build quantitative models of system performance
    Design and implement a fault-tolerant distributed system running with a complex network topology
    Debug kernel-level network latency spikes in a containerized environment

Please provide a link to, or describe in 1 paragraph, the most impressive low-level or performance thing you've done.


# Elective



    If you’re excited about optimizing code that runs equally well on a single or thousands of GPUs and if you have the ability to submit a single substantial PR to a major OSS library, we want you on the PyTorch team - especially if you’re early in your career.

The crazy low-level ML Systems stuff you’ll work on

If you’re impatient and hate waiting weeks for training runs, if you believe good ideas should help others, and if you want to understand how computers really work - ML Systems is the perfect career start and PyTorch is one of the most succesful ML Systems projects of all time.

If you’re early career and smart you should not be vibe coding, you should be working on harder problems. Here is a random list of projects we’ve been up to in PyTorch.

    Rewriting core collectives to introduce fault tolerance with RDMA and GPUDirect, allowing training to continue even when nodes fail
    Building a custom Python bytecode interpreter so you can capture PyTorch graphs without forcing users to rewrite their Python code
    Rewriting PyTorch Distributed from scratch so you can pdb across a training job
    Rewriting all of our C++ code so it’s ABI compatible for another 20 years
    Fixing performance problems by changing a single register value from 1 to 0

These aren’t theoretical problems – for example, when we built FSDP, it triggered a wave of new distributed training papers so your work determines the shape of next gen research.
What we’re actually looking for

We don’t care about which university you went to. Instead, show us what you’ve built, even if it’s small. If you’ve never contributed to open source before, make a small PR to PyThe open-source advantage

You work with users directly. There are no manufactured problems, you need to work on the actual bottlenecks preventing AI progress. Open source means you’re building in public, getting feedback in public, and earning a public reputation for your work. Your GitHub commits become your resume and we hope to be the last leetcode interview you ever need to go through.Torch and then apply!

Why you should apply

This job isn’t for everyone — but if the points below sound like you, you’ll probably thrive here:

    You enjoy autonomy and defining your own direction. We’re a flat organization where engineers choose their own projects. If you find freedom motivating and like carving your own path, you’ll feel right at home.

    You like being part of the wider community. We work in the open — on GitHub, Discord, and X — and you’ll get to collaborate with users, contributors, and even critics. If you like building in public, this is the place for you.

    You think in trade-offs, not tunnel vision. We care about usability, performance, compatibility, and maintainability. If you’re comfortable balancing competing priorities instead of chasing a single metric, you’ll fit right in.

    You care about long-term impact and ownership. Our work supports thousands of downstream projects. If you take pride in maintaining and improving what you build over time, not just shipping and forgetting, you’ll do well here.


ML Systems Onboarding Reading List

This is a reading list of papers/videos/repos I've personally found useful as I was ramping up on ML Systems and that I wish more people would just sit and study carefully during their work hours. If you're looking for more recommendations, go through the citations of the below papers and enjoy!

Conferences where MLSys papers get published
Attention Mechanism

    Attention is all you need: Start here, Still one of the best intros
    Online normalizer calculation for softmax: A must read before reading the flash attention. Will help you get the main "trick"
    Self Attention does not need O(n^2) memory:
    Flash Attention 2: The diagrams here do a better job of explaining flash attention 1 as well
    Llama 2 paper: Skim it for the model details
    gpt-fast: A great repo to come back to for minimal yet performant code
    Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation: There's tons of papers on long context lengths but I found this to be among the clearest
    Google the different kinds of attention: cosine, dot product, cross, local, sparse, convolutional

Performance Optimizations

    Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems: Wonderful survey, start here
    Efficiently Scaling transformer inference: Introduced many ideas most notably KV caches
    Making Deep Learning go Brrr from First Principles: One of the best intros to fusions and overhead
    Fast Inference from Transformers via Speculative Decoding: This is the paper that helped me grok the difference in performance characteristics between prefill and autoregressive decoding
    Group Query Attention: KV caches can be chunky this is how you fix it
    Orca: A Distributed Serving System for Transformer-Based Generative Models: introduced continuous batching (great pre-read for the PagedAttention paper).
    Efficient Memory Management for Large Language Model Serving with PagedAttention: the most crucial optimization for high throughput batch inference
    Colfax Research Blog: Excellent blog if you're interested in learning more about CUTLASS and modern GPU programming
    Sarathi LLM: Introduces chunked prefill to make workloads more balanced between prefill and decode
    Epilogue Visitor Tree: Fuse custom epilogues by adding more epilogues to the same class (visitor design pattern) and represent the whole epilogue as a tree

Quantization

    A White Paper on Neural Network Quantization: Start here this is will give you the foundation to quickly skim all the other papers
    LLM.int8: All of Dettmers papers are great but this is a natural intro
    FP8 formats for deep learning: For a first hand look of how new number formats come about
    Smoothquant: Balancing rounding errors between weights and activations
    Mixed precision training: The OG paper describing mixed precision training strategies for half

Long context length

    RoFormer: Enhanced Transformer with Rotary Position Embedding: The paper that introduced rotary positional embeddings
    YaRN: Efficient Context Window Extension of Large Language Models: Extend base model context lengths with finetuning
    Ring Attention with Blockwise Transformers for Near-Infinite Context: Scale to infinite context lengths as long as you can stack more GPUs

Sparsity

    Venom: Vectorized N:M Format for sparse tensor cores when hardware only supports 2:4
    Megablocks: Efficient Sparse training with mixture of experts
    ReLu Strikes Back: Really enjoyed this paper as an example of doing model surgery for more efficient inference

Distributed

    Singularity: Shows how to make jobs preemptible, migratable and elastic
    Local SGD: So hot right now
    OpenDiloco: Asynchronous training for decentralized training
    torchtitan: Minimal repository showing how to implement 4D parallelism in pure PyTorch
    pipedream: The pipeline parallel paper
    jit checkpointing: a very clever alternative to periodic checkpointing
    Reducing Activation Recomputation in Large Transformer models: THe paper thatt introduced selective activation checkpointing and goes over activation recomputation strategies
    Breaking the computation and communication abstraction barrier: God tier paper that goes over research at the intersection of distributed computing and compilers to maximize comms overlap
    ZeRO: Memory Optimizations Toward Training Trillion Parameter Models: The ZeRO algorithm behind FSDP and DeepSpeed intelligently reducing memory usage for data parallelism.
    Megatron-LM: For an introduction to Tensor Parallelism


Conferences

    NeurIPS: https://neurips.cc/
    OSDI: https://www.usenix.org/conference/osdi
    MLSys: https://mlsys.org/
    ICLR: https://iclr.cc/
    ICML: https://icml.cc/
    ASPLOS: https://www.asplos-conference.org/
    ISCA: https://iscaconf.org/isca2024/
    HPCA: https://hpca-conf.org/2025/
    MICRO: https://microarch.org/micro57/
    SOCP: https://www.socp.org/
    OSDI: https://www.usenix.org/conference/osdi25
    SOSP: https://www.sosp.org/
    NSDI: https://www.usenix.org/conference/nsdi25

Labs

    https://catalyst.cs.cmu.edu/

Researchers (WIP)

Very incomplete list of researchers and their google scholar profiles.

    Bill Dally: https://scholar.google.com/citations?user=YZHj-Y4AAAAJ&hl=en on efficiency and compression
    Jeff Dean: https://scholar.google.com/citations?user=NMS69lQAAAAJ&hl=en on everything
    Matei Zaharia: https://scholar.google.com/citations?user=I1EvjZsAAAAJ&hl=en on large scale data processing
    Xupeng Mia: https://scholar.google.com/citations?user=aCAgdYkAAAAJ&hl=zh-CN on LLM inference
    Minjia Zhang: https://scholar.google.com/citations?user=98vX7S8AAAAJ&hl=en on LLM inference, Moe and open models
    Dan Fu: https://scholar.google.com/citations?user=Ov-sMBIAAAAJ&hl=en on flash attention and state space models
    Lianmin Zheng: https://scholar.google.com/citations?user=_7Q8uIYAAAAJ&hl=en on LLM inference and evals
    Hui Guan: https://scholar.google.com/citations?user=rfPAfBkAAAAJ&hl=en on quantization, datareuse
    Tian Li: https://scholar.google.com/citations?user=8JWoJrAAAAAJ&hl=en on federated learning
    Gauri Joshi: https://scholar.google.com/citations?user=yqIoH34AAAAJ&hl=en on federated learning
    Tianqi Chen: https://scholar.google.com/citations?user=7nlvOMQAAAAJ&hl=en on compilers and a lot more stuff
    Philip Gibbons: https://scholar.google.com/citations?user=F9kqUXkAAAAJ&hl=en on federated learning
    Christopher De Sa: https://scholar.google.com/citations?user=v7EjGHkAAAAJ&hl=en on data augmentation
    Gennady Pekhimenko: https://scholar.google.com/citations?user=ZgqVLuMAAAAJ&hl=en on mlperf and DRAM
    Onur Mutlu: https://scholar.google.com/citations?user=7XyGUGkAAAAJ&hl=en on computer architecture
    Michael Carbin: https://scholar.google.com/citations?user=mtejbKYAAAAJ&hl=en on pruning

