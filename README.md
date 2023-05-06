# ChatGLM-Tuning-Zh-And-En

一种平价的 chatgpt 实现方案，基于清华的 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) + LoRA 进行 finetune 。我这里模仿了大佬的项目【见文末】用 alpaca 的英文和中文数据分别进行了微调，并将模型一并上传，感兴趣的可以按照下面步骤重新训练，或者跳过训练直接加载模型进行推理体验。



### 准备

关于 requirements.txt 文件，我和原作者配置有些不同，用户在 requirements.txt 中自行选择，删除没用的即可。

原作者环境配置：

    显卡: 显存 >= 16G (最好24G或者以上)
    python>=3.8
    cuda>=11.6 
    bitsandbytes==0.37.1
    accelerate==0.17.1
    protobuf>=3.19.5,<3.20.1
    transformers==4.27.1
    icetk
    cpm_kernels==1.0.11
    torch>=1.13.1
    tensorboard
    datasets==2.10.1
    git+https://github.com/huggingface/peft.git  # 最新版本 >=0.3.0.dev0

我的配置环境：

    4090 显卡 24G 显存
    python==3.8
    accelerate==0.18.0
    bitsandbytes==0.37.5  # 这个包可能在 win 上装不了，安装这个 git+https://github.com/Keith-Hon/bitsandbytes-windows.git
    protobuf==3.19.6
    transformers==4.27.1
    icetk==0.0.4
    cpm_kernels==1.0.11
    torch==1.13.1+cu116
    datasets==2.11.0
    peft==0.2.0

### 数据预处理


1. 转化 alpaca 数据集为 jsonl 格式

```bash
python cover_alpaca2jsonl.py \
    --data_path data/alpaca_data_zh.json \   # 记得中文要选择中文数据，英文选择英文数据
    --save_path data/alpaca_data_zh.jsonl 
```

2. tokenization

```bash
python tokenize_dataset_rows.py \
    --jsonl_path data/alpaca_data_zh.jsonl \   # 记得中文要选择中文数据，英文选择英文数据
    --save_path data/alpaca_zh \    # 中文和英文数据的处理结果文件路径不一样
    --max_seq_length 200 \ 
    --skip_overlength true
```


### 训练

```bash
python finetune.py \
    --dataset_path data/alpaca_zh \   # 记得中文要选择中文数据，英文选择英文数据
    --lora_rank 8 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --max_steps 52000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --fp16 false\
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir output_zh  # 记得中文要选择中文的模型存储位置，英文选择英文模型存储位置
```

### 推理

```bash
python infer.py \
--data_path \
data/alpaca_data_zh.json \   # 记得中文要选择中文数据，英文选择英文数据
--model_path \
output_zh    # 记得中文要选择中文的模型存储位置，英文选择英文模型存储位置
```

### 微调中文数据结果对比

只选取了三个指令进行结果对比展示。

    （1）微调后的结果:
     Instruction: 保持健康的三个提示。
    Answer: 1. 均衡饮食: 均衡饮食是保持健康的关键。确保你摄入足够的蛋白质、碳水化合物、脂肪、维生素和矿物质。避免过多的糖和饱和脂肪酸,因为它们可能会增加你的体重并增加患慢性疾病的风险。
    
    2. 运动锻炼: 运动锻炼有助于保持身体健康,增强心血管功能,提高骨密度,并帮助减轻压力和焦虑。每周至少进行150分钟的中等强度或75分钟的高强度有氧运动,如快走、骑自行车或游泳。
    
       3. 保持良好的睡眠习惯: 睡眠对于身体健康至关重要。睡眠有助于恢复精力,提高注意力和记忆力,并有助于减少压力和焦虑。确保你每晚至少睡7-8个小时,并
    
    原结果:
     以下是保持健康的三个提示：
    
    1. 保持身体活动。每天做适当的身体运动，如散步、跑步或游泳，能促进心血管健康，增强肌肉力量，并有助于减少体重。
    
       2. 均衡饮食。每天食用新鲜的蔬菜、水果、全谷物和脂肪含量低的蛋白质食物，避免高糖、高脂肪和加工食品，以保持健康的饮食习惯。
    
       3. 睡眠充足。睡眠对人体健康至关重要，成年人每天应保证 7-8 小时的睡眠。良好的睡眠有助于减轻压力，促进身体恢复，并提高注意力和记忆力。
    

    （2）微调后的结果:
     Instruction: 三原色是什么?
    Answer: 三原色是红色、绿色和蓝色,它们被称为RGB颜色模式。在RGB颜色模式中,红色、绿色和蓝色分别代表0,0,0和255,它们分别对应于红色、绿色和蓝色的不同波长。三原色可以组合成其他颜色,例如红色、绿色和蓝色可以组合成黄色。
    
    原结果:
     三原色通常指的是红色、绿色和蓝色（RGB）。它们是通过加色混合原理创建色彩的三种基础颜色。在以发光为基础的显示设备中（如电视、计算机显示器、智能手机和平板电脑显示屏）, 三原色可混合产生大量色彩。其中红色和绿色可以混合生成黄色，红色和蓝色可以混合生成品红色，蓝色和绿色可以混合生成青色。当红色、绿色和蓝色按相等比例混合时，可以产生白色或灰色。
    
    此外，在印刷和绘画中，三原色指的是以颜料为基础的红、黄和蓝颜色（RYB）。这三种颜色用以通过减色混合原理来创建色彩。不过，三原色的具体定义并不唯一，不同的颜色系统可能会采用不同的三原色。
    


    （3）微调后的结果:
     Instruction: 描述原子的结构。
    Answer: 原子是构成物质的基本单位,它由一个质子和两个中子组成。质子位于原子核的中心,中子围绕质子运动。原子核由质子和中子组成,其中质子数等于核子数,中子数等于质子数。
    
    原子的外层电子围绕原子核运动,电子的轨道和能级是原子结构的重要组成部分。原子的外层电子能级可以分为几个阶段,每个能级对应着电子的能级。
    
    原子的内层电子可以按不同的方式排列,例如,电子可以围绕原子核运动,也可以围绕其他电子运动。
    
    原子的结构是复杂的,它
    
    原结果:
     原子是物质的基本单位，它由三种基本粒子组成：质子、中子和电子。质子和中子形成原子核，位于原子中心，核外的电子围绕着原子核运动。
    
    原子结构具有层次性。原子核中，质子带正电，中子不带电（中性）。原子核非常小且致密，占据了原子总质量的绝大部分。电子带负电，通常围绕核运动，形成若干层次，称为壳层或电子层。电子数量与质子数量相等，使原子呈电中性。
    
    电子在每个壳层中都呈规律分布，并且不同壳层所能容纳的电子数也不同。在最里面的壳层一般只能容纳2个电子，其次一层最多可容纳8个电子，再往外的壳层可容纳的电子数逐层递增。
    
    原子核主要受到两种相互作用力的影响：强力和电磁力。强力的作用范围非常小，主要限制在原子核内，具有极强的吸引作用，使核子（质子和中子）紧密结合在一起。电磁力的作用范围较大，主要通过核外的电子与原子核相互作用，发挥作用。
    
    这就是原子的基本结构。原子内部结构复杂多样，不同元素的原子核中质子、中子数量不同，核外电子排布分布也不同，形成了丰富多彩的化学世界。


### 微调英文数据结果对比

只选取了三个指令进行结果对比展示。


    微调后的结果:
     Instruction: Give three tips for staying healthy.
    Answer: 1. Eat a balanced diet and limit your consumption of processed foods.
    2. Get regular exercise, whether it's walking, running, or swimming.
       3. Stay hydrated by drinking plenty of water and drinking certain drinks such as orange juice, tea, and coffee.
    
    原结果:
     1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. 
    2. Exercise regularly to keep your body active and strong. 
       3. Get enough sleep and maintain a consistent sleep schedule.
    
    微调后的结果:
     Instruction: What are the three primary colors?
    Answer: The three primary colors are red, blue, and yellow.
    
    原结果:
     The three primary colors are red, blue, and yellow.
    
    微调后的结果:
     Instruction: Describe the structure of an atom.
    Answer: An atom is made up of a number of different elements, each with its own unique structure. The most common type of atom is the hydrogen atom, which consists of one electron and one electron-positron pair. Other atoms include the carbon atom, which has two electrons, the oxygen atom, which has four electrons, and the nitrogen atom, which has three electrons. Each of these atoms has its own unique structure, which is determined by the number of electrons and the number of protons in its atoms.
    
    原结果:
     An atom is made up of a nucleus, which contains protons and neutrons, surrounded by electrons that travel in orbits around the nucleus. The protons and neutrons have a positive charge, while the electrons have a negative charge, resulting in an overall neutral atom. The number of each particle determines the atomic number and the type of atom.
    
     


### 感谢

1. https://github.com/xyliu-uir/ChatGLM_LoRA_zh
2. https://github.com/mymusise/ChatGLM-Tuning
3. https://github.com/THUDM/ChatGLM-6B