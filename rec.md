# rdt
1. diffsuion expressiveness, transformer scalability
2. To characterize the nonlinear dynamics, high-frequency changes and unstable numerical range inherent in robotic data,
3. Physically Interpretable Unified Action Space,
4. Finally, ablation studies show that diffusion modeling, large model size, and large data size all contribute to superior performance.
5. 包括aloha在内的方法都是task specified， 双手操作没有好办法
6. 直接使用大模型将动作离散的方法精度不行
7. 小模型和基于规则的方法都没有泛化性
8. 使用多种单臂的数据只是为了增强双臂的操作能力而不是给多个机器人用
## Challenge
### model
1. powerful architecture ->***RDT**
    - diffsuion **expressiveness**, transformer multimodal inputs **scalability**

2. 异质的机器人数据（heterogeneous data.）->proposing a **physically interpretable unified action space** to unify various robot action spaces
    - `之前要么找个像数据集的robot，要么筛跟robot一样的数据；`
    - 多模态输入：把low_dim, image, language三个obs 模态编码后cat起来作为diffsion model的输入
    - 多种类的机器人数据：做一个大的robot_state,对于特定机器人有就填，没有就pad
### data
1. 大的robot_state
2. rewrite instruction with GPT4-Turbo

# meeting 
*train config*:16batch, 200epoch, 4min/epoch, 150epochs/10h, 1500epochs/100h
*data config*: img_size(256,256,3), 2obs->16actions, 14000steps, 80episodes, 10hz, 
![alt text](<2024-10-24 21-13-13 的屏幕截图.png>)

# suffer:
1. 显卡不行
2. 采集的数据不是严格的30hz，帧差0.032-0.036不等， 
