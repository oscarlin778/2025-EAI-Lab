# Lab4 - Homework Template
### 1. About Knowledge Distillation (15%)
- What Modes of Distillation is used in this Lab ?
    
    這次作業使用的是 Offline Distillation。 先將一個 Teacher 模型 (ResNet50) 訓練到一定的準確度。在蒸餾階段，Teacher 模型的參數被凍結 (Frozen/Pre-trained)，不會進行更新，僅負責單向將知識傳遞給 Student 模型 (ResNet18)。
    
- What role do logits play in knowledge distillation? What effect does a higher temperature parameter have on logits conversion ?

    Logits 在神經網路的最後一層經過 Softmax 之前的原始輸出分數。它們包含了類別之間相似度的資訊，而不僅僅是最終的Hard Label。
    較高的 Temperature ($T$) 會讓 Softmax 產生的機率分佈變得更平滑。這會提升錯誤類別的機率值，也知道 Teacher 模型所學到的類別間的關聯性結構可以讓 Student Model 學習到比標準答案更多的細節。
    
- In Feature-Based Knowledge Distillation, from which parts of the Teacher model do we extract features for knowledge transfer?

    我們從 Intermediate Hidden Layers 提取特徵。 在作業中的 ResNet 架構，我們提取了 layer1、layer2、layer3 以及 layer4 (即四個主要的 Residual Blocks) 輸出的 Feature Maps，用來讓 Student Model 的中間層學習。

### 2. Response-Based KD (30%)

Please explain the following:
- How you choose the Temperature and alpha?


    我設定 Temperature ($T$) = 5.0 以及 alpha ($\alpha$) = 0.9。

    Temperature ($T$) 的影響：
    
    較高 ($T > 1$): 會讓 Softmax 輸出分佈變得更平滑。
        
        優點： 能放大錯誤類別的機率值
        缺點： 若 T 過高，機率分佈會趨近於完全均勻 (Uniform Distribution)，導致資訊量消失 (Maximum Entropy)，Student 學不到任何東西。
    較低 ($T \to 1$): 會讓機率分佈變得尖銳 (Sharper)。
        
        優點： 模型對預測結果更有信心。
        缺點： 輸出會變得像 Hard Labels (0 或 1)，微小的關聯性資訊被壓縮至接近 0，導致蒸餾失去意義。
        
        最終設定 T=5 是一個平衡點，既能保留大部分資訊，又不會讓雜訊過大。

    Alpha ($\alpha$) 的影響：
    
    較高 ($\alpha \to 1$, e.g., 0.9): 訓練主要由 Teacher 的 Soft label主導。
        
        優點： 當 Teacher (ResNet50) 遠強於 Student (ResNet18) 時，讓 Student 全力模仿 Teacher 的思考模式通常能獲得最佳的泛化效果。
        缺點： 若 Teacher 本身有錯誤判斷，Student 會跟著一起錯，且可能忽略 Ground Truth 的絕對正確性。
        
    較低 ($\alpha \to 0$, e.g., 0.1): 訓練主要由 Ground Truth Hard label 主導。
        
        優點： 確保 Student 至少能學會資料集原本的正確答案。
        缺點： 蒸餾的效果會變得微乎其微，Student 幾乎等於是 From Scratch 訓練，無法學習 Teacher 的能力。
        
        選擇理由： 由於本次實驗 Teacher 準確度高 (>90%)，因此我選擇較高的值，讓 Student 盡可能吸收 Teacher 的知識。

- How you design the loss function?  

    Loss Function是標準 Cross-Entropy Loss 與 KL Divergence Loss 的線性組合：$$L_{total} = (1 - \alpha) \cdot L_{CE}(p_s, y) + \alpha \cdot T^2 \cdot L_{KD}(p_s^\tau, p_t^\tau)$$

    1. Student Loss ($L_{CE}$): 使用 nn.CrossEntropyLoss 計算 Student Logits 與 Ground Truth 的差異。
    2. Distillation Loss ($L_{KD}$): 使用 nn.KLDivLoss 計算。Teacher 的 Logits 先除以 $T$ 再做 softmax。Student 的 Logits 先除以 $T$ 再做 log_softmax。
    3. Scaling ($T^2$): 蒸餾損失項乘上了 $T^2$。這是因為經過 $T$ 除法後的 Softmax 梯度量級會縮小約 $1/T^2$ 倍，乘回 $T^2$ 可以確保蒸餾梯度的量級與標準 Cross-Entropy 梯度相當，避免梯度失衡。


### 3. Feature-based KD (30%)

Please explain the following:
- How you extract features from the choosing intermediate layers?

    我修改了 ResNet 類別中的 forward 函式。 原本函式只回傳最後的分類結果，我將其改為回傳一個 Tuple：(logits, [f1, f2, f3, f4])。其中 list [f1, f2, f3, f4] 分別對應資料流經 self.layer1 至 self.layer4 後輸出的 Feature Maps，這樣就能在訓練迴圈中取得多層次的特徵表示。
- How you design the loss function?

    總損失函數定義為：$$L_{total} = (1 - \alpha) \cdot L_{CE} + \alpha \cdot L_{Feature}$$
    1. 維度對齊 (Connectors):由於 Teacher (ResNet50) 中間層的通道數 (Channels: 256, 512, 1024, 2048) 遠大於 Student (ResNet18) (Channels: 64, 128, 256, 512)，直接計算距離會導致維度不匹配。因此，在 Distiller 類別中做了 Connectors (1x1 卷積層)，將 Student 的 Feature Maps 進行升維投影，使其通道數一致。
    2. Loss 計算:使用 Mean Squared Error (MSE) Loss 來計算「投影後的 Student 特徵」與「Teacher 特徵」之間的距離。將四層的 MSE Loss 加總後作為特徵蒸餾的損失項。

### 4. Comparison of student models w/ & w/o KD (5%)

Provide results according to the following structure:
|                            | loss     | accuracy |
| -------------------------- | -------- | -------- |
| Teacher from scratch       | 0.46     | 89.84     |
| Student from scratch       | 0.51     | 87.50     |
| Response-based student     | 1.36     | 89.46     |
| Featured-based student     | 1.24     | 88.65    |

### 5. Implementation Observations and Analysis (20%)
Based on the comparison results above:
- Did any KD method perform unexpectedly? 

    在實作 Feature-Based Distillation 時，我遇到錯誤：RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
- What do you think are the reasons? 

    這部分可能是 Teacher 和 Student 模型雖然都已透過 .to(device) 搬移到了 GPU，但在 Distiller 類別的 __init__ 方法中，新建立的 Connector 層 (nn.ModuleList) 預設是在 CPU 上初始化的。當 GPU 上的 Student 模型在 forward pass 中將特徵傳給 CPU 上的 Connector 時，就發生了裝置不匹配的錯誤。

    解決方法：
    
    在初始化 Distiller 後，再寫一次呼叫 .to(device) 
    
    distiller_fe = Distiller(...).to(device)。確保新建立的 Connector 參數也被正確地搬移到 GPU 記憶體中，從而解決問題。
- If not, please share your observations during the implementation process, or what difficulties you encountered and how you solved them?

    解決上述問題後，兩種蒸餾方法相較於 "Student from scratch" 都有進步。 Feature-based distillation 雖然實作較複雜，需要處理維度對齊，但能透過對齊網路內部的特徵表示提供更深層的指導；而 Response-based distillation 計算成本較低，且能有效透過 Soft Targets 提升模型的泛化能力。總結來說 Teacher Model 有達到 Accuracy > 90% 是本次蒸餾成功的點。