# SwiftThief
**Jeonghyun Lee, Sungmin Han, Sangkyun Lee**

**Korea University (AIRLAB)**

Model-stealing attacks are emerging as a severe threat to AI-based services because an adversary can create models that duplicate the functionality of the black-box AI models inside the services with regular query-based access. To avoid detection or query costs, the model-stealing adversary must consider minimizing the number of queries to obtain an accurate clone model. To achieve this goal, we propose SwiftThief, a novel model-stealing framework that utilizes both queried and unqueried data to reduce query complexity. In particular, SwiftThief uses contrastive learning, a recent technique for representation learning. We formulate a new objective function for model stealing consisting of self-supervised (for abundant unqueried inputs from public datasets) and soft-supervised (for queried inputs) contrastive losses, jointly optimized with an output matching loss (for queried inputs). In addition, we suggest a new sampling strategy to prioritize rarely queried classes to improve attack performance. Our experiments proved that SwiftThief could significantly enhance the efficiency of model-stealing attacks compared to the existing methods, achieving similar attack performance using only half of the query budgets of the competing approaches. Also, SwiftThief showed high competence even when a defense was activated for the victims.

## Prepare

### Download Surrogate Dataset 


### Download Victim Models

For preparing vicitm models, plz download below files and store it to `save/victim/{DATASET}/model.pt`.

(e.g. save/victim/cifar10/model.pt)

- CIFAR-10 : [save/victim/cifar10/model.pt](https://drive.google.com/file/d/1VogchHb9bmaNqXRZA4KpE1yXiuPY5gc4/view?usp=sharing)
- EuroSAT : [save/victim/eurosat/model.pt](https://drive.google.com/file/d/1Yh4qU7QxwItAv6bJPOG2PbI5-Qb9KfHj/view?usp=sharing)
- GTSRB : [save/victim/gtsrb/model.pt](https://drive.google.com/file/d/1hXlOWypX0vvfCcHtycExrOnHY-WuHiYI/view?usp=sharing)
- MNIST : [save/victim/mnist/model.pt](https://drive.google.com/file/d/1oT0j-s42ppdFn1KpeQJEXz7FhY0nYMMf/view?usp=sharing)
- SVHN : [save/victim/svhn/model.pt](https://drive.google.com/file/d/12WPj13A3XbBCs8K65Xe6SLXxaR_IeoWY/view?usp=sharing)


## Run
`bash scripts/swiftthief/{DATASET}.sh`

## Running Environment Details (For Exact Reproduction)

Based on Pytorch docker images (https://hub.docker.com/r/pytorch/pytorch):
- **OS**: UBUNTU 22.04.5 LTS
- **NVIDIA RTX** A5000
- **Driver Version**: 535.104.05   
- **CUDA Version**: 12.2
- **Python**: 3.10.11

**Key libraries**:
```
torch==2.0.1
torchvision==0.15.2
scikit-learn==1.3.2
scipy==1.11.3
```

## Ciitation
```
@inproceedings{ijcai2024p47,
  title     = {SwiftThief: Enhancing Query Efficiency of Model Stealing by Contrastive Learning},
  author    = {Lee, Jeonghyun and Han, Sungmin and Lee, Sangkyun},
  year      = {2024},
  booktitle = {IJCAI},
}
```