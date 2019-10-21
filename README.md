# Learn-to-pay-attention

Reproduce ICLR18 paper "Learn to Pay Attention"

The modified network archtecture(based on VGG19) is shown below,

![image](https://github.com/iversonicter/Learn-to-pay-attention/blob/master/img/vgg-att.png)

The model follow the SaoYan's archtecture, a little different from original VGG19-BN.

# Requirements

- Pytorch > 1.0
- Tensorboard

Please install corresponding libraries("pip install XXX") according to the hint "ImportError: No module named XXX" 

# Training/Test Settings and Visualization

Experimental settings are listed here, without intensive tuning and sophisticated tricks:

- VGG19
- SGD with initial learning rate 0.1
- 100 epoches, with lr decay by 0.1 at [50, 70, 90] epoch
- Batch size 64

The accuracy/loss on test set during training stage are listed here


![image](https://github.com/iversonicter/Learn-to-pay-attention/blob/master/img/Loss.png)

![image](https://github.com/iversonicter/Learn-to-pay-attention/blob/master/img/Acc.png)

# Results

The experimental results demonstrate that with Attention module, the model can achieve better results. 
Repeating three times eliminates the random initialization problems.


| Model        | Dataset        | Top-1 Error  | Top-5 Error | Epoch  | Shared u |
| -------------|:--------------:|:------------:|:-----------:|:-------|:---------|
| VGG19        | CIFAR-100      | 26.2%        | 7.65%       | 100    |N.A.      |
| VGG19        | CIFAR-100      | 26.84%       | 7.53%       | 100    |N.A.      |
| VGG19        | CIFAR-100      | 27.39%       | 7.47%       | 100    |N.A.      |
| VGG19-att,softmax| CIFAR-100      | 28.2%        | 7.63%       | 100    |No       |
| VGG19-att,softmax| CIFAR-100      | 27.17%       | 7.49%       | 100    |No       |
| VGG19-att,softmax| CIFAR-100      | 27.29%       | 7.57%       | 100    |No       |
| VGG19-att,sigmiod| CIFAR-100      | 25.2%        | 6.52%       | 100    |No       |
| VGG19-att,sigmoid| CIFAR-100      | 25.94%       | 6.71%       | 100    |No       |
| VGG19-att,sigmoid| CIFAR-100      | 25.24%       | 6.75%       | 100    |No       |



Notices: 
- The VGG19 without attention can achieve higher results than report in original VGG Paper.
- The training epoches in most experiments are 100, the highest one is 300.
- The parameter u in parametersed compatibility(pc) is not shared between 3 attention layers

I found the shared U may perform better than seperate ones?

# References

Github repo: https://github.com/SaoYan/LearnToPayAttention

Blog: https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79

The repo and blog are helpful to understand this paper. Recommand strongly!!


If these repo is useful, please cite these papers below.

```
@article{DBLP:journals/corr/abs-1804-02391,
  author    = {Saumya Jetley and
               Nicholas A. Lord and
               Namhoon Lee and
               Philip H. S. Torr},
  title     = {Learn To Pay Attention},
  journal   = {CoRR},
  volume    = {abs/1804.02391},
  year      = {2018},
  url       = {http://arxiv.org/abs/1804.02391},
  archivePrefix = {arXiv},
  eprint    = {1804.02391},
  timestamp = {Mon, 13 Aug 2018 16:47:20 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1804-02391},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```
@article{DBLP:journals/corr/abs-1808-08114,
  author    = {Jo Schlemper and
               Ozan Oktay and
               Michiel Schaap and
               Mattias P. Heinrich and
               Bernhard Kainz and
               Ben Glocker and
               Daniel Rueckert},
  title     = {Attention Gated Networks: Learning to Leverage Salient Regions in
               Medical Images},
  journal   = {CoRR},
  volume    = {abs/1808.08114},
  year      = {2018},
  url       = {http://arxiv.org/abs/1808.08114},
  archivePrefix = {arXiv},
  eprint    = {1808.08114},
  timestamp = {Sun, 02 Sep 2018 15:01:56 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1808-08114},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```
@article{DBLP:journals/corr/abs-1804-05338,
  author    = {Jo Schlemper and
               Ozan Oktay and
               Liang Chen and
               Jacqueline Matthew and
               Caroline L. Knight and
               Bernhard Kainz and
               Ben Glocker and
               Daniel Rueckert},
  title     = {Attention-Gated Networks for Improving Ultrasound Scan Plane Detection},
  journal   = {CoRR},
  volume    = {abs/1804.05338},
  year      = {2018},
  url       = {http://arxiv.org/abs/1804.05338},
  archivePrefix = {arXiv},
  eprint    = {1804.05338},
  timestamp = {Tue, 17 Sep 2019 14:15:15 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1804-05338},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

```
